#!/usr/bin/env python3
"""
Export des modèles vers TensorRT pour Jetson Orin NX.

Le processus est volontairement coupé en 2 étapes séparées car un engine
TensorRT n'est PAS portable d'une machine à l'autre : il dépend de l'archi
GPU + de la version exacte de TensorRT/CUDA/cuDNN installée. Concrètement :

    Étape "onnx"   -> peut tourner n'importe où (PC de dev, CI, etc.)
                      MAIS ne concerne QUE les modèles Ultralytics (.pt).
    Étape "engine" -> DOIT tourner directement sur le Jetson Orin NX,
                      avec le JetPack / TensorRT qui sera utilisé en prod.
                      Fonctionne pour N'IMPORTE QUEL .onnx.

MODÈLES NON-ULTRALYTICS (D-FINE, PaddleOCR)
-------------------------------------------
L'étape "onnx" passe par `ultralytics.YOLO` / `ultralytics.RTDETR`, qui ne
savent charger qu'un checkpoint Ultralytics. Un modèle D-FINE (dépôt
officiel Peterande/D-FINE) ou PaddleOCR (Paddle2ONNX) n'est PAS chargeable
ainsi -- même si son nom contient « rtdetr » et même si D-FINE est bâti sur
RT-DETR : ce sont des implémentations différentes.

Pour ces modèles, on saute l'étape "onnx" : l'ONNX existe déjà, on passe
directement à "engine".

PROFILS DE SHAPES
-----------------
Un ONNX à dimensions dynamiques ne peut pas être construit sans profil
d'optimisation : trtexec s'arrête sur « Network has dynamic or shape
inputs, but no optimization profile has been defined ». C'est le cas du
D-FINE exporté (`images: [N,3,640,640]`, `orig_target_sizes: [?,2]`) et
d'un PaddleOCR non figé (`x: [?,3,48,?]`).

Ce script LIT les entrées du graphe et construit les profils tout seul,
plutôt que de les supposer : les dimensions statiques sont reprises telles
quelles, la dimension de lot est résolue par --batch, et toute autre
dimension dynamique doit être donnée explicitement par --shape. Une
dimension non résolue provoque une erreur qui nomme l'entrée et l'axe
concernés, au lieu d'un échec obscur de trtexec.

Usage :
    # Modèle Ultralytics, sur le PC de dev :
    python scripts/export_jetson.py onnx --weights yolo11n_detect.pt --imgsz 640

    # SUR le Jetson, pour n'importe quel ONNX :
    python scripts/export_jetson.py engine --onnx yolo11n_detect.onnx --fp16
    python scripts/export_jetson.py engine --onnx models/dfine_n/best_stg2_single.onnx --fp16
    python scripts/export_jetson.py engine --onnx models/paddleocr/paddleocr_rec.onnx \
        --shape x:1x3x48x320 --fp16

    
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Étape 1 : export vers ONNX (portable) -- ULTRALYTICS UNIQUEMENT
# ---------------------------------------------------------------------------
def export_onnx(weights: str, imgsz: int, model_type: str, opset: int,
                simplify: bool) -> Path:
    # Ce contrôle passe AVANT l'import d'ultralytics : un .onnx donné ici est
    # une erreur d'usage, et le message doit l'expliquer même sur une machine
    # où ultralytics n'est pas installé.
    weights_path = Path(weights)
    if weights_path.suffix.lower() != ".pt":
        sys.exit(
            f"{weights} n'est pas un checkpoint Ultralytics (.pt).\n"
            f"Un modèle D-FINE ou PaddleOCR est DÉJÀ en ONNX : passe "
            f"directement à l'étape 'engine'.")

    try:
        from ultralytics import YOLO, RTDETR
    except ImportError:
        sys.exit("ultralytics manquant. Installe avec: pip install ultralytics")

    name = weights_path.stem.lower()
    if model_type == "auto":
        model_type = "rtdetr" if "rtdetr" in name else "yolo"

    print(f"[onnx] chargement {weights} (type={model_type})")
    model = RTDETR(weights) if model_type == "rtdetr" else YOLO(weights)

    # simplify=True réduit le graphe ONNX (utile pour trtexec), dynamic=False
    # car on veut une shape fixe -> engine plus rapide et plus simple à builder.
    out_path = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        simplify=simplify,
        dynamic=False,
        half=False,  # la conversion fp16 se fait à l'étape engine, pas ici
    )
    print(f"[onnx] écrit -> {out_path}")
    return Path(out_path)


# ---------------------------------------------------------------------------
# Profils de shapes : lus dans le graphe, pas supposés
# ---------------------------------------------------------------------------
def _parse_shape_overrides(valeurs: List[str]) -> Dict[str, List[int]]:
    """--shape x:1x3x48x320 -> {"x": [1, 3, 48, 320]}"""
    out: Dict[str, List[int]] = {}
    for v in valeurs or []:
        if ":" not in v:
            sys.exit(f"--shape mal formé : {v!r}. Attendu : nom:1x3x48x320")
        nom, dims = v.rsplit(":", 1)
        try:
            out[nom] = [int(d) for d in dims.lower().split("x")]
        except ValueError:
            sys.exit(f"--shape mal formé : {v!r}. Attendu : nom:1x3x48x320")
    return out


def resolve_input_shapes(onnx_path: Path, batch: int,
                         overrides: Dict[str, List[int]]
                         ) -> Tuple[Dict[str, List[int]], bool]:
    """Lit les entrées du graphe -> shapes concrètes + « y avait-il du dynamique ».

    Règles, dans l'ordre :
      1. une shape donnée par --shape gagne toujours ;
      2. une dimension statique est reprise telle quelle ;
      3. la dimension 0 (lot) dynamique est résolue par --batch ;
      4. toute autre dimension dynamique -> erreur nommant l'entrée et l'axe.

    La règle 4 est volontairement stricte : deviner une largeur d'entrée
    donnerait un engine qui se construit et qui lit faux.
    """
    try:
        import onnx
    except ModuleNotFoundError:
        sys.exit("Le paquet `onnx` est requis pour lire les shapes.\n"
                 "    pip install onnx\n")

    modele = onnx.load(str(onnx_path), load_external_data=False)
    initialiseurs = {i.name for i in modele.graph.initializer}

    shapes: Dict[str, List[int]] = {}
    dynamique = False

    for entree in modele.graph.input:
        if entree.name in initialiseurs:      # poids déclarés en entrée
            continue
        if entree.name in overrides:
            shapes[entree.name] = overrides[entree.name]
            dynamique = True                  # un override implique du dynamique
            continue

        dims: List[int] = []
        for axe, d in enumerate(entree.type.tensor_type.shape.dim):
            if d.HasField("dim_value") and d.dim_value > 0:
                dims.append(int(d.dim_value))
                continue
            dynamique = True
            if axe == 0:
                dims.append(batch)            # dimension de lot
                continue
            sys.exit(
                f"L'entrée {entree.name!r} a une dimension dynamique sur "
                f"l'axe {axe} que ce script ne peut pas deviner.\n"
                f"Donne-la explicitement, par exemple :\n"
                f"    --shape {entree.name}:{'x'.join(str(x) for x in dims)}x<taille>...\n"
                f"(Deviner une taille d'entrée produirait un engine qui se "
                f"construit sans erreur et qui lit faux.)")
        shapes[entree.name] = dims

    if not shapes:
        sys.exit(f"Aucune entrée trouvée dans {onnx_path}")
    return shapes, dynamique


def format_profile_flags(shapes: Dict[str, List[int]]) -> List[str]:
    """{"images": [1,3,640,640]} -> --minShapes=... --optShapes=... --maxShapes=..."""
    spec = ",".join(f"{nom}:{'x'.join(str(d) for d in dims)}"
                    for nom, dims in shapes.items())
    return [f"--minShapes={spec}", f"--optShapes={spec}", f"--maxShapes={spec}"]


# ---------------------------------------------------------------------------
# Étape 2 : ONNX -> engine TensorRT (doit tourner sur le Jetson)
# ---------------------------------------------------------------------------
def find_trtexec() -> str:
    candidates = [
        shutil.which("trtexec"),
        "/usr/src/tensorrt/bin/trtexec",
        "/usr/bin/trtexec",
    ]
    for c in candidates:
        if c and Path(c).exists():
            return c
    sys.exit(
        "trtexec introuvable. Sur Jetson il est normalement dans "
        "/usr/src/tensorrt/bin/trtexec (fourni par JetPack). "
        "Vérifie ton install JetPack ou ajoute-le au PATH."
    )


def workspace_flag(trtexec: str, workspace_mb: int) -> str:
    """Choisit --memPoolSize .
    """
    try:
        res = subprocess.run([trtexec, "--help"], capture_output=True,
                             text=True, timeout=60)
        aide = (res.stdout or "") + (res.stderr or "")
    except (OSError, subprocess.SubprocessError):
        aide = ""

    if "memPoolSize" in aide:
        return f"--memPoolSize=workspace:{workspace_mb}M"
    if "--workspace" in aide:
        return f"--workspace={workspace_mb}"
    print("[engine] aide de trtexec illisible — on suppose --memPoolSize "
          "(syntaxe TensorRT >= 8.4).")
    return f"--memPoolSize=workspace:{workspace_mb}M"


def build_engine(onnx_path: str, fp16: bool, int8: bool,
                 calib_cache: Optional[str], workspace_mb: int,
                 out: Optional[str], batch: int, shape_overrides: List[str],
                 dry_run: bool) -> Optional[Path]:
    onnx_file = Path(onnx_path)
    if not onnx_file.exists():
        sys.exit(f"ONNX introuvable : {onnx_file}")
    out_path = Path(out) if out else onnx_file.with_suffix(".engine")

    shapes, dynamique = resolve_input_shapes(
        onnx_file, batch, _parse_shape_overrides(shape_overrides))
    print("[engine] entrées résolues :")
    for nom, dims in shapes.items():
        print(f"           {nom} : {dims}")

    trtexec = "trtexec" if dry_run else find_trtexec()
    cmd = [trtexec, f"--onnx={onnx_file}", f"--saveEngine={out_path}"]

    if fp16:
        cmd.append("--fp16")
    if int8:
        if not calib_cache:
            sys.exit(
                "--int8 nécessite --calib-cache (fichier de cache de calibration).\n"
                "Attention : la calibration doit utiliser LE MÊME prétraitement "
                "que l'inférence, et il diffère par modèle (letterbox pour le "
                "détecteur YOLO, étirement pour l'OCR par détection, "
                "hauteur fixe + normalisation [-1,1] pour PaddleOCR).")
        cmd += ["--int8", f"--calib={calib_cache}"]

    # Sans profil, trtexec refuse un graphe à dimensions dynamiques :
    # « Network has dynamic or shape inputs, but no optimization profile
    # has been defined. »
    if dynamique:
        cmd += format_profile_flags(shapes)

    cmd.append(workspace_flag(trtexec, workspace_mb) if not dry_run
               else f"--memPoolSize=workspace:{workspace_mb}M")

    print("[engine] commande:", " ".join(cmd))
    if dry_run:
        print("[engine] --dry-run : rien n'est construit.")
        return None

    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"[engine] trtexec a échoué (code {result.returncode})")

    print(f"[engine] écrit -> {out_path}")
    return out_path


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)

    p_onnx = sub.add_parser("onnx", help="export .pt Ultralytics -> .onnx (portable)")
    p_onnx.add_argument("--weights", required=True,
                        help="chemin du .pt Ultralytics (yolov8*, yolo11*, rtdetr*)")
    p_onnx.add_argument("--imgsz", type=int, default=640)
    p_onnx.add_argument("--model-type", choices=["auto", "yolo", "rtdetr"], default="auto")
    p_onnx.add_argument("--opset", type=int, default=12)
    p_onnx.add_argument("--no-simplify", action="store_true")

    p_engine = sub.add_parser("engine",
                              help="export .onnx -> .engine (SUR le Jetson)")
    p_engine.add_argument("--onnx", required=True, help="chemin du .onnx")
    p_engine.add_argument("--fp16", action="store_true", default=True)
    p_engine.add_argument("--no-fp16", dest="fp16", action="store_false")
    p_engine.add_argument("--int8", action="store_true")
    p_engine.add_argument("--calib-cache", default=None)
    p_engine.add_argument("--workspace-mb", type=int, default=2048)
    p_engine.add_argument("--out", default=None)
    p_engine.add_argument("--batch", type=int, default=1,
                          help="valeur donnée aux dimensions de lot dynamiques")
    p_engine.add_argument("--shape", action="append", default=[],
                          metavar="NOM:1x3x48x320",
                          help="fixe la shape d'une entrée (répétable)")
    p_engine.add_argument("--dry-run", action="store_true",
                          help="affiche la commande sans construire (marche hors Jetson)")

    args = ap.parse_args()

    if args.stage == "onnx":
        export_onnx(args.weights, args.imgsz, args.model_type, args.opset,
                    not args.no_simplify)
    elif args.stage == "engine":
        build_engine(args.onnx, args.fp16, args.int8, args.calib_cache,
                     args.workspace_mb, args.out, args.batch, args.shape,
                     args.dry_run)


if __name__ == "__main__":
    main()