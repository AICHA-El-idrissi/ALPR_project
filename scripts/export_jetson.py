#!/usr/bin/env python3
"""
Export des modèles vers TensorRT pour Jetson.

Le processus est volontairement coupé en 2 étapes séparées car un engine
TensorRT n'est PAS portable d'une machine à l'autre : il dépend de l'archi
GPU + de la version exacte de TensorRT/CUDA/cuDNN installée. Concrètement :

    Étape "onnx"   -> peut tourner n'importe où (PC de dev, CI, etc.)
                      MAIS ne concerne QUE les modèles Ultralytics (.pt).
    Étape "engine" -> DOIT tourner directement sur le Jetson,
                      avec le JetPack / TensorRT qui sera utilisé en prod.
                      Fonctionne pour N'IMPORTE QUEL .onnx.

OÙ L'ENGINE EST ÉCRIT
---------------------
C'est le `config.yaml` qui décide, pas ce script. `--section` nomme la
section de `models:` visée, `--precision` le niveau, et la destination est
lue dans son `paths.tensorrt` avec `{precision}` substitué :

    models.detector.paths.tensorrt = models/exported/tensorrt/yolo/{precision}/detector
    --section detector --precision fp16
        -> models/exported/tensorrt/yolo/fp16/detector/model.engine

Ainsi le pipeline trouve l'engine sans qu'on ait à le déplacer à la main, et
le chemin ne peut pas diverger du config. Le wrapper TensorRT cherche un
`*.engine` DANS ce dossier : la destination est un dossier, pas un fichier.

`--out` reste disponible pour écrire ailleurs, mais prévient alors que le
pipeline n'ira pas chercher l'engine là.

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
inputs, but no optimization profile has been defined. » C'est le cas du
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
    python3 scripts/export_jetson.py onnx --weights yolo11n_detect.pt --imgsz 640

    # SUR le Jetson : la destination vient du config.yaml
    python3 scripts/export_jetson.py engine --onnx yolov8n.onnx \
        --section detector --precision fp16
    python3 scripts/export_jetson.py engine \
        --onnx models/dfine_n/best_stg2_single.onnx \
        --section detector_dfine --precision fp16
    python3 scripts/export_jetson.py engine \
        --onnx models/paddleocr/paddleocr_rec_fixed.onnx \
        --section ocr_paddle --precision fp32 --shape x:1x3x48x320

    # Voir la commande trtexec sans rien construire (marche hors Jetson) :
    python3 scripts/export_jetson.py engine --onnx m.onnx --section ocr \
        --precision fp16 --dry-run
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PRECISIONS = ("fp32", "fp16", "int8")


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
    print(f"[onnx] le metadata.yaml écrit à côté porte la VRAIE imgsz "
          f"({imgsz}) ; c'est lui qui fait foi au chargement, pas le config.")
    return Path(out_path)


# ---------------------------------------------------------------------------
# Destination : lue dans le config, jamais devinée
# ---------------------------------------------------------------------------
def destination_depuis_config(config_path: str, section: str,
                              precision: str) -> Path:
    """Rend le DOSSIER où le pipeline ira chercher l'engine.

    Le wrapper TensorRT fait `path.glob("*.engine")` sur ce dossier, donc on
    rend un dossier et l'appelant y dépose `model.engine`.

    Lire le config plutôt que reconstruire le chemin à la main est ce qui
    garantit qu'un engine produit ici sera trouvé là-bas : les deux ne
    peuvent pas diverger puisqu'il n'y a qu'une source.
    """
    try:
        import yaml
    except ModuleNotFoundError:
        sys.exit("Le paquet `pyyaml` est requis pour lire le config.\n"
                 "    pip install pyyaml\n")

    chemin = Path(config_path)
    if not chemin.is_file():
        sys.exit(f"Config introuvable : {chemin}\n"
                 f"Lance depuis la racine du projet, ou passe --config, ou "
                 f"donne une destination explicite avec --out.")

    cfg = yaml.safe_load(chemin.read_text(encoding="utf-8")) or {}
    modeles = cfg.get("models") or {}
    if section not in modeles:
        sys.exit(f"Section `models.{section}` absente du config.\n"
                 f"Sections disponibles : {sorted(modeles)}")
    sec = modeles[section]
    if not isinstance(sec, dict):
        sys.exit(f"Section `models.{section}` vide (valeur nulle) : c'est "
                 f"presque toujours une erreur d'indentation dans le YAML.")

    paths = sec.get("paths") or {}
    if "tensorrt" not in paths or paths["tensorrt"] is None:
        # `null` veut dire « cet export n'existera pas ». Si on construit
        # quand même un engine, le pipeline ne le chargera jamais : autant le
        # dire ici plutôt que de le découvrir au lancement.
        sys.exit(
            f"`models.{section}.paths.tensorrt` est absent ou vaut null : le "
            f"pipeline n'ira jamais chercher d'engine pour ce modèle.\n"
            f"Renseigne ce chemin dans le config avant d'exporter, ou utilise "
            f"--out pour écrire ailleurs en connaissance de cause.")

    return Path(str(paths["tensorrt"]).format(precision=precision))


def valider_precision(precision: str, fp16: bool, int8: bool
                      ) -> Tuple[bool, bool]:
    """Fait correspondre --precision et les drapeaux, ou refuse.

    La version précédente de ce script déclarait `--fp16 default=True` : un
    engine demandé en fp32 sortait en fp16 sans que rien ne le signale, et
    il allait se ranger dans le dossier `fp32/`. Deux fichiers différents
    sous le même nom de précision, avec des latences qui ne se comparent
    plus. La précision est maintenant PILOTÉE par --precision, et un drapeau
    qui la contredit est une erreur, pas un réglage silencieux.
    """
    if precision == "fp32":
        if fp16 or int8:
            sys.exit("--precision fp32 avec --fp16 ou --int8 : "
                     "contradictoire.\n"
                     "Demande --precision fp16 ou --precision int8.")
        return False, False
    if precision == "fp16":
        if int8:
            sys.exit("--precision fp16 avec --int8 : contradictoire.\n"
                     "Pour de l'INT8 avec repli fp16, écris "
                     "--precision int8 --fp16.")
        return True, False
    # int8 : le repli fp16 est un CHOIX, pas un automatisme. Les couches que
    # TensorRT ne quantifie pas retombent en fp16 si --fp16 est donné, en
    # fp32 sinon.
    return fp16, True


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
    connues = [e.name for e in modele.graph.input
               if e.name not in initialiseurs]

    # Un --shape visant un nom inexistant serait ignoré en silence et
    # laisserait la vraie dimension dynamique non résolue.
    inconnues = [n for n in overrides if n not in connues]
    if inconnues:
        sys.exit(f"--shape vise des entrées absentes du graphe : {inconnues}\n"
                 f"Entrées réelles : {connues}")

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
    """Choisit --memPoolSize ou --workspace selon la version de trtexec.

    `--workspace` a disparu en TensorRT 10, `--memPoolSize` n'existe pas
    avant 8.4 : on lit l'aide plutôt que de supposer une version.
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


def resoudre_destination(out: Optional[str], section: Optional[str],
                         config_path: str, precision: str) -> Path:
    """Rend le CHEMIN DU FICHIER .engine à produire."""
    if out:
        chemin = Path(out)
        if chemin.is_dir() or not chemin.suffix:
            chemin = chemin / "model.engine"
        print("[engine] destination forcée par --out : le pipeline, lui, "
              "cherche l'engine dans le chemin déclaré au config. Vérifie "
              "qu'ils coïncident.")
        return chemin
    if section:
        return destination_depuis_config(config_path, section,
                                         precision) / "model.engine"
    sys.exit(
        "Il faut dire où écrire l'engine : --section <section du config> "
        "(recommandé), ou --out <chemin>.\n"
        "Sans ça l'engine atterrirait à côté du .onnx, et le pipeline ne "
        "l'y chercherait jamais.")


def build_engine(onnx_path: str, precision: str, fp16: bool, int8: bool,
                 calib_cache: Optional[str], workspace_mb: int,
                 out: Optional[str], section: Optional[str],
                 config_path: str, batch: int, shape_overrides: List[str],
                 dry_run: bool) -> Optional[Path]:
    onnx_file = Path(onnx_path)
    if not onnx_file.exists():
        sys.exit(f"ONNX introuvable : {onnx_file}")

    fp16, int8 = valider_precision(precision, fp16, int8)
    out_path = resoudre_destination(out, section, config_path, precision)

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
                "--precision int8 exige --calib-cache (cache de "
                "calibration).\n"
                "Sans calibrateur, trtexec construit quand même un engine "
                "avec des échelles arbitraires : il tourne, il est rapide, "
                "et il lit faux.\n"
                "La calibration doit par ailleurs utiliser LE MÊME "
                "prétraitement que l'inférence, et il diffère par modèle "
                "(letterbox pour le détecteur YOLO, étirement pour l'OCR par "
                "détection, hauteur fixe + normalisation [-1,1] pour "
                "PaddleOCR).")
        if not Path(calib_cache).is_file():
            sys.exit(f"Cache de calibration introuvable : {calib_cache}")
        cmd += ["--int8", f"--calib={calib_cache}"]

    # Sans profil, trtexec refuse un graphe à dimensions dynamiques :
    # « Network has dynamic or shape inputs, but no optimization profile
    # has been defined. »
    if dynamique:
        cmd += format_profile_flags(shapes)

    cmd.append(workspace_flag(trtexec, workspace_mb) if not dry_run
               else f"--memPoolSize=workspace:{workspace_mb}M")

    print(f"[engine] précision : {precision}"
          f"{' avec repli fp16' if int8 and fp16 else ''}")
    print(f"[engine] sortie    : {out_path}")
    print("[engine] commande  :", " ".join(str(c) for c in cmd))
    if dry_run:
        print("[engine] --dry-run : rien n'est construit.")
        return None

    # Créé APRÈS toutes les validations : un dossier vide laissé derrière une
    # commande qui échoue ferait croire à un export fait.
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"[engine] trtexec a échoué (code {result.returncode})")

    print(f"[engine] écrit -> {out_path}")
    if section:
        print(f"[engine] le pipeline le trouvera avec : --backend tensorrt "
              f"--precision {precision}")
    return out_path


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)

    p_onnx = sub.add_parser("onnx",
                            help="export .pt Ultralytics -> .onnx (portable)")
    p_onnx.add_argument("--weights", required=True,
                        help="chemin du .pt Ultralytics (yolov8*, yolo11*, rtdetr*)")
    p_onnx.add_argument("--imgsz", type=int, default=640)
    p_onnx.add_argument("--model-type", choices=["auto", "yolo", "rtdetr"],
                        default="auto")
    p_onnx.add_argument("--opset", type=int, default=12)
    p_onnx.add_argument("--no-simplify", action="store_true")

    p_engine = sub.add_parser("engine",
                              help="export .onnx -> .engine (SUR le Jetson)")
    p_engine.add_argument("--onnx", required=True, help="chemin du .onnx")
    p_engine.add_argument("--section", default=None, metavar="SECTION",
                          help="section de models: qui donne la destination "
                               "(detector, detector_dfine, ocr, ocr_paddle)")
    p_engine.add_argument("--config", default="configs/config.yaml")
    p_engine.add_argument("--precision", default="fp16", choices=PRECISIONS,
                          help="pilote la précision ET le dossier de sortie")
    p_engine.add_argument("--fp16", action="store_true",
                          help="avec --precision int8 : autorise le repli fp16")
    p_engine.add_argument("--int8", action="store_true",
                          help="redondant avec --precision int8")
    p_engine.add_argument("--calib-cache", default=None)
    p_engine.add_argument("--workspace-mb", type=int, default=2048)
    p_engine.add_argument("--out", default=None,
                          help="écrire ailleurs que là où le config l'attend")
    p_engine.add_argument("--batch", type=int, default=1,
                          help="valeur donnée aux dimensions de lot dynamiques")
    p_engine.add_argument("--shape", action="append", default=[],
                          metavar="NOM:1x3x48x320",
                          help="fixe la shape d'une entrée (répétable)")
    p_engine.add_argument("--dry-run", action="store_true",
                          help="affiche la commande sans construire "
                               "(marche hors Jetson)")

    args = ap.parse_args()

    if args.stage == "onnx":
        export_onnx(args.weights, args.imgsz, args.model_type, args.opset,
                    not args.no_simplify)
    elif args.stage == "engine":
        build_engine(args.onnx, args.precision, args.fp16, args.int8,
                     args.calib_cache, args.workspace_mb, args.out,
                     args.section, args.config, args.batch, args.shape,
                     args.dry_run)


if __name__ == "__main__":
    main()
