#!/usr/bin/env python3
"""Fige les axes dynamiques d'un graphe ONNX, et VERIFIE que rien n'a bouge.

« Figer » ne repare rien et ne change aucun poids : cela remplace les
dimensions symboliques (lot, largeur...) par des nombres, puis relance
l'inference de formes et le repliement des constantes. Les chaines
Shape -> Gather -> Concat -> Reshape, qui calculaient une forme a l'execution,
s'effondrent alors en constantes litterales.

C'est indispensable pour ncnn, dont le graphe porte des formes concretes
couche par couche et n'a aucune notion de dimension symbolique.

    python3 scripts/fix_onnx_shapes.py \\
        --onnx  models/paddleocr/paddleocr_recv2.onnx \\
        --out   models/paddleocr/paddleocr_recv2_fixed.onnx \\
        --shape 1,3,48,320

Le script refuse de produire un fichier si les sorties du graphe fige ne
reproduisent pas celles de l'original : un figeage mal fait peut changer le
calcul sans rien signaler.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

try:
    import onnx
except ModuleNotFoundError:
    sys.exit("Le paquet `onnx` est requis.\n\n    pip install onnx\n")
try:
    import onnxruntime as ort
except ModuleNotFoundError:
    sys.exit("Le paquet `onnxruntime` est requis.\n\n    pip install onnxruntime\n")


def dims(proto) -> list:
    """Rend les dimensions d'une entree/sortie : un entier, ou son nom si elle
    est symbolique."""
    return [d.dim_param if d.dim_param else d.dim_value
            for d in proto.type.tensor_type.shape.dim]


def est_dynamique(d) -> bool:
    return isinstance(d, str) or d == 0


def decrire(modele, titre: str) -> int:
    """Affiche la signature et rend le nombre d'axes dynamiques restants."""
    n = 0
    print(f"  {titre}")
    for p in list(modele.graph.input) + list(modele.graph.output):
        d = dims(p)
        n += sum(est_dynamique(x) for x in d)
        role = "entree" if p in modele.graph.input else "sortie"
        print(f"    {role} {p.name:<16} {d}")
    return n


def compter_ops(modele) -> dict:
    from collections import Counter
    return dict(Counter(node.op_type for node in modele.graph.node))


def figer_par_onnxsim(src: Path, dst: Path, nom_entree: str, forme) -> bool:
    """Voie principale : onnxsim fige et replie en une passe."""
    forme_txt = ",".join(str(v) for v in forme)
    cmd = [sys.executable, "-m", "onnxsim", str(src), str(dst),
           "--overwrite-input-shape", f"{nom_entree}:{forme_txt}"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return res.returncode == 0 and dst.exists() and dst.stat().st_size > 0


def figer_a_la_main(src: Path, dst: Path, nom_entree: str, forme) -> bool:
    """Repli si onnxsim est absent : on ecrit les dimensions dans le graphe et
    on relance l'inference de formes. Moins efficace (les chaines Shape ne sont
    pas toutes repliees), mais suffisant pour beaucoup de graphes."""
    modele = onnx.load(str(src))
    for entree in modele.graph.input:
        if entree.name != nom_entree:
            continue
        axes = entree.type.tensor_type.shape.dim
        if len(axes) != len(forme):
            print(f"  [!] l'entree a {len(axes)} axes, --shape en donne {len(forme)}")
            return False
        for axe, valeur in zip(axes, forme):
            axe.ClearField("dim_param")
            axe.dim_value = int(valeur)
    # Les formes de sortie deviennent fausses si on les garde : on les efface
    # pour laisser l'inference les recalculer.
    for sortie in modele.graph.output:
        sortie.type.tensor_type.ClearField("shape")
    modele = onnx.shape_inference.infer_shapes(modele, strict_mode=False)
    onnx.save(modele, str(dst))
    return True


def sortie(chemin: Path, x: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(str(chemin), providers=["CPUExecutionProvider"])
    nom = sess.get_inputs()[0].name
    return np.squeeze(sess.run(None, {nom: x})[0])


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shape", default="1,3,48,320",
                    help="forme NCHW a figer, ex. 1,3,48,320")
    ap.add_argument("--seuil", type=float, default=1e-5,
                    help="ecart maximal tolere entre original et graphe fige")
    ap.add_argument("--n-tests", type=int, default=3)
    args = ap.parse_args()

    if not args.onnx.exists():
        sys.exit(f"Introuvable : {args.onnx}")
    forme = tuple(int(v) for v in args.shape.split(","))

    original = onnx.load(str(args.onnx))
    nom_entree = original.graph.input[0].name
    print(f"AVANT  ({args.onnx.name})")
    n_avant = decrire(original, f"opset {original.opset_import[0].version}, "
                                f"{len(original.graph.node)} noeuds")
    ops_avant = compter_ops(original)
    print(f"    axes dynamiques : {n_avant}")
    if n_avant == 0:
        print("\n  Ce graphe n'a aucun axe dynamique : il est deja fige.")
        if args.onnx.resolve() != args.out.resolve():
            shutil.copy(args.onnx, args.out)
            print(f"  Copie vers {args.out}")
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        provisoire = Path(tmp) / "fige.onnx"
        if figer_par_onnxsim(args.onnx, provisoire, nom_entree, forme):
            methode = "onnxsim"
        else:
            print("\n  [!] onnxsim indisponible ou en echec "
                  "(pip install onnxsim) — repli sur le figeage manuel.")
            if not figer_a_la_main(args.onnx, provisoire, nom_entree, forme):
                sys.exit("  Le figeage a echoue.")
            methode = "manuel"

        fige = onnx.load(str(provisoire))
        print(f"\nAPRES  (methode : {methode})")
        n_apres = decrire(fige, f"opset {fige.opset_import[0].version}, "
                                f"{len(fige.graph.node)} noeuds")
        ops_apres = compter_ops(fige)
        print(f"    axes dynamiques : {n_apres}")

        disparus = {k: v for k, v in ops_avant.items() if k not in ops_apres}
        if disparus:
            print(f"    operateurs elimines : {disparus}")

        # ------------------------------------------------------------------
        # Verification : le graphe fige doit calculer LA MEME CHOSE.
        # Un figeage mal fait change le calcul sans lever d'erreur.
        # ------------------------------------------------------------------
        print(f"\nVERIFICATION ({args.n_tests} tirages)")
        pire = 0.0
        for k in range(args.n_tests):
            x = np.random.default_rng(k).standard_normal(forme).astype(np.float32)
            try:
                ref, got = sortie(args.onnx, x), sortie(provisoire, x)
            except Exception as exc:
                sys.exit(f"  Execution impossible : {exc}")
            if ref.shape != got.shape:
                sys.exit(f"  Formes de sortie differentes : {ref.shape} vs {got.shape}")
            ecart = float(np.abs(ref - got).max())
            pire = max(pire, ecart)
            accord = float((ref.argmax(-1) == got.argmax(-1)).mean())
            print(f"  tirage {k} : ecart={ecart:.2e}  argmax={accord:.0%}")

        print("\n" + "=" * 70)
        if pire > args.seuil:
            print(f"ECHEC — ecart {pire:.2e} au-dessus du seuil {args.seuil:.0e}.\n"
                  f"Le graphe fige ne calcule pas la meme chose que l'original.\n"
                  f"Aucun fichier n'a ete ecrit.")
            return 1
        if n_apres > 0:
            print(f"ATTENTION — il reste {n_apres} axe(s) dynamique(s).\n"
                  f"pnnx acceptera peut-etre quand meme (il fige aux valeurs de\n"
                  f"inputshape=), mais verifie la sortie de verif_ncnn.py.")

        args.out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(provisoire, args.out)
        avant_mo = args.onnx.stat().st_size / 1e6
        apres_mo = args.out.stat().st_size / 1e6
        print(f"ECRIT  {args.out}  ({avant_mo:.1f} Mo -> {apres_mo:.1f} Mo)")
        print(f"ecart maximal original/fige : {pire:.2e}  (seuil {args.seuil:.0e})")
        print("\nEtape suivante :")
        print(f"  pnnx {args.out} inputshape=[{','.join(str(v) for v in forme)}] fp16=0 \\")
        print(f"       ncnnparam=<dossier>/model.ncnn.param ncnnbin=<dossier>/model.ncnn.bin")
        print("  (cree le dossier AVANT : pnnx rend 0 meme quand il n'ecrit rien)")
    return 0


if __name__ == "__main__":
    sys.exit(main())