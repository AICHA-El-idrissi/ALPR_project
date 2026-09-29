#!/usr/bin/env python3
"""V
alide un portage ncnn en comparant ses sorties a celles du graphe ONNX.
\

    python3 scripts/verif_ncnn.py \\
        --onnx        models/paddleocr/paddleocr_rec_fixed.onnx \\
        --ncnn-param  models/exported/ncnn/paddle_rec/model.ncnn.param \\
        --ncnn-bin    models/exported/ncnn/paddle_rec/model.ncnn.bin \\
        --shape 1,3,48,320 \\
        --crops data/crops/            \

Sans --crops, la comparaison se fait sur du bruit gaussien : cela valide la
CONVERSION (memes poids, memes operateurs) mais pas le comportement sur des
donnees reelles. Avec --crops, on mesure en plus l'accord des sequences CTC
reellement lues, qui est le seul chiffre qui compte en exploitation.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import ncnn
except ModuleNotFoundError:
    sys.exit("Le paquet `ncnn` est requis.\n\n    pip install ncnn\n")
try:
    import onnxruntime as ort
except ModuleNotFoundError:
    sys.exit("Le paquet `onnxruntime` est requis.\n\n    pip install onnxruntime\n")


# ---------------------------------------------------------------------------
# Pretraitement : identique a celui de l'inference (
    
# ---------------------------------------------------------------------------
def preprocess_rec(img: np.ndarray, hauteur: int, largeur_max: int) -> np.ndarray:
    """Hauteur FIXE, ratio conserve, completion a DROITE, plage [-1,1]."""
    import cv2
    h, w = img.shape[:2]
    largeur = max(1, min(int(np.ceil(hauteur * (w / float(h)))), largeur_max))
    interp = cv2.INTER_AREA if largeur < w else cv2.INTER_LINEAR
    redim = cv2.resize(img, (largeur, hauteur), interpolation=interp)
    rgb = (redim[:, :, ::-1].astype(np.float32) / 255.0 - 0.5) / 0.5
    out = np.zeros((1, 3, hauteur, largeur_max), np.float32)
    out[0, :, :, :largeur] = rgb.transpose(2, 0, 1)
    return out


# ---------------------------------------------------------------------------
# Moteurs
# ---------------------------------------------------------------------------
def sortie_onnx(session, x: np.ndarray) -> np.ndarray:
    nom = session.get_inputs()[0].name
    return np.squeeze(session.run(None, {nom: x})[0])


def sortie_ncnn(param: str, bin_: str, x: np.ndarray,
                entree: str, sortie: str, fp16: bool) -> np.ndarray:
    """Execute le reseau ncnn sur une entree NCHW.

    DEUX PIEGES DE L'API PYTHON, tous deux silencieux :

    1. ``ncnn.Mat(tableau)`` ENVELOPPE le tampon numpy sans le posseder. Si on
       lui passe un temporaire (``ncnn.Mat(x[0].copy())``), celui-ci est libere
       des la construction du Mat : le reseau lit alors de la memoire morte et
       rend des valeurs aberrantes (de l'ordre de 1e28) qu'on prendrait a tort
       pour une divergence de conversion. D'ou la reference vivante `tampon`.

    2. ``np.array(mat)`` rend une VUE sur la memoire du blob, pas une copie.
       Si l'extracteur ou le Net meurt avant la lecture, meme symptome. D'ou le
       ``.copy()`` explicite, extracteur encore vif.
    """
    net = ncnn.Net()
    if not fp16:
        net.opt.use_fp16_packed = False
        net.opt.use_fp16_storage = False
        net.opt.use_fp16_arithmetic = False
    if net.load_param(param) != 0:
        sys.exit(f"ncnn n'a pas pu lire le .param : {param}")
    if net.load_model(bin_) != 0:
        sys.exit(f"ncnn n'a pas pu lire le .bin : {bin_}")

    tampon = np.ascontiguousarray(x[0], dtype=np.float32)   # reference vivante
    ex = net.create_extractor()
    ex.input(entree, ncnn.Mat(tampon))
    code, mat = ex.extract(sortie)
    if code != 0:
        sys.exit(f"ncnn extract({sortie!r}) a echoue (code {code}).\n"
                 f"Verifie les noms de blobs en tete et en fin du .param : "
                 f"pnnx nomme generalement l'entree « in0 » et la sortie « out0 ».")
    resultat = np.array(mat).copy()                          # copie avant destruction
    del ex, net, tampon
    return np.squeeze(resultat)


# ---------------------------------------------------------------------------
def aligner(ref: np.ndarray, got: np.ndarray):
    """ncnn peut rendre la matrice (T,C) transposee selon la derniere couche."""
    if ref.shape == got.shape:
        return ref, got, False
    if ref.T.shape == got.shape:
        return ref.T, got, True
    return None, None, False


def ctc_greedy(logits: np.ndarray, blank: int = 0) -> str:
    """Fusion des repetitions AVANT retrait du blanc."""
    idx, jetons, prec = logits.argmax(-1), [], -1
    for i in idx:
        i = int(i)
        if i == prec:
            continue
        prec = i
        if i != blank:
            jetons.append(str(i))
    return "-".join(jetons)


def comparer(nom, ref, got, seuil):
    ref, got, transpose = aligner(ref, got)
    if ref is None:
        return None
    ecart = float(np.abs(ref - got).max())
    echelle = max(float(np.abs(ref).max()), 1e-9)
    relatif = ecart / echelle
    argmax = float((ref.argmax(-1) == got.argmax(-1)).mean())
    seq_ref, seq_got = ctc_greedy(ref), ctc_greedy(got)
    print(f"  {nom:<22} ecart={ecart:.2e}  relatif={relatif:.2e}  "
          f"argmax={argmax:6.1%}  sequence={'identique' if seq_ref == seq_got else 'DIFFERENTE'}"
          + ("  [transposee]" if transpose else ""))
    return {"relatif": relatif, "argmax": argmax, "seq_ok": seq_ref == seq_got}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--ncnn-param", type=Path, required=True)
    ap.add_argument("--ncnn-bin", type=Path, required=True)
    ap.add_argument("--shape", default="1,3,48,320",
                    help="forme NCHW de l'entree, ex. 1,3,48,320")
    ap.add_argument("--input-blob", default="in0")
    ap.add_argument("--output-blob", default="out0")
    ap.add_argument("--crops", type=Path, default=None,
                    help="dossier d'imagettes de plaques reelles")
    ap.add_argument("--fp16", action="store_true",
                    help="laisse ncnn calculer en fp16 (defaut : fp32, pour "
                         "ne mesurer QUE l'erreur de conversion)")
    ap.add_argument("--seuil", type=float, default=1e-3,
                    help="ecart relatif maximal tolere")
    ap.add_argument("--n-alea", type=int, default=3)
    args = ap.parse_args()

    for f in (args.onnx, args.ncnn_param, args.ncnn_bin):
        if not f.exists():
            sys.exit(f"Introuvable : {f}")

    forme = tuple(int(v) for v in args.shape.split(","))
    if len(forme) != 4:
        sys.exit(f"--shape attend 4 entiers NCHW, recu {args.shape!r}")
    _, _, hauteur, largeur = forme

    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    print(f"ONNX  : {args.onnx}")
    print(f"  entree {session.get_inputs()[0].name} {session.get_inputs()[0].shape}"
          f"  ->  sortie {session.get_outputs()[0].name} {session.get_outputs()[0].shape}")
    print(f"ncnn  : {args.ncnn_param}  (calcul {'fp16' if args.fp16 else 'fp32'})")

    resultats = []

    print(f"\n[1] Bruit gaussien ({args.n_alea} tirages) — valide la CONVERSION")
    for k in range(args.n_alea):
        x = np.random.default_rng(k).standard_normal(forme).astype(np.float32)
        r = comparer(f"alea #{k}",
                     sortie_onnx(session, x),
                     sortie_ncnn(str(args.ncnn_param), str(args.ncnn_bin), x,
                                 args.input_blob, args.output_blob, args.fp16),
                     args.seuil)
        if r is None:
            sys.exit("  Formes de sortie incompatibles entre ONNX et ncnn.")
        resultats.append(r)

    if args.crops:
        import cv2
        fichiers = sorted(f for f in args.crops.iterdir()
                          if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if not fichiers:
            print(f"\n[2] Aucune imagette dans {args.crops}")
        else:
            print(f"\n[2] {len(fichiers)} imagettes reelles — valide le COMPORTEMENT")
            for f in fichiers[:20]:
                img = cv2.imread(str(f))
                if img is None:
                    continue
                x = preprocess_rec(img, hauteur, largeur)
                r = comparer(f.name[:22],
                             sortie_onnx(session, x),
                             sortie_ncnn(str(args.ncnn_param), str(args.ncnn_bin), x,
                                         args.input_blob, args.output_blob, args.fp16),
                             args.seuil)
                if r:
                    resultats.append(r)

    pire = max(r["relatif"] for r in resultats)
    seqs = sum(r["seq_ok"] for r in resultats)
    print("\n" + "=" * 74)
    print(f"ecart relatif maximal : {pire:.2e}   (seuil {args.seuil:.0e})")
    print(f"sequences CTC identiques : {seqs}/{len(resultats)}")

    if seqs == len(resultats) and pire < args.seuil:
        print("PORTAGE VALIDE — ncnn reproduit ONNX.")
        return 0
    if seqs == len(resultats):
        print("ACCEPTABLE — les sequences lues sont identiques, mais l'ecart\n"
              "numerique depasse le seuil. Typique de poids stockes en fp16\n"
              "(pnnx convertit en fp16 PAR DEFAUT : relance avec fp16=0 pour\n"
              "isoler l'erreur de conversion de l'erreur de quantification).")
        return 0
    print("PORTAGE INVALIDE — les sequences lues divergent. Ne deploie pas ce\n"
          "modele : il lira faux sans lever d'erreur. Pistes : operateur mal\n"
          "converti, mauvais nom de blob, ou forme d'entree differente de celle\n"
          "passee a pnnx via inputshape=.")
    return 1


if __name__ == "__main__":
    sys.exit(main())