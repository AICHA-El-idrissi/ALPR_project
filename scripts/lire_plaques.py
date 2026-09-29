#!/usr/bin/env python3
"""Lit de vraies plaques avec le modele et AFFICHE LE TEXTE decode.

Pourquoi ce script existe, alors que verif_ncnn.py dit deja « PORTAGE VALIDE » :

    verif_ncnn.py compare ncnn a ONNX EN LEUR DONNANT LA MEME FORME D'ENTREE.
    Si cette forme est la mauvaise, les deux moteurs se trompent exactement de
    la meme maniere, l'ecart reste nul, et le script annonce « VALIDE » sur un
    modele qui lit faux. Il valide la TRADUCTION, jamais la LECTURE.

Seul le texte decode tranche. Ce script le produit, avec le charset du projet,
pour qu'on puisse le comparer a ce qu'on lit sur l'image.

    python3 scripts/lire_plaques.py \\
        --modele models/exported/ncnn/paddle_rec_v2/model.ncnn.param \\
        --crops  data/crops/ \\
        --config configs/config.yaml

    # comparer deux largeurs de canvas pour choisir la bonne
    python3 scripts/lire_plaques.py --modele ... --crops ... --largeurs 192,256,320
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import cv2
except ModuleNotFoundError:
    sys.exit("Le paquet `opencv-python` est requis.\n\n    pip install opencv-python\n")

CHARSET_DEFAUT = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                  "a", "b", "d", "gr", "h", "t", "w", "waw", "y"]


def preprocess_rec(img: np.ndarray, hauteur: int, largeur_max: int):
    """Hauteur FIXE, ratio conserve, completion a DROITE, plage [-1,1].

    Rend aussi le nombre de colonnes reellement occupees par l'image, pour
    pouvoir mesurer la part de remplissage.
    """
    h, w = img.shape[:2]
    largeur = max(1, min(int(np.ceil(hauteur * (w / float(h)))), largeur_max))
    interp = cv2.INTER_AREA if largeur < w else cv2.INTER_LINEAR
    redim = cv2.resize(img, (largeur, hauteur), interpolation=interp)
    # PaddleOCR ne convertit PAS en RGB : resize_norm_img travaille sur
    # l'image telle que cv2 la charge, en BGR. Mesure a l'appui : 0,955 de
    # confiance en BGR contre 0,946 en RGB sur les memes imagettes.
    bgr = (redim.astype(np.float32) / 255.0 - 0.5) / 0.5
    out = np.zeros((1, 3, hauteur, largeur_max), np.float32)
    out[0, :, :, :largeur] = bgr.transpose(2, 0, 1)
    return out, largeur


def _softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def ctc_greedy(sortie, charset, blank: int = 0):
    """Fusion des repetitions AVANT retrait du blanc : c'est le blanc qui
    separe deux caracteres identiques (« 11 » ne doit pas devenir « 1 »)."""
    probs = np.squeeze(np.asarray(sortie, np.float32))
    if probs.ndim != 2:
        raise ValueError(f"Sortie CTC attendue en (T,C), recu {probs.shape}")
    deja = (probs.min() >= 0 and probs.max() <= 1
            and np.allclose(probs.sum(axis=-1), 1.0, atol=1e-3))
    if not deja:
        probs = _softmax(probs)
    indices, scores = probs.argmax(axis=-1), probs.max(axis=-1)
    jetons, confs, prec = [], [], -1
    for t, idx in enumerate(indices):
        idx = int(idx)
        if idx == prec:
            continue
        prec = idx
        if idx == blank:
            continue
        pos = idx - 1
        if 0 <= pos < len(charset):
            jetons.append(str(charset[pos]))
            confs.append(float(scores[t]))
    return "".join(jetons), (float(np.mean(confs)) if confs else 0.0), len(indices)


# ---------------------------------------------------------------------------
class MoteurOnnx:
    def __init__(self, chemin: Path):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(str(chemin), providers=["CPUExecutionProvider"])
        self.nom = self.sess.get_inputs()[0].name
        self.kind = "onnx"

    def run(self, x):
        return np.squeeze(self.sess.run(None, {self.nom: x})[0])


class MoteurNcnn:
    """Attention aux deux pieges de l'API python de ncnn, tous deux
    silencieux : ncnn.Mat ENVELOPPE le tampon numpy sans le posseder, et
    np.array(mat) rend une VUE. D'ou la reference vivante et la copie."""

    def __init__(self, param: Path, bin_: Path, entree="in0", sortie="out0"):
        import ncnn
        self._ncnn = ncnn
        self.net = ncnn.Net()
        self.net.opt.use_fp16_packed = False
        self.net.opt.use_fp16_storage = False
        self.net.opt.use_fp16_arithmetic = False
        if self.net.load_param(str(param)) != 0:
            sys.exit(f"ncnn n'a pas pu lire {param}")
        if self.net.load_model(str(bin_)) != 0:
            sys.exit(f"ncnn n'a pas pu lire {bin_}")
        self.entree, self.sortie, self.kind = entree, sortie, "ncnn"

    def run(self, x):
        tampon = np.ascontiguousarray(x[0], dtype=np.float32)
        ex = self.net.create_extractor()
        ex.input(self.entree, self._ncnn.Mat(tampon))
        code, mat = ex.extract(self.sortie)
        if code != 0:
            sys.exit(f"ncnn extract({self.sortie!r}) a echoue (code {code}).")
        resultat = np.array(mat).copy()
        del ex, tampon
        return np.squeeze(resultat)


def lire_dictionnaire(chemin: Path):
    """Un caractere par ligne, ordre = indices du modele. La derniere ligne
    vide, due au \n final, n'est pas une classe."""
    lignes = chemin.read_text(encoding="utf-8").split("\n")
    if lignes and lignes[-1] == "":
        lignes = lignes[:-1]
    return lignes


def charger_charset(config: Path | None, dico: Path | None, section: str):
    """Ordre de priorite : --dict explicite, puis charset_file de la config,
    puis charset inline, puis le defaut code en dur.

    La section compte : `models.ocr` decrit le YOLO-OCR (19 classes de
    detection, sans blanc) et `models.ocr_paddle` decrit PaddleOCR (19
    caracteres + 1 blanc CTC). Les deux listes contiennent presque les memes
    symboles mais PAS dans le meme ordre : les confondre traduit mal toutes
    les lettres tout en laissant les chiffres justes, ce qui rend l'erreur
    difficile a voir.
    """
    if dico is not None:
        if not dico.exists():
            sys.exit(f"Dictionnaire introuvable : {dico}")
        return lire_dictionnaire(dico), f"--dict {dico}"

    if config is None or not config.exists():
        return list(CHARSET_DEFAUT), "defaut interne"

    import yaml
    cfg = yaml.safe_load(config.read_text()) or {}
    bloc = cfg.get("models", {}).get(section)
    if bloc is None:
        sys.exit(f"Section models.{section} absente de {config}.\n"
                 f"Sections presentes : "
                 f"{sorted(cfg.get('models', {}))}")

    fichier = bloc.get("charset_file")
    if fichier:
        chemin = Path(fichier)
        if not chemin.exists():
            sys.exit(f"charset_file introuvable : {chemin}  "
                     f"(declare dans models.{section})")
        return lire_dictionnaire(chemin), f"{config}::{section}.charset_file"

    cs = bloc.get("charset")
    if cs:
        return [str(c) for c in cs], f"{config}::{section}.charset"
    return list(CHARSET_DEFAUT), "defaut interne"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modele", type=Path, required=True,
                    help=".onnx, ou le .param ncnn (le .bin est deduit)")
    ap.add_argument("--ncnn-bin", type=Path, default=None)
    ap.add_argument("--crops", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--section", default="ocr_paddle",
                    help="section de models: a lire dans la config "
                         "(ocr_paddle pour PaddleOCR, ocr pour le YOLO-OCR)")
    ap.add_argument("--dict", type=Path, default=None,
                    help="dictionnaire PaddleOCR, prioritaire sur la config")
    ap.add_argument("--hauteur", type=int, default=48)
    ap.add_argument("--largeurs", default="320",
                    help="une ou plusieurs largeurs de canvas a comparer, "
                         "ex. 192,256,320")
    ap.add_argument("--input-blob", default="in0")
    ap.add_argument("--output-blob", default="out0")
    ap.add_argument("--max", type=int, default=30)
    args = ap.parse_args()

    if not args.modele.exists():
        sys.exit(f"Introuvable : {args.modele}")
    charset, origine = charger_charset(args.config, args.dict, args.section)
    print(f"charset : {len(charset)} caracteres -> {len(charset)+1} classes "
          f"attendues (dont le blanc CTC)")
    print(f"  source : {origine}")
    # repr() sur les entrees vides ou blanches : une ligne invisible du
    # dictionnaire occupe un indice et decale tout ce qui suit.
    print(f"  {[c if c.strip() else repr(c) for c in charset]}")

    fichiers = sorted(f for f in args.crops.iterdir()
                      if f.suffix.lower() in (".jpg", ".jpeg", ".png"))[:args.max]
    if not fichiers:
        sys.exit(f"Aucune imagette dans {args.crops}")

    largeurs = [int(v) for v in args.largeurs.split(",")]

    for largeur_max in largeurs:
        # Un modele ncnn est fige a UNE forme : changer de largeur exige un
        # reexport. En ONNX a axes libres, on peut comparer sans reexporter.
        if args.modele.suffix == ".onnx":
            moteur = MoteurOnnx(args.modele)
        else:
            bin_ = args.ncnn_bin or args.modele.with_suffix(".bin")
            if not bin_.exists():
                sys.exit(f"Fichier .bin introuvable : {bin_}  (--ncnn-bin)")
            moteur = MoteurNcnn(args.modele, bin_, args.input_blob, args.output_blob)

        print(f"\n{'='*74}")
        print(f"canvas {args.hauteur}x{largeur_max}  ({moteur.kind})")
        print(f"{'imagette':<28} {'taille':>10} {'utile':>7}  {'T':>3}  "
              f"{'conf':>5}  lecture")
        print("-" * 74)

        for f in fichiers:
            img = cv2.imread(str(f))
            if img is None:
                print(f"  {f.name[:26]:<26} [illisible]")
                continue
            h, w = img.shape[:2]
            x, utiles = preprocess_rec(img, args.hauteur, largeur_max)
            try:
                brut = moteur.run(x)
            except Exception as exc:
                print(f"  {f.name[:26]:<26} ECHEC : {exc}")
                continue
            texte, conf, T = ctc_greedy(brut, charset)
            part = 100.0 * utiles / largeur_max
            print(f"  {f.name[:26]:<26} {w:>4}x{h:<3}  {part:5.1f}%  {T:>3}  "
                  f"{conf:5.3f}  {texte or '(vide)'}")

    print(f"\n{'='*74}")
    print("Compare ces lectures a ce que TU lis sur les imagettes.")
    print("C'est le seul controle qui detecte une forme d'entree mal choisie :")
    print("verif_ncnn.py donne la meme forme aux deux moteurs, donc il ne peut")
    print("pas voir qu'elle est fausse.")
    return 0


if __name__ == "__main__":
    sys.exit(main())