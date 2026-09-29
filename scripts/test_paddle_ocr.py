#!/usr/bin/env python3
"""
Valide la chaîne PaddleOCR DE BOUT EN BOUT, en ONNX, avant toute conversion.

Ordre des opérations volontaire : on vérifie d'abord que le modèle ONNX,
le prétraitement (src/models/paddle_rec.py) et le jeu de caractères de
configs/config.yaml produisent ENSEMBLE le bon texte. Tant que ce n'est pas
établi, convertir en ncnn puis en INT8 reviendrait à empiler des étapes
au-dessus d'une hypothèse non vérifiée : si la lecture est fausse à la fin,
impossible de savoir si c'est la conversion, la quantification, le charset
ou le prétraitement.

Ce que le script contrôle, dans cet ordre :
  1. la hauteur d'entrée attendue par le modèle (lue, pas supposée)
  2. la cohérence entre le nombre de classes de sortie et le charset
     (PaddleOCR : classes = 1 blanc + len(charset))
  3. le texte décodé sur chaque image

Usage :
    python scripts/test_paddle_ocr.py models/paddleocr/paddleocr_rec.onnx \
        --image data_calibration/ocr/plaque_001.jpg
    python scripts/test_paddle_ocr.py models/paddleocr/paddleocr_rec.onnx \
        --dir data_calibration/ocr --max 10
    # comparer les deux modèles sur les mêmes images :
    python scripts/test_paddle_ocr.py models/paddleocr/paddleocr_rec.onnx \
        models/paddleocr/paddleocr_recv2.onnx --dir data_calibration/ocr --max 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.postprocess_paddle_rec import (ctc_greedy_decode,  # noqa: E402
                                   preprocess_rec)

try:
    import onnxruntime as ort
except ModuleNotFoundError:
    sys.exit("Le paquet `onnxruntime` est requis pour ce test.\n"
             "    pip install onnxruntime\n")


def hauteur_du_modele(session, defaut: int = 48) -> int:
    """Lit la hauteur d'entrée dans le graphe plutôt que de la supposer.

    L'entrée PaddleOCR est [N, 3, H, W] avec N et W dynamiques mais H figée
    à l'export. Se tromper de hauteur ne lève aucune erreur : le modèle
    redimensionnerait implicitement et lirait n'importe quoi.
    """
    shape = session.get_inputs()[0].shape
    if len(shape) == 4 and isinstance(shape[2], int) and shape[2] > 0:
        return int(shape[2])
    print(f"  [!] hauteur non figée dans le modèle ({shape}), on prend {defaut}")
    return defaut


def charger_charset(config_path: Path) -> list:
    cfg = yaml.safe_load(config_path.read_text())
    charset = cfg["models"]["ocr"].get("charset")
    if not charset:
        raise SystemExit(f"Pas de `models.ocr.charset` dans {config_path}")
    return [str(c) for c in charset]


def lire(session, img, charset, hauteur, largeur_max):
    tenseur = preprocess_rec(img, hauteur=hauteur, largeur_max=largeur_max)
    nom_entree = session.get_inputs()[0].name
    sortie = session.run(None, {nom_entree: tenseur})[0]
    texte, conf, jetons = ctc_greedy_decode(sortie, charset)
    return texte, conf, jetons, sortie.shape


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("modeles", nargs="+", type=Path,
                    help="un ou plusieurs .onnx de reconnaissance à comparer")
    ap.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    ap.add_argument("--image", type=Path, default=None)
    ap.add_argument("--dir", type=Path, default=None,
                    help="dossier de vignettes de plaques")
    ap.add_argument("--max", type=int, default=10)
    ap.add_argument("--largeur-max", type=int, default=320)
    args = ap.parse_args()

    if not args.image and not args.dir:
        return ap.error("donne --image ou --dir")

    charset = charger_charset(args.config)
    print(f"charset ({len(charset)} caractères) : {charset}")
    print(f"-> classes attendues en sortie : {len(charset) + 1} "
          f"(= {len(charset)} caractères + 1 blanc CTC)")

    if args.image:
        images = [args.image]
    else:
        images = sorted(f for f in args.dir.iterdir()
                        if f.suffix.lower() in (".jpg", ".jpeg", ".png"))[:args.max]
    if not images:
        return print("Aucune image trouvée.") or 1

    for modele in args.modeles:
        if not modele.exists():
            print(f"\n[!] {modele} introuvable, ignoré.")
            continue

        print(f"\n=== {modele} ===")
        session = ort.InferenceSession(str(modele),
                                       providers=["CPUExecutionProvider"])
        hauteur = hauteur_du_modele(session)
        sortie_shape = session.get_outputs()[0].shape
        n_classes = sortie_shape[-1] if isinstance(sortie_shape[-1], int) else None
        print(f"  entrée  : {session.get_inputs()[0].name} "
              f"{session.get_inputs()[0].shape}  -> hauteur retenue : {hauteur}")
        print(f"  sortie  : {session.get_outputs()[0].name} {sortie_shape}")

        if n_classes is not None:
            attendu = len(charset) + 1
            if n_classes == attendu:
                print(f"  charset : OK ({n_classes} classes = {len(charset)} + blanc)")
            else:
                print(f"  [!] INCOHÉRENCE : le modèle sort {n_classes} classes, "
                      f"le charset en implique {attendu}. Le texte décodé sera "
                      f"faux ou décalé -- vérifie le dictionnaire fourni avec "
                      f"ce modèle.")

        print(f"\n  {'image':<28} {'texte lu':<16} {'conf':>6}   jetons")
        print("  " + "-" * 70)
        for f in images:
            img = cv2.imread(str(f))
            if img is None:
                print(f"  {f.name:<28} [illisible]")
                continue
            texte, conf, jetons, shape = lire(session, img, charset,
                                              hauteur, args.largeur_max)
            print(f"  {f.name:<28} {texte or '(vide)':<16} {conf:>6.3f}   "
                  f"{' · '.join(jetons)}")

        print(f"\n  (forme de sortie sur la dernière image : {shape} "
              f"-> {shape[1] if len(shape) > 2 else '?'} pas de temps)")

    print("\nSi le texte lu correspond aux plaques, la chaîne "
          "prétraitement + charset + décodage est validée : on peut passer "
          "à la conversion ncnn.")
    return 0


if __name__ == "__main__":
    sys.exit(main())