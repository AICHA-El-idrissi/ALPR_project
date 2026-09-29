#!/usr/bin/env python3
"""Cherche le bon pretraitement PaddleOCR en les essayant tous.

Quand un modele CTC rend des lectures incoherentes ET une confiance basse
(0,1 a 0,3 la ou un modele qui reconnait donne 0,8 a 0,99), le probleme est
presque toujours en amont du modele : l'entree ne ressemble pas a ce qu'il a
vu a l'entrainement. Plutot que de deviner laquelle des hypotheses est la
bonne, ce script les essaie toutes et les classe par confiance.

La confiance est ici un bon juge parce qu'on n'a pas la verite terrain : un
modele nourri correctement est SUR de lui, un modele nourri n'importe comment
repartit ses probabilites et plafonne bas.

    python3 scripts/diagnostic_ocr.py \\
        --onnx  models/paddleocr/paddleocr_recv2.onnx \\
        --crops data/crops/ \\
        --dict  models/paddleocr/dict.txt

Utiliser l'ONNX NON FIGE (axes libres) : le script fait varier la largeur, ce
qu'un modele ncnn, fige a une seule forme, ne permet pas.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

try:
    import cv2
except ModuleNotFoundError:
    sys.exit("Le paquet `opencv-python` est requis.\n\n    pip install opencv-python\n")
try:
    import onnxruntime as ort
except ModuleNotFoundError:
    sys.exit("Le paquet `onnxruntime` est requis.\n\n    pip install onnxruntime\n")


def preparer(img, hauteur, largeur_max, canaux, normalisation, remplissage, mode):
    """Construit l'entree NCHW selon une combinaison d'hypotheses.

    mode          : "ratio"     -> hauteur fixe, ratio conserve, completion a
                                   droite (schema officiel de PaddleOCR)
                    "etirement" -> redimensionnement DIRECT vers hauteur x
                                   largeur_max, aucun remplissage. Le rapport
                                   d'aspect est ecrase, mais tout le canvas
                                   porte de l'image.
    canaux        : "bgr" (comme cv2 le charge, ce que fait PaddleOCR) ou "rgb"
    normalisation : "moins1_1" -> (x/255 - 0.5) / 0.5   (PaddleOCR)
                    "zero_un"  -> x/255
    remplissage   : "zero"  -> zeros dans l'espace normalise (PaddleOCR)
                    "bord"  -> repetition de la derniere colonne de l'image
                    (sans effet en mode "etirement")
    """
    h, w = img.shape[:2]
    if mode == "etirement":
        largeur = largeur_max
    else:
        largeur = max(1, min(int(np.ceil(hauteur * (w / float(h)))), largeur_max))
    interp = cv2.INTER_AREA if largeur < w else cv2.INTER_LINEAR
    redim = cv2.resize(img, (largeur, hauteur), interpolation=interp)

    if canaux == "rgb":
        redim = redim[:, :, ::-1]
    x = redim.astype(np.float32) / 255.0
    if normalisation == "moins1_1":
        x = (x - 0.5) / 0.5

    out = np.zeros((1, 3, hauteur, largeur_max), np.float32)
    out[0, :, :, :largeur] = x.transpose(2, 0, 1)
    if mode == "ratio" and remplissage == "bord" and largeur < largeur_max:
        out[0, :, :, largeur:] = out[0, :, :, largeur - 1:largeur]
    return out, largeur


def _softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def ctc_greedy(sortie, charset, blank=0):
    probs = np.squeeze(np.asarray(sortie, np.float32))
    if probs.ndim != 2:
        raise ValueError(f"Sortie CTC attendue en (T,C), recu {probs.shape}")
    deja = (probs.min() >= 0 and probs.max() <= 1
            and np.allclose(probs.sum(axis=-1), 1.0, atol=1e-3))
    if not deja:
        probs = _softmax(probs)
    idx, sc = probs.argmax(axis=-1), probs.max(axis=-1)
    jetons, confs, prec = [], [], -1
    for t, i in enumerate(idx):
        i = int(i)
        if i == prec:
            continue
        prec = i
        if i == blank:
            continue
        p = i - 1
        if 0 <= p < len(charset):
            jetons.append(str(charset[p]))
            confs.append(float(sc[t]))
    # part_blanc : sur une entree correcte, le CTC passe l'essentiel de son
    # temps sur le blanc. Une part tres basse trahit un modele qui « bavarde ».
    part_blanc = float((idx == blank).mean())
    return "".join(jetons), (float(np.mean(confs)) if confs else 0.0), part_blanc


def charger_dict(chemin: Path | None):
    if chemin is None or not chemin.exists():
        return None
    lignes = chemin.read_text(encoding="utf-8").split("\n")
    if lignes and lignes[-1] == "":
        lignes = lignes[:-1]          # derniere ligne vide due au \n final
    return lignes


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", type=Path, required=True,
                    help="ONNX a axes LIBRES (non fige)")
    ap.add_argument("--crops", type=Path, required=True)
    ap.add_argument("--dict", type=Path, default=None,
                    help="character_dict_path de l'entrainement PaddleOCR")
    ap.add_argument("--hauteurs", default="48")
    ap.add_argument("--largeurs", default="160,192,256,320")
    ap.add_argument("--modes", default="ratio,etirement",
                    help="ratio (PaddleOCR officiel) et/ou etirement")
    ap.add_argument("--max", type=int, default=8,
                    help="imagettes utilisees par combinaison")
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()

    if not args.onnx.exists():
        sys.exit(f"Introuvable : {args.onnx}")

    sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    nom_entree = sess.get_inputs()[0].name
    forme_entree = sess.get_inputs()[0].shape
    forme_sortie = sess.get_outputs()[0].shape
    print(f"ONNX   : {args.onnx.name}")
    print(f"  entree {nom_entree} {forme_entree}  ->  sortie {forme_sortie}")
    n_classes = forme_sortie[-1] if isinstance(forme_sortie[-1], int) else None
    if n_classes:
        print(f"  {n_classes} classes -> le charset doit compter {n_classes - 1} "
              f"caracteres (le blanc CTC occupe l'indice 0)")

    if any(isinstance(d, int) and d > 0 for d in (forme_entree[0], forme_entree[3])):
        print("\n  [!] Cet ONNX semble FIGE : le balayage des largeurs sera sans effet.\n"
              "      Utilise l'ONNX d'origine, avant fix_onnx_shapes.py.")

    dico = charger_dict(args.dict)
    if dico:
        print(f"  dict : {len(dico)} lignes -> {len(dico) + 1} classes sans espace, "
              f"{len(dico) + 2} avec espace")
        if n_classes and len(dico) + 1 == n_classes:
            charset = dico
            print("  -> concordance SANS espace")
        elif n_classes and len(dico) + 2 == n_classes:
            charset = dico + [" "]
            print("  -> concordance AVEC espace final (use_space_char actif)")
        else:
            charset = dico
            print(f"  [!] ni {len(dico)+1} ni {len(dico)+2} ne vaut {n_classes} : "
                  f"ce dictionnaire ne correspond pas a ce modele.")
    else:
        charset = [str(i) for i in range((n_classes or 21) - 1)]
        print("  [!] pas de --dict : les jetons affiches sont des INDICES, pas des "
              "caracteres.\n      La confiance reste exploitable, la lecture non.")

    fichiers = sorted(f for f in args.crops.iterdir()
                      if f.suffix.lower() in (".jpg", ".jpeg", ".png"))[:args.max]
    if not fichiers:
        sys.exit(f"Aucune imagette dans {args.crops}")
    images = [cv2.imread(str(f)) for f in fichiers]
    images = [(f, im) for f, im in zip(fichiers, images) if im is not None]
    print(f"  {len(images)} imagettes de test\n")

    hauteurs = [int(v) for v in args.hauteurs.split(",")]
    largeurs = [int(v) for v in args.largeurs.split(",")]
    combinaisons = []
    for h, lmax, mode, canaux, norm in itertools.product(
            hauteurs, largeurs, tuple(args.modes.split(",")),
            ("bgr", "rgb"), ("moins1_1", "zero_un")):
        # En mode "etirement" il n'y a aucun remplissage : faire varier cet
        # axe produirait des doublons qui polluent le classement.
        for rempl in (("zero", "bord") if mode == "ratio" else ("zero",)):
            combinaisons.append((h, lmax, mode, canaux, norm, rempl))

    print(f"{len(combinaisons)} combinaisons x {len(images)} imagettes\n")
    resultats = []
    for (h, lmax, mode, canaux, norm, rempl) in combinaisons:
        confs, blancs, textes, parts = [], [], [], []
        for _, img in images:
            x, utiles = preparer(img, h, lmax, canaux, norm, rempl, mode)
            parts.append(utiles / lmax)
            try:
                brut = np.squeeze(sess.run(None, {nom_entree: x})[0])
                texte, conf, pb = ctc_greedy(brut, charset)
            except Exception:
                confs = []
                break
            confs.append(conf)
            blancs.append(pb)
            textes.append(texte)
        if not confs:
            continue
        resultats.append({
            "cle": (h, lmax, mode, canaux, norm, rempl),
            "conf": float(np.mean(confs)),
            "blanc": float(np.mean(blancs)),
            "utile": float(np.mean(parts)),
            "exemple": textes[0] if textes else "",
        })

    if not resultats:
        sys.exit("Aucune combinaison n'a pu s'executer.")
    resultats.sort(key=lambda r: -r["conf"])

    print(f"{'h':>3} {'larg':>5} {'mode':>10} {'canaux':>6} {'normalis.':>10} "
          f"{'rempl':>5} {'utile':>6}  {'conf':>5} {'blanc':>6}  lecture")
    print("-" * 100)
    for r in resultats[:args.top]:
        h, lmax, mode, canaux, norm, rempl = r["cle"]
        print(f"{h:>3} {lmax:>5} {mode:>10} {canaux:>6} {norm:>10} "
              f"{rempl:>5} {r['utile']:>5.0%}  {r['conf']:5.3f} {r['blanc']:5.1%}  "
              f"{r['exemple'][:28]}")

    # Comparaison directe des deux schemas de redimensionnement : c'est la
    # question posee, elle merite sa propre ligne.
    for mode in ("ratio", "etirement"):
        sous = [r for r in resultats if r["cle"][2] == mode]
        if sous:
            m = max(sous, key=lambda r: r["conf"])
            print(f"\n  meilleur en mode {mode:<10} : conf={m['conf']:.3f}  "
                  f"blanc={m['blanc']:.1%}  ({m['cle'][1]} px de large)")

    meilleur, pire = resultats[0], resultats[-1]
    print("\n" + "=" * 92)
    print(f"meilleure confiance : {meilleur['conf']:.3f}   "
          f"pire : {pire['conf']:.3f}")
    if meilleur["conf"] < 0.5:
        print(
            "AUCUNE combinaison ne depasse 0,5. Le pretraitement n'est donc PAS\n"
            "le seul probleme. A verifier, dans cet ordre :\n"
            "  1. les imagettes de data/crops/ sont-elles bien des plaques cadrees ?\n"
            "     Ouvre-les une par une : un cadrage trop large ou decale suffit.\n"
            "  2. ce .onnx correspond-il au dict fourni, et a un entrainement mene\n"
            "     a terme ? Un modele interrompu tot rend exactement ce genre de\n"
            "     bouillie repetitive.\n"
            "  3. la hauteur : essaie --hauteurs 32,48 si le graphe l'autorise.")
    else:
        h, lmax, mode, canaux, norm, rempl = meilleur["cle"]
        print(f"Reprends cette combinaison dans preprocess_rec :\n"
              f"  hauteur={h}, largeur={lmax}, mode={mode}, canaux={canaux},\n"
              f"  normalisation={norm}, remplissage={rempl}\n"
              f"puis refais l'export ncnn a cette largeur et relance "
              f"lire_plaques.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())