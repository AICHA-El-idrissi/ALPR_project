#!/usr/bin/env python3

from __future__ import annotations

from typing import List, Sequence, Tuple

import cv2
import numpy as np

HAUTEUR_DEFAUT = 48
LARGEUR_MAX_DEFAUT = 320

# (x/255 - 0.5) / 0.5  ->  exprimé comme ncnn2table les attend.
NCNN_MEAN = [127.5, 127.5, 127.5]
NCNN_NORM = [1 / 127.5, 1 / 127.5, 1 / 127.5]


def preprocess_rec(img: np.ndarray, hauteur: int = HAUTEUR_DEFAUT,
                   largeur_max: int = LARGEUR_MAX_DEFAUT) -> np.ndarray:
    """Vignette de plaque BGR -> tenseur NCHW float32 normalisé [-1, 1].

    Hauteur fixe, ratio d'aspect CONSERVÉ, complétion à droite par du noir.
    C'est le prétraitement de PaddleOCR, et il doit être reproduit à
    l'identique par la calibration INT8 (voir NCNN_MEAN / NCNN_NORM).
    """
    if img is None or img.size == 0:
        raise ValueError("Image vide passée à preprocess_rec")
    if hauteur <= 0 or largeur_max <= 0:
        raise ValueError(f"Dimensions invalides : {hauteur}x{largeur_max}")

    h, w = img.shape[:2]
    ratio = w / float(h)
    largeur = int(np.ceil(hauteur * ratio))
    largeur = max(1, min(largeur, largeur_max))

    # INTER_AREA en réduction, INTER_LINEAR en agrandissement -- même règle
    # que postprocess.letterbox, et pour la même raison :
    interp = cv2.INTER_AREA if largeur < w else cv2.INTER_LINEAR
    redim = cv2.resize(img, (largeur, hauteur), interpolation=interp)

    # BGR -> RGB, CHW, [-1, 1]
    bgr = (redim.astype(np.float32) / 255.0 - 0.5) / 0.5
    chw = bgr.transpose(2, 0, 1)

    # Complétion à DROITE uniquement (jamais centrée) : le décodage CTC lit
    # la séquence de gauche à droite, un padding à gauche décalerait tout.
    tenseur = np.zeros((1, 3, hauteur, largeur_max), dtype=np.float32)
    tenseur[0, :, :, :largeur] = chw
    return tenseur


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)      # stabilité numérique
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _est_deja_normalise(probs: np.ndarray) -> bool:
    """Un modèle PaddleOCR exporté en inférence finit généralement par un
    softmax, mais pas toujours. Plutôt que de le supposer, on regarde si les
    lignes somment à 1 -- appliquer un softmax sur des probabilités déjà
    normalisées écraserait les écarts et ferait chuter la confiance."""
    if probs.min() < 0.0 or probs.max() > 1.0:
        return False
    return bool(np.allclose(probs.sum(axis=-1), 1.0, atol=1e-3))


def ctc_greedy_decode(sortie: np.ndarray, charset: Sequence[str],
                      blank: int = 0) -> Tuple[str, float, List[str]]:
    """Sortie brute du réseau -> (texte, confiance, jetons).

    Décodage glouton : on prend l'indice le plus probable à chaque pas de
    temps, on fusionne les répétitions consécutives, puis on retire le blanc.
    Cet ordre compte : fusionner APRÈS avoir retiré le blanc collerait deux
    caractères identiques séparés par un blanc -- c'est précisément ce que le
    blanc CTC sert à distinguer (« 11 » ne doit pas devenir « 1 »).

    Retourne aussi la liste des jetons : le vote inter-lectures du pipeline
    compare des jetons, pas des caractères, car un jeton du jeu marocain peut
    s'écrire sur plusieurs lettres latines (« waw », « gr »).
    """
    probs = np.asarray(sortie, dtype=np.float32)
    probs = np.squeeze(probs)
    if probs.ndim != 2:
        raise ValueError(
            f"Sortie CTC attendue en (T, C) ou (1, T, C), reçu {sortie.shape}")

    if not _est_deja_normalise(probs):
        probs = _softmax(probs, axis=-1)

    indices = probs.argmax(axis=-1)
    scores = probs.max(axis=-1)

    jetons: List[str] = []
    confiances: List[float] = []
    precedent = -1
    for t, idx in enumerate(indices):
        idx = int(idx)
        if idx == precedent:          # répétition : même caractère maintenu
            precedent = idx
            continue
        precedent = idx
        if idx == blank:
            continue
        # Indice 0 = blanc, donc le caractère i vient de charset[i - 1].
        pos = idx - 1 if blank == 0 else idx
        if 0 <= pos < len(charset):
            jetons.append(str(charset[pos]))
            confiances.append(float(scores[t]))
        # Un indice hors charset signale un décalage entre le modèle et le
        # dictionnaire : on l'ignore plutôt que de produire un caractère faux.

    texte = "".join(jetons)
    conf = float(np.mean(confiances)) if confiances else 0.0
    return texte, conf, jetons


def load_charset(source, n_classes: int | None = None) -> List[str]:
    """Charge le jeu de caractères depuis une liste ou un fichier texte.

    PaddleOCR distribue un dictionnaire à raison d'un caractère par ligne,
    SANS le blanc (qui est ajouté à l'indice 0 au décodage). On respecte
    cette convention : `charset[0]` est donc le caractère d'indice 1.

    UNE LIGNE VIDE EST UNE CLASSE. 
    Le dictionnaire commence par une ligne vide : elle occupe l'indice 1 du modèle. La filtrer décale
    TOUS les caractères suivants d'un cran, ce qui produit un texte
    parfaitement plausible et entièrement faux -- la panne la plus difficile
    à voir de toute la chaîne. Seule la dernière ligne, artefact du « \n »
    final, n'est pas une classe.

    `n_classes` (la dernière dimension de la sortie du modèle) permet de
    vérifier la concordance au chargement plutôt que de la découvrir sur des
    lectures fausses.
    """
    if isinstance(source, (list, tuple)):
        charset = [str(c) for c in source]
    else:
        from pathlib import Path
        chemin = Path(source)
        if not chemin.exists():
            raise FileNotFoundError(
                f"Dictionnaire de caractères introuvable : {chemin}\n"
                f"C'est le fichier fourni avec le modèle PaddleOCR "
                f"(un caractère par ligne, sans le blanc CTC).")
        # split("\n") plutôt que splitlines() : on maîtrise ainsi le retrait
        # de la seule dernière ligne, au lieu de filtrer toutes les vides.
        lignes = chemin.read_text(encoding="utf-8").split("\n")
        if lignes and lignes[-1] == "":
            lignes = lignes[:-1]
        charset = [ligne.rstrip("\r") for ligne in lignes]

    if n_classes is not None and len(charset) + 1 != n_classes:
        raise ValueError(
            f"Le dictionnaire compte {len(charset)} caractères, soit "
            f"{len(charset) + 1} classes avec le blanc CTC, mais le modèle en "
            f"sort {n_classes}.\n"
            f"Écart de {n_classes - len(charset) - 1}. Causes habituelles : une "
            f"ligne vide filtrée à tort, ou `use_space_char` actif à "
            f"l'entraînement (qui ajoute un espace en DERNIÈRE position).")
    return charset