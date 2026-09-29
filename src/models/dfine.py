#!/usr/bin/env python3
"""
Famille « D-FINE » : prétraitement et interprétation des sorties.

D-FINE est un détecteur de type DETR (bâti sur RT-DETR). Il diffère d'un
YOLO sur trois points qui cassent tous le chemin habituel du projet :

1. DEUX ENTRÉES au lieu d'une. Le graphe exporté attend `images`
   (float32 [N,3,640,640]) ET `orig_target_sizes` (int64 [N,2]), cette
   seconde entrée servant au post-traitement interne à remettre les boîtes
   à l'échelle de l'image d'origine.

2. TROIS SORTIES, déjà décodées : `labels` [N,300], `boxes` [N,300,4] et
   `scores` [N,300]. Il n'y a ni tête YOLO brute à décoder, ni NMS à
   appliquer -- DETR produit un ensemble de 300 prédictions appariées, et
   la suppression des doublons est intrinsèque à l'entraînement. Passer ces
   sorties dans `decode_detections()` n'aurait aucun sens.

3. PAS DE LETTERBOX. Le déploiement officiel redimensionne directement en
   640x640 (`T.Resize((640, 640))`), sans conserver le ratio d'aspect et
   sans padding. C'est le modèle lui-même qui rétablit les coordonnées
   d'origine, à partir de `orig_target_sizes`.

"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import cv2
import numpy as np


from src.models.postprocess import Detection

# Noms des tenseurs dans l'export officiel. Ils sont paramétrables partout
# où ils servent : un export maison peut les avoir nommés autrement.
INPUT_IMAGES = "images"
INPUT_SIZES = "orig_target_sizes"
OUTPUT_LABELS = "labels"
OUTPUT_BOXES = "boxes"
OUTPUT_SCORES = "scores"

IMGSZ_DEFAUT = 640


def preprocess_dfine(img_bgr: np.ndarray, imgsz: int = IMGSZ_DEFAUT
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Image BGR -> (images NCHW float32 [0,1], orig_target_sizes int64).

    Redimensionnement DIRECT, sans letterbox : le ratio d'aspect n'est pas
    conservé, conformément au déploiement officiel. C'est `orig_target_sizes`
    qui permet au modèle de rendre des boîtes dans le repère d'origine.
    """
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("Image vide passée à preprocess_dfine")
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError(f"Image BGR 3 canaux attendue, reçu {img_bgr.shape}")

    h, w = img_bgr.shape[:2]

    interp = cv2.INTER_AREA if max(h, w) > imgsz else cv2.INTER_LINEAR
    redim = cv2.resize(img_bgr, (imgsz, imgsz), interpolation=interp)

    rgb = redim[:, :, ::-1].astype(np.float32) / 255.0   # BGR->RGB, [0,1]
    images = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None, ...])

    # Ordre (largeur, hauteur) -- voir l'avertissement en tête de module.
    sizes = np.array([[w, h]], dtype=np.int64)
    return images, sizes


def decode_dfine(labels: np.ndarray, boxes: np.ndarray, scores: np.ndarray,
                 conf_thres: float = 0.35,
                 max_det: int = 100) -> List[Detection]:
    """Sorties D-FINE -> Detection[], sans NMS.

    Les boîtes sont DÉJÀ en xyxy dans le repère de l'image d'origine, et
    déjà dédoublonnées : DETR produit un ensemble apparié, il n'y a rien à
    supprimer. Le seul filtrage utile est celui du score.
    """
    labels = np.asarray(labels).reshape(-1)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)

    if not (len(labels) == len(scores) == len(boxes)):
        raise ValueError(
            f"Sorties incohérentes : labels={len(labels)}, "
            f"scores={len(scores)}, boxes={len(boxes)}. "
            f"Vérifie l'ordre des sorties de l'engine "
            f"({OUTPUT_LABELS}/{OUTPUT_BOXES}/{OUTPUT_SCORES}).")

    garde = scores >= conf_thres
    idx = np.nonzero(garde)[0]
    if idx.size > max_det:
        idx = idx[np.argsort(-scores[idx])][:max_det]

    return [Detection(float(boxes[i, 0]), float(boxes[i, 1]),
                      float(boxes[i, 2]), float(boxes[i, 3]),
                      float(scores[i]), int(labels[i])) for i in idx]


def outputs_in_order(sorties: dict, noms: Sequence[str] = (
        OUTPUT_LABELS, OUTPUT_BOXES, OUTPUT_SCORES)) -> Tuple[np.ndarray, ...]:
    """Récupère les sorties PAR NOM, jamais par position.

    L'ordre des tenseurs de sortie d'un engine TensorRT n'est pas garanti
    d'être celui du graphe ONNX. Prendre `outputs[0]` pour les labels
    marcherait sur une machine et donnerait des classes aberrantes sur une
    autre, sans erreur.
    """
    manquants = [n for n in noms if n not in sorties]
    if manquants:
        raise KeyError(
            f"Sorties absentes de l'engine : {manquants}. "
            f"Présentes : {sorted(sorties)}")
    return tuple(sorties[n] for n in noms)