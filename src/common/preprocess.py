#!/usr/bin/env python3
"""Prétraitement par famille de modèles, commun aux TROIS backends.

Ce module ne dépend d'aucun moteur d'inférence. C'est délibéré : ncnn,
TensorRT et ONNX Runtime doivent nourrir un modèle EXACTEMENT de la même
façon, sinon deux backends portant le même modèle lisent différemment et
rien ne le signale.

Le même prétraitement sert aussi à la calibration INT8. Une table construite
sur des images préparées autrement que ne le fera l'inférence produit un
modèle qui lit faux sans lever d'erreur.

QUATRE familles, quatre conventions incompatibles :

    famille      redimensionnement          canaux  plage      remplissage
    -----------  -------------------------  ------  ---------  ------------------
    yolo         letterbox (ratio + gris)   RGB     [0, 1]     gris 114, CENTRÉ
    yolo_chars   ÉTIREMENT carré            RGB     [0, 1]     aucun
    dfine        resize DIRECT              RGB     [0, 1]     aucun
    paddle_rec   hauteur fixe, ratio        BGR     [-1, 1]    gris 127,5, À DROITE

Les différences ne sont pas cosmétiques, et elles ne se déduisent pas les
unes des autres. Deux mesures du projet le montrent, et elles vont en sens
contraire.

Sur l'OCR par détection de caractères, une vignette de plaque de 135x28
letterboxée vers 608x608 n'occupe que 20,7 % du canvas : 79,3 % de gris que
le modèle n'a jamais vu à l'entraînement. Score maximal mesuré 0,011-0,015
en letterbox contre 0,27-0,29 en étirement. D'où la famille `yolo_chars`,
distincte de `yolo`.

Sur PaddleOCR, l'ordre s'inverse : 0,955 de confiance à ratio conservé avec
remplissage contre 0,871 en étirement, mesuré sur 11 imagettes réelles.
Chaque famille attend ce qu'elle a vu à l'entraînement, et rien d'autre.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

from src.models.postprocess import Detection , letterbox
from src.models import dfine as dfine_family
from src.models.paddle_rec import preprocess_rec

FAMILIES = ("yolo", "yolo_chars", "dfine", "paddle_rec")

# Familles qui rendent des boîtes, par opposition à celles qui rendent du
# texte. `yolo_chars` rend des boîtes de CARACTÈRES : c'est un détecteur au
# sens du moteur, mais un étage de reconnaissance au sens du pipeline.
DETECTION_FAMILIES = ("yolo", "yolo_chars", "dfine")
RECOGNITION_FAMILIES = ("paddle_rec",)
# Les deux familles utilisables comme étage OCR du pipeline.
OCR_FAMILIES = ("yolo_chars", "paddle_rec")

REC_HEIGHT_DEFAUT = 48
REC_WIDTH_DEFAUT = 320


def verify_model(family: str) -> None:
    if family not in FAMILIES:
        raise ValueError(f"Famille inconnue : {family!r}. Attendu : {FAMILIES}")


def expected_input_shapes(family: str, imgsz: int,
                          rec_height: int = REC_HEIGHT_DEFAUT,
                          rec_width: int = REC_WIDTH_DEFAUT
                          ) -> Dict[str, Tuple[int, ...]]:
    """Shapes attendues par famille, pour résoudre un graphe dynamique."""
    verify_model(family)
    if family in ("yolo", "yolo_chars"):
        return {"images": (1, 3, imgsz, imgsz)}
    if family == "dfine":
        return {dfine_family.INPUT_IMAGES: (1, 3, imgsz, imgsz),
                dfine_family.INPUT_SIZES: (1, 2)}
    return {"x": (1, 3, rec_height, rec_width)}


# ---------------------------------------------------------------------------
# Redimensionnements
# ---------------------------------------------------------------------------
def stretch_to_square(img: np.ndarray, imgsz: int) -> np.ndarray:
    """Étire vers un canvas carré, SANS remplissage.

    Pour l'OCR par détection de caractères. Le rapport d'aspect est écrasé,
    mais c'est justement ce que le modèle a vu à l'entraînement : ses
    imagettes d'apprentissage étaient étirées de la même façon. Préserver le
    ratio ici remplirait le canvas de gris et ferait chuter les scores d'un
    facteur vingt.
    """
    h, w = img.shape[:2]
    interp = cv2.INTER_AREA if max(h, w) > imgsz else cv2.INTER_LINEAR
    return cv2.resize(img, (imgsz, imgsz), interpolation=interp)


def _to_nchw_rgb(canvas: np.ndarray) -> np.ndarray:
    """Canvas BGR uint8 -> tenseur NCHW RGB float32 dans [0, 1]."""
    rgb = canvas[:, :, ::-1].astype(np.float32) / 255.0
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None, ...])


def preprocess_yolo(img_bgr: np.ndarray, imgsz: int
                    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Letterbox -> tenseur NCHW RGB [0, 1], plus la géométrie du letterbox.

    Le ratio et le décalage sont indispensables au décodage : sans eux, les
    boîtes restent dans le repère du canvas au lieu de celui de l'image.
    """
    canvas, ratio, pad = letterbox(img_bgr, imgsz)
    return _to_nchw_rgb(canvas), ratio, pad


def preprocess_yolo_chars(img_bgr: np.ndarray, imgsz: int
                          ) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Étirement carré -> tenseur NCHW RGB [0, 1], plus les DEUX facteurs.

    L'étirement est ANISOTROPE : il ne se décrit pas par un ratio scalaire.
    `decode_detections` n'en accepte qu'un seul, donc on décode dans le
    repère du canvas (ratio=1, pad=0) puis on remet à l'échelle axe par axe
    avec `rescale_from_stretch`.
    """
    h, w = img_bgr.shape[:2]
    canvas = stretch_to_square(img_bgr, imgsz)
    return _to_nchw_rgb(canvas), (w / float(imgsz), h / float(imgsz))


def rescale_from_stretch(dets: Sequence[Detection],
                         facteurs: Tuple[float, float]) -> List[Detection]:
    """Ramène des boîtes du canvas carré vers le repère de l'imagette.

    Deux facteurs, un par axe : c'est ce qu'un ratio scalaire ne peut pas
    exprimer. Le tri des caractères par abscisse ne dépend pas de cette
    remise à l'échelle (elle est monotone), mais les boîtes rendues au
    pipeline, elles, doivent être justes.
    """
    fx, fy = facteurs
    return [Detection(d.x1 * fx, d.y1 * fy, d.x2 * fx, d.y2 * fy, d.conf, d.cls)
            for d in dets]


# ---------------------------------------------------------------------------
def build_feeds(family: str, img_bgr: np.ndarray, imgsz: int,
                input_names: Sequence[str],
                rec_height: int = REC_HEIGHT_DEFAUT,
                rec_width: int = REC_WIDTH_DEFAUT) -> Dict[str, np.ndarray]:
    """Image BGR -> dictionnaire {nom de tenseur: tableau prêt à copier}.

    Le dictionnaire, plutôt qu'un tableau unique, parce que D-FINE a DEUX
    entrées : `images` et `orig_target_sizes`. Un backend qui ne garderait
    que la dernière casserait en silence.
    """
    verify_model(family)
    if not input_names:
        raise ValueError("Aucun nom de tenseur d'entrée fourni.")

    if family == "yolo":
        tenseur, _, _ = preprocess_yolo(img_bgr, imgsz)
        return {input_names[0]: tenseur}

    if family == "yolo_chars":
        tenseur, _ = preprocess_yolo_chars(img_bgr, imgsz)
        return {input_names[0]: tenseur}

    if family == "dfine":
        images, sizes = dfine_family.preprocess_dfine(img_bgr, imgsz)
        return {dfine_family.INPUT_IMAGES: images,
                dfine_family.INPUT_SIZES: sizes}

    tenseur = preprocess_rec(img_bgr, hauteur=rec_height, largeur_max=rec_width)
    return {input_names[0]: tenseur}


def yolo_geometry(img_bgr: np.ndarray, imgsz: int
                  ) -> Tuple[float, Tuple[int, int]]:
    """(ratio, pad) du letterbox, pour ramener les boîtes dans l'image."""
    _, ratio, pad = letterbox(img_bgr, imgsz)
    return ratio, pad


def stretch_factors(img_bgr: np.ndarray, imgsz: int) -> Tuple[float, float]:
    """(fx, fy) de l'étirement carré, pour la remise à l'échelle."""
    h, w = img_bgr.shape[:2]
    return w / float(imgsz), h / float(imgsz)