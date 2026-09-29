#!/usr/bin/env python3
"""Interface commune aux backends d'inférence, et matrice de compatibilité.

Trois moteurs (ncnn, TensorRT, ONNX Runtime) x trois familles de modèles
(yolo, dfine, paddle_rec). Toutes les combinaisons ne sont pas possibles, et
l'incompatibilité doit être annoncée AU CHARGEMENT avec sa raison, jamais
découverte sur des résultats aberrants.

Aucun import de moteur ici : ce module reste utilisable sur un PC de dev sans
ncnn, sans TensorRT et sans onnxruntime.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np

from src.models.postprocess import Detection
from src.common.preprocess import  DETECTION_FAMILIES , FAMILIES , RECOGNITION_FAMILIES , verify_model


BACKENDS = ("ncnn", "tensorrt", "onnx")

# ---------------------------------------------------------------------------
# Matrice de compatibilité. La valeur est None quand c'est supporté, sinon
# c'est la RAISON du refus -- affichée telle quelle à l'utilisateur.
# ---------------------------------------------------------------------------
INCOMPATIBILITES: Dict[Tuple[str, str], str] = {
    ("dfine", "ncnn"):
        "D-FINE n'est pas porté vers ncnn dans ce projet. Son décodage DETR "
        "repose sur des opérateurs (GridSample, TopK à K dynamique) dont la "
        "conversion n'a pas été validée, et le gain attendu sur ARM est nul "
        "face à un convolutif. Utilise le backend 'onnx' pour D-FINE.",
}


def verify_backend(backend: str) -> None:
    if backend not in BACKENDS:
        raise ValueError(f"Backend inconnu : {backend!r}. Attendu : {BACKENDS}")


def verify_compatibilities(family: str, backend: str) -> None:
    """Refuse tôt une combinaison non supportée, en disant pourquoi."""
    verify_model(family)
    verify_backend(backend)
    raison = INCOMPATIBILITES.get((family, backend))
    if raison:
        raise NotImplementedError(
            f"La famille {family!r} ne tourne pas sur le backend {backend!r}.\n"
            f"{raison}")


def supported_combinations() -> List[Tuple[str, str]]:
    """Toutes les paires (famille, backend) utilisables. Sert aux tests et à
    l'affichage d'aide."""
    return [(f, b) for f in FAMILIES for b in BACKENDS
            if (f, b) not in INCOMPATIBILITES]


class InferenceBackend(ABC):
    """Contrat commun. Un backend expose soit `detect`, soit `read`, selon la
    famille du modèle qu'il porte -- jamais les deux.

    Les sous-classes doivent renseigner `family`, `imgsz`, `num_classes` et
    `charset` avant d'appeler `_valider()`.
    """

    family: str
    imgsz: int
    num_classes: int
    charset: List[str]
    backend_name: str

    # ------------------------------------------------------------------
    def _valider(self) -> None:
        """Contrôles communs, à appeler à la fin de chaque __init__."""
        verify_compatibilities(self.family, self.backend_name)
        if self.family in RECOGNITION_FAMILIES and not self.charset:
            raise ValueError(
                f"La famille {self.family!r} exige un `charset` : sans lui, "
                f"les indices CTC ne peuvent pas être traduits en texte.")
        if self.family in DETECTION_FAMILIES and self.num_classes < 1:
            raise ValueError(
                f"La famille {self.family!r} exige `num_classes >= 1`, "
                f"reçu {self.num_classes}.")

    @property
    def est_detecteur(self) -> bool:
        return self.family in DETECTION_FAMILIES

    # ------------------------------------------------------------------
    @abstractmethod
    def infer_raw(self, feeds: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Exécute le modèle sur un dictionnaire d'entrées nommées."""

    @abstractmethod
    def detect(self, img_bgr: np.ndarray, conf_thres: float = 0.35,
               iou_thres: float = 0.45,
               class_agnostic_nms: bool = False) -> List[Detection]:
        """Familles de détection uniquement."""

    @abstractmethod
    def read(self, img_bgr: np.ndarray) -> Tuple[str, float, List[str]]:
        """Familles de reconnaissance uniquement."""

    def close(self) -> None:
        """Libère les ressources. Sans effet par défaut."""

    # ------------------------------------------------------------------
    def _refuser_detect(self) -> None:
        raise TypeError(
            f"family={self.family!r} n'est pas un détecteur : utilise read().")

    def _refuser_read(self) -> None:
        raise TypeError(
            f"read() est réservé aux familles {RECOGNITION_FAMILIES}, pas à "
            f"{self.family!r}. Utilise detect().")

    def __repr__(self) -> str:
        return (f"<{type(self).__name__} family={self.family!r} "
                f"backend={self.backend_name!r} imgsz={self.imgsz}>")


def decode_par_famille(family: str, brute: np.ndarray, img_bgr: np.ndarray,
                       imgsz: int, num_classes: int, conf_thres: float,
                       iou_thres: float, class_agnostic_nms: bool
                       ) -> List[Detection]:
    """Décodage d'une tête YOLO, selon la géométrie de la famille.

    `yolo` letterboxe : ratio scalaire et décalage, décodage direct dans le
    repère de l'image.

    `yolo_chars` étire : la transformation est ANISOTROPE et ne se décrit pas
    par un ratio unique. On décode donc dans le repère du canvas (ratio=1,
    pad=0) puis on remet à l'échelle axe par axe. Décoder avec un ratio
    scalaire déformerait toutes les boîtes sans lever d'erreur.
    """
    from src.models.postprocess import decode_detections
    from src.common.preprocess import (rescale_from_stretch,
                                          stretch_factors, yolo_geometry)

    if family == "yolo_chars":
        dets = decode_detections(brute, num_classes, 1.0, (0, 0),
                                 (imgsz, imgsz), conf_thres, iou_thres,
                                 class_agnostic_nms=True)
        return rescale_from_stretch(dets, stretch_factors(img_bgr, imgsz))

    ratio, pad = yolo_geometry(img_bgr, imgsz)
    return decode_detections(brute, num_classes, ratio, pad,
                             img_bgr.shape[:2], conf_thres, iou_thres,
                             class_agnostic_nms)


def deviner_backend(chemin: Union[str, Path]) -> str:
    """Déduit le moteur du chemin fourni, pour éviter un drapeau de plus.

        *.engine              -> tensorrt
        *.onnx                -> onnx
        *.param, ou un dossier contenant model*.param -> ncnn
    """
    p = Path(chemin)
    if p.is_file():
        if p.suffix == ".engine":
            return "tensorrt"
        if p.suffix == ".onnx":
            return "onnx"
        if p.suffix == ".param":
            return "ncnn"
        raise ValueError(
            f"Extension non reconnue : {p.suffix!r} ({p}). "
            f"Attendu .engine, .onnx ou .param.")
    if not p.exists():
        raise FileNotFoundError(f"Chemin de modèle introuvable : {p}")
    if list(p.glob("*.param")):
        return "ncnn"
    if list(p.glob("*.engine")):
        return "tensorrt"
    if list(p.glob("*.onnx")):
        return "onnx"
    raise FileNotFoundError(
        f"Aucun modèle reconnu dans {p} (ni .param, ni .engine, ni .onnx).")