#!/usr/bin/env python3
"""
Post-traitement YOLOv8/v11 (Ultralytics) : letterbox, décodage de la sortie
brute, NMS. Aucune dépendance ncnn/TensorRT -> testable sur PC.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls: int

    @property
    def box(self) -> np.ndarray:
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.float32)

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


# --------------------------------------------------------------------------
# Pré-traitement
# --------------------------------------------------------------------------
def letterbox(img: np.ndarray, imgsz: int,
              color: Tuple[int, int, int] = (114, 114, 114)
              ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Resize + padding en conservant le ratio d'aspect (comme Ultralytics)."""
    h, w = img.shape[:2]
    r = min(imgsz / h, imgsz / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    # INTER_AREA est plus rapide ET plus propre en réduction (cas courant)
    interp = cv2.INTER_AREA if r < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
    canvas = np.full((imgsz, imgsz, 3), color, dtype=np.uint8)
    top = (imgsz - new_h) // 2
    left = (imgsz - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas, r, (left, top)


# --------------------------------------------------------------------------
# Décodage
# --------------------------------------------------------------------------
def decode_detections(out_np: np.ndarray,
                      num_classes: int,
                      ratio: float,
                      pad: Tuple[int, int],
                      orig_shape: Tuple[int, int],
                      conf_thres: float = 0.35,
                      iou_thres: float = 0.45,
                      class_agnostic_nms: bool = False,
                      max_candidates: int = 300) -> List[Detection]:
    """
    Formats acceptés : (1, 4+nc, N), (4+nc, N), (N, 4+nc).

    class_agnostic_nms=True  -> OCR (deux caractères ne peuvent pas occuper
                                la même position).
    class_agnostic_nms=False -> détecteur (une plaque EST dans un véhicule,
                                les deux boîtes se recouvrent légitimement).
    """
    expected = 4 + num_classes
    out = np.squeeze(np.asarray(out_np, dtype=np.float32))

    if out.ndim != 2:
        raise ValueError(f"Sortie invalide : shape={out.shape} ; une matrice 2D "
                         f"est attendue après squeeze().")

    if out.shape[0] == expected:
        preds = out.T                      # (4+nc, N) -> (N, 4+nc)
    elif out.shape[1] == expected:
        preds = out                        # déjà (N, 4+nc), cas TensorRT
    else:
        raise ValueError(f"Sortie YOLO non interprétable : shape={out.shape}, "
                         f"num_classes={num_classes}, attendu={expected} canaux.")

    scores_all = preds[:, 4:expected]
    cls_ids = np.argmax(scores_all, axis=1)
    confs = scores_all[np.arange(scores_all.shape[0]), cls_ids]

    # --- filtrage par confiance : TOUT est filtré ensemble ------------------
    keep = confs >= conf_thres
    if not np.any(keep):
        return []
    boxes = preds[keep, :4]
    confs = confs[keep]
    cls_ids = cls_ids[keep]

    # --- borne le nombre de candidats envoyés à la NMS ----------------------
    if boxes.shape[0] > max_candidates:
        top = np.argpartition(-confs, max_candidates)[:max_candidates]
        boxes, confs, cls_ids = boxes[top], confs[top], cls_ids[top]

    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    pad_x, pad_y = pad
    orig_h, orig_w = orig_shape
    inv = 1.0 / ratio
    x1 = np.clip((cx - w * 0.5 - pad_x) * inv, 0, orig_w)
    y1 = np.clip((cy - h * 0.5 - pad_y) * inv, 0, orig_h)
    x2 = np.clip((cx + w * 0.5 - pad_x) * inv, 0, orig_w)
    y2 = np.clip((cy + h * 0.5 - pad_y) * inv, 0, orig_h)

    # OpenCV NMS attend (x, y, largeur, hauteur)
    if class_agnostic_nms:
        nms_boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1)
    else:
        # Offset par classe : on décale chaque classe dans un plan disjoint,
        # une seule NMS suffit alors pour un résultat "par classe".
        offset = cls_ids.astype(np.float32) * (max(orig_h, orig_w) + 1.0)
        nms_boxes = np.stack([x1 + offset, y1 + offset, x2 - x1, y2 - y1], axis=1)

    idxs = cv2.dnn.NMSBoxes(nms_boxes.tolist(), confs.tolist(),
                            float(conf_thres), float(iou_thres))
    if idxs is None or len(idxs) == 0:
        return []
    idxs = np.asarray(idxs, dtype=np.int64).reshape(-1)

    return [Detection(float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i]),
                      float(confs[i]), int(cls_ids[i])) for i in idxs]


# --------------------------------------------------------------------------
# Utilitaires
# --------------------------------------------------------------------------
def sharpness(img: np.ndarray) -> float:
    """Variance du Laplacien : mesure de netteté (sert à filtrer les crops flous)."""
    if img is None or img.size == 0:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def sort_ocr_detections(dets: List[Detection], axis: str = "x") -> List[Detection]:
    """Trie les caractères gauche->droite (ou haut->bas) pour reconstruire le texte."""
    key = (lambda d: d.x1) if axis == "x" else (lambda d: d.y1)
    return sorted(dets, key=key)