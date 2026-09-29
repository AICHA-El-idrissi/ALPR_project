#!/usr/bin/env python3
"""
Suivi multi-objets façon ByteTrack, allégé.
Un détecteur classique garde ce qui dépasse 0.35 et jette le reste.
Or une voitire partiellement occultee , ou floue par le mouvement tombe souvent a 0.2
si on la jette , sapiste meurt

Idea :
    1. les détections SÛRES sont associées aux pistes ;
  2. les pistes restées orphelines retentent leur chance avec les détections
     FAIBLES -- qui ne peuvent donc que prolonger une piste existante, jamais
     en créer une nouvelle.

"""
from dataclasses import dataclass, field
from typing import List

import numpy as np
from src.tracking.kalman import KalmanBox


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU de chaque boîte de `a` contre chaque boîte de `b`.
    boite : [x1, y1, x2, y2]
    """
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    inter_x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    inter_y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    inter_x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    inter_y2 = np.minimum(a[:, None, 3], b[None, :, 3])

    inter = np.clip(inter_x2 - inter_x1, 0, None) * np.clip(inter_y2 - inter_y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0).astype(np.float32)


@dataclass(slots=True)
class Track:
    track_id: int
    cls: int
    kf: KalmanBox
    age: int = 0
    hits: int = 1
    time_since_update: int = 0
    conf: float = 0.0
    box: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))

    @property
    def confirmed(self) -> bool:
        """Piste jugée assez stable pour être exploitée en aval"""
        return self.hits >= 3


class ByteTrackLite:

    def __init__(self, track_buffer: int = 30, match_iou_thres: float = 0.3,
                 low_conf_thres: float = 0.1, high_conf_thres: float = 0.5):
        self.track_buffer = track_buffer
        self.match_iou_thres = match_iou_thres
        self.low_conf_thres = low_conf_thres
        self.high_conf_thres = high_conf_thres
        self.tracks: List[Track] = []
        self._next_id = 1

    def _greedy_match(self, tracks: List[Track], dets: np.ndarray):
        n, m = len(tracks), len(dets)
        
        if n == 0 or m == 0:
            return [], list(range(n)), list(range(m))

        pred_boxes = np.empty((n, 4), dtype=np.float32)
        for i, t in enumerate(tracks):
            pred_boxes[i] = t.kf.as_xyxy()

        iou = iou_xyxy(pred_boxes, dets[:, :4])

        track_cls = np.fromiter((t.cls for t in tracks), dtype=np.float32, count=n)
        det_cls = dets[:, 5]
        # iou :nintersection pour les tracks et detections d ememe classe
        same_cls_mask = track_cls[:, None] == det_cls[None, :]
        iou = np.where(same_cls_mask, iou, -1.0)

        flat = iou.ravel()
        # tri décroissant vectorisé, puis on ne garde que les paires
        # au-dessus du seuil AVANT la boucle Python -> beaucoup moins
        # d'itérations que N*M dans la majorité des scènes réelles.
        order = np.argsort(-flat, kind="stable")
        order = order[flat[order] >= self.match_iou_thres]

        matched_t = np.zeros(n, dtype=bool)
        matched_d = np.zeros(m, dtype=bool)
        matches = []
        for flat_idx in order:
            i, j = divmod(int(flat_idx), m)
            if not matched_t[i] and not matched_d[j]:
                matched_t[i] = True
                matched_d[j] = True
                matches.append((i, j))

        unmatched_tracks = np.nonzero(~matched_t)[0].tolist()
        unmatched_dets = np.nonzero(~matched_d)[0].tolist()
        return matches, unmatched_tracks, unmatched_dets

    # Predire les tracks existantes sans mise à jour (pour les orphelines)
    def predict_only(self) -> List[Track]:
        for t in self.tracks:
            t.box = t.kf.predict()
            t.time_since_update += 1
            t.age += 1
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.track_buffer]
        return self.tracks

    def update(self, detections) -> List[Track]:
        
        """detections: iterable de (x1,y1,x2,y2,conf,cls)."""
        if len(detections):
            dets = np.asarray(detections, dtype=np.float32)
        else:
            dets = np.zeros((0, 6), dtype=np.float32)
    #    # 1. prédire les pistes existantes
        self.predict_only()
        
    # trier selon la confiance pour que les plus sûres soient associées en premier
        if len(dets):
            high = dets[dets[:, 4] >= self.high_conf_thres]
            low = dets[(dets[:, 4] >= self.low_conf_thres) & (dets[:, 4] < self.high_conf_thres)]
        else:
            high = dets
            low = dets

        matches, unmatched_tracks, unmatched_high = self._greedy_match(self.tracks, high)
        for ti, di in matches:
            self.tracks[ti].kf.update(high[di, :4])
            self.tracks[ti].conf = high[di, 4]
            self.tracks[ti].box = high[di, :4]
            self.tracks[ti].hits += 1
            self.tracks[ti].time_since_update = 0

        remaining_tracks = [self.tracks[i] for i in unmatched_tracks]
        
        # 2 tour : associer les pistes orphelines aux détections FAIBLES
        matches2, _, _ = self._greedy_match(remaining_tracks, low)
        for ti_rel, di in matches2:
            t = remaining_tracks[ti_rel]
            t.kf.update(low[di, :4])
            t.conf = low[di, 4]
            t.box = low[di, :4]
            t.hits += 1
            t.time_since_update = 0

        for di in unmatched_high:
            kf = KalmanBox(high[di, :4])
            self.tracks.append(Track(track_id=self._next_id, cls=int(high[di, 5]),
                                       kf=kf, conf=high[di, 4], box=high[di, :4]))
            self._next_id += 1

        # (filtrage déjà fait par predict_only ; les nouvelles pistes ont
        # time_since_update=0, donc pas besoin de refiltrer ici)

        return [t for t in self.tracks if t.time_since_update == 0]