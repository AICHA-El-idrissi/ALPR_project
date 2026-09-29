#!/usr/bin/env python3
"""
Vote temporel sur les lectures OCR d'une même plaque suivie, et association
géométrique plaque <-> véhicule (une plaque appartient au véhicule dont la
boîte contient son centre).
"""
from collections import defaultdict, Counter, deque
from typing import Dict, List, Optional

import numpy as np


def box_center_inside(inner_box: np.ndarray, outer_box: np.ndarray) -> bool:
    """True si le centre de inner_box (x1,y1,x2,y2) est à l'intérieur de outer_box."""
    cx = (inner_box[0] + inner_box[2]) / 2
    cy = (inner_box[1] + inner_box[3]) / 2
    ox1, oy1, ox2, oy2 = outer_box
    return bool(ox1 <= cx <= ox2 and oy1 <= cy <= oy2)


def associate_plates_to_vehicles(plate_tracks, vehicle_tracks) -> Dict[int, int]:
    """
    Retourne {plate_track_id: vehicle_track_id} pour chaque piste de plaque
    dont le centre tombe dans une piste de véhicule. S'il y a plusieurs
    véhicules candidats, on prend le plus petit (le plus proche typiquement).
    """
    assoc = {}
    for pt in plate_tracks:
        candidates = [vt for vt in vehicle_tracks if box_center_inside(pt.box, vt.box)]
        if not candidates:
            continue
        best = min(candidates, key=lambda vt: (vt.box[2] - vt.box[0]) * (vt.box[3] - vt.box[1]))
        assoc[pt.track_id] = best.track_id
    return assoc


class PlateVoter:
    """
    Accumule les lectures OCR successives d'un même véhicule (track_id) et
    ne "verrouille" une lecture qu'après min_votes_before_lock occurrences
    identiques -- évite qu'une seule lecture bruitée pollue le résultat final.
    """

    def __init__(self, min_votes_before_lock: int = 5, window: int = 15):
        self.min_votes_before_lock = min_votes_before_lock
        self.window = window
        self._history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.window))
        self.locked: Dict[int, str] = {}

    def add_reading(self, vehicle_id: int, text: str) -> None:
        if not text:
            return
        self._history[vehicle_id].append(text)

    def best_guess(self, vehicle_id: int) -> Optional[str]:
        if vehicle_id in self.locked:
            return self.locked[vehicle_id]
        hist = self._history.get(vehicle_id)
        if not hist:
            return None
        text, count = Counter(hist).most_common(1)[0]
        if count >= self.min_votes_before_lock:
            self.locked[vehicle_id] = text
        return text

    def forget(self, vehicle_id: int) -> None:
        self._history.pop(vehicle_id, None)
        self.locked.pop(vehicle_id, None)