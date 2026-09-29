#!/usr/bin/env python3
"""
Enregistrement local des artefacts produits par le pipeline : image source,
frame vidéo annotée, crop de plaque -- organisés pour être consommés plus
tard par un dashboard.

ARCHITECTURE (100% locale + MQTT, pas de cloud) :
  - Les crops sont écrits localement (léger) et, si mqtt_hook est fourni,
    ENVOYÉS DIRECTEMENT dans le message MQTT (compressés, base64) -- voir
    src/common/mqtt_publisher.py. Pas de S3, pas de dépendance cloud.
  - Les événements (texte de plaque + métadonnées) sont écrits dans
    events.jsonl EN LOCAL (résilience si le réseau MQTT tombe), et publiés
    en temps réel via mqtt_hook si configuré.
  - Comme on n'archive plus rien dans le cloud, le stockage local doit être
    borné : `retention_days` purge automatiquement les sessions trop
    anciennes au démarrage, pour ne pas remplir la carte SD du Pi.

STRUCTURE SUR DISQUE (sous `output_root`, par défaut outputs/) :
    outputs/
      YYYY-MM-DD/
        session_<timestamp>/
          frames/           <- frame pleine résolution (désactivé par défaut)
          crops/             <- crop de chaque plaque détectée
          events.jsonl       <- 1 ligne JSON par détection/lecture
"""
import json
import shutil
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from src.utils.logger import get_logger

log = get_logger("storage")


@dataclass
class StoredEvent:
    timestamp: float
    frame_idx: Optional[int]
    vehicle_track_id: Optional[int]
    vehicle_cls: int
    plate_text: str
    is_locked: bool
    frame_path: Optional[str]
    crop_path: Optional[str]


class ResultStorage:
    """
    Usage:
        storage = ResultStorage(save=True, output_root="outputs", retention_days=3)
        storage.save_frame(frame, frame_idx)          # optionnel, throttle conseillé
        crop_path = storage.save_crop(crop, vehicle_track_id, frame_idx)
        storage.save_event(plate_text=..., vehicle_cls=..., crop_path=crop_path, crop=crop)
        storage.close()                                # IMPORTANT : à appeler en fin de programme
    """

    def __init__(self, save: bool = False, output_root: str = "outputs",
                 save_full_frames: bool = False,   # <- désactivé par défaut : c'est le point lourd
                 frame_save_every_n: int = 30,
                 retention_days: Optional[int] = 3,   # <- purge auto : plus besoin de cloud pour rester léger
                 mqtt_hook: Optional[Callable[..., None]] = None):
        self.save = save
        self.output_root = Path(output_root)
        self.save_full_frames = save_full_frames
        self.frame_save_every_n = frame_save_every_n
        self.retention_days = retention_days
        self.mqtt_hook = mqtt_hook

        self._events_file = None
        if self.save:
            if self.retention_days:
                self._cleanup_old_sessions()

            session_ts = time.strftime("%Y%m%d-%H%M%S")
            day = time.strftime("%Y-%m-%d")
            self.session_dir = self.output_root / day / f"session_{session_ts}"
            self.frames_dir = self.session_dir / "frames"
            self.crops_dir = self.session_dir / "crops"
            self.frames_dir.mkdir(parents=True, exist_ok=True)
            self.crops_dir.mkdir(parents=True, exist_ok=True)

            self.events_path = self.session_dir / "events.jsonl"
            self._events_file = open(self.events_path, "a", encoding="utf-8")
            log.info(f"Session de stockage initialisée : {self.session_dir}")
        else:
            log.info("Stockage local désactivé (save=False) -- seul mqtt_hook, si fourni, publiera.")

    def _cleanup_old_sessions(self) -> None:
        """Supprime les dossiers YYYY-MM-DD plus vieux que retention_days.
        Pas de cloud -> le local doit s'auto-nettoyer."""
        if not self.output_root.exists() or self.retention_days is None:
            return
        if self.retention_days <= 0:
            return
        cutoff = time.time() - self.retention_days * 86400
        removed = 0
        
        for day_dir in self.output_root.iterdir():
            if not day_dir.is_dir():
                continue
            try:
                day_ts = time.mktime(time.strptime(day_dir.name, "%Y-%m-%d"))
            except ValueError:
                continue
            if day_ts < cutoff:
                shutil.rmtree(day_dir, ignore_errors=True)
                removed += 1
        if removed:
            log.info(f"Nettoyage stockage local : {removed} jour(s) > {self.retention_days}j supprimé(s)")

    def save_frame(self, frame: np.ndarray, frame_idx: Optional[int] = None) -> Optional[Path]:
        if not self.save or not self.save_full_frames:
            return None
        if frame_idx is not None and self.frame_save_every_n > 1:
            if frame_idx % self.frame_save_every_n != 0:
                return None

        name = f"frame_{frame_idx if frame_idx is not None else int(time.time()*1000)}.jpg"
        path = self.frames_dir / name
        ok = cv2.imwrite(str(path), frame)
        if not ok:
            log.warning(f"Échec écriture frame : {path}")
            return None
        return path

    def save_crop(self, crop: np.ndarray, vehicle_track_id: Optional[int],
                  frame_idx: Optional[int] = None) -> Optional[Path]:
        """Sauvegarde le crop d'une plaque détectée (copie locale légère)."""
        if not self.save:
            return None
        if crop is None or crop.size == 0:
            return None
        vid_part = f"vid{vehicle_track_id}" if vehicle_track_id is not None else "novid"
        idx_part = f"f{frame_idx}" if frame_idx is not None else str(int(time.time() * 1000))
        name = f"plate_{vid_part}_{idx_part}.jpg"
        path = self.crops_dir / name
        ok = cv2.imwrite(str(path), crop)
        if not ok:
            log.warning(f"Échec écriture crop : {path}")
            return None
        return path

    def save_event(self, plate_text: str, vehicle_cls: int,
                   vehicle_track_id: Optional[int] = None,
                   is_locked: bool = False,
                   frame_idx: Optional[int] = None,
                   frame_path: Optional[Path] = None,
                   crop_path: Optional[Path] = None,
                   crop: Optional[np.ndarray] = None) -> None:
        """
        Ajoute une ligne JSON dans events.jsonl (si save=True), ET publie le
        même événement via mqtt_hook si configuré. Si `crop` est fourni,
        l'image est envoyée compressée dans le message MQTT lui-même
        (voir MQTTPublisher._encode_crop) -- pas de stockage cloud requis.
        """
        event = StoredEvent(
            timestamp=time.time(),
            frame_idx=frame_idx,
            vehicle_track_id=vehicle_track_id,
            vehicle_cls=vehicle_cls,
            plate_text=plate_text,
            is_locked=is_locked,
            frame_path=str(frame_path) if frame_path else None,
            crop_path=str(crop_path) if crop_path else None,
        )
        payload = asdict(event)

        if self._events_file is not None:
            self._events_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._events_file.flush()  # flush immédiat: un crash ne doit pas perdre les events

        if self.mqtt_hook is not None:
            try:
                self.mqtt_hook(payload, crop_bgr=crop)
            except Exception:
                log.exception("Échec de mqtt_hook (non bloquant)")

    def close(self) -> None:
        """À appeler en fin de programme (ex: bloc finally de main.py)."""
        if self._events_file is not None and not self._events_file.closed:
            self._events_file.close()
            log.info(f"Session de stockage fermée : {getattr(self, 'session_dir', None)}")