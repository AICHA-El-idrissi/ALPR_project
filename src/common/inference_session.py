#!/usr/bin/env python3
"""
Session d'inference : toute la logique metier (detecteur vehicules+plaque,
OCR, trackers, association, vote temporel). pipeline.py ne fait que fournir
des frames.

Ce module ne connait AUCUN moteur d'inference. Il recoit du registre des
objets qui exposent `detect()` ou `read()`, et c'est tout. C'est ce qui
permet de faire tourner D-FINE en ONNX a cote d'un OCR ncnn sans qu'une
ligne d'ici ne change.

Deux familles peuvent occuper l'etage OCR, et elles produisent du texte
differemment :

    yolo_chars   detecte des boites de caracteres -> tri par abscisse
                 -> traduction par idx_to_char
    paddle_rec   rend une sequence -> decodage CTC dans le wrapper

`_read_plate` absorbe cette difference. Le reste du pipeline ne voit que du
texte.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.models.postprocess import Detection, sharpness, sort_ocr_detections
from src.common.voting import (PlateVoter, associate_plates_to_vehicles,
                               box_center_inside)
from src.tracking.byte_tracker import ByteTrackLite, Track
from src.utils.logger import get_logger
from src.utils.metrics import StageTimer

log = get_logger("inference_session")


@dataclass
class PlateResult:
    vehicle_track_id: Optional[int]
    vehicle_cls: int
    plate_box: np.ndarray
    plate_text: str
    conf: float = 0.0
    plate_crop: Optional[np.ndarray] = None
    is_locked: bool = False
    just_locked: bool = False          # True uniquement sur la frame du verrouillage
    det_ms: float = 0.0
    ocr_ms: float = 0.0


class InferenceSession:

    def __init__(self, edge_type: str, detector_dir: str, ocr_dir: str,
                 detector_imgsz: int, ocr_imgsz: int,
                 num_detector_classes: int, idx_to_char: Dict[int, str],
                 plate_cls_id: int,
                 detector_family: str = "yolo",
                 ocr_family: str = "yolo_chars",
                 ocr_charset: Optional[Sequence[str]] = None,
                 rec_height: int = 48,
                 rec_width: int = 320,
                 backend: Optional[str] = None,
                 frame_skip: int = 3,
                 vehicle_track_buffer: int = 30,
                 plate_track_buffer: int = 10,
                 min_votes_before_lock: int = 5,
                 vote_window: int = 15,
                 detector_conf_thres: float = 0.35,
                 detector_iou_thres: float = 0.45,
                 ocr_conf_thres: float = 0.25,
                 ocr_iou_thres: float = 0.45,
                 min_plate_width: int = 24,
                 min_sharpness: float = 0.0,
                 num_threads: int = 4):

        self.edge_type = edge_type
        self.idx_to_char = idx_to_char
        self.plate_cls_id = plate_cls_id
        self.frame_skip = max(1, int(frame_skip))
        self.detector_conf_thres = detector_conf_thres
        self.detector_iou_thres = detector_iou_thres
        self.ocr_conf_thres = ocr_conf_thres
        self.ocr_iou_thres = ocr_iou_thres
        self.min_plate_width = min_plate_width
        self.min_sharpness = min_sharpness
        self.detector_family = detector_family
        self.ocr_family = ocr_family

        self.vehicle_tracker = ByteTrackLite(track_buffer=vehicle_track_buffer)
        self.plate_tracker = ByteTrackLite(track_buffer=plate_track_buffer)
        self.voter = PlateVoter(min_votes_before_lock=min_votes_before_lock,
                                window=vote_window)

        self._active_vehicle_ids: set = set()
        self._announced_locks: set = set()
        self.timer = StageTimer()

        self.load_models(edge_type, detector_dir, ocr_dir, detector_imgsz,
                         ocr_imgsz, num_detector_classes, idx_to_char,
                         num_threads, detector_family, ocr_family,
                         ocr_charset, rec_height, rec_width, backend)

    # ------------------------------------------------------------------
    def load_models(self, edge_type: str, detector_dir: str, ocr_dir: str,
                    detector_imgsz: int, ocr_imgsz: int,
                    num_detector_classes: int, idx_to_char: Dict[int, str],
                    num_threads: int = 4,
                    detector_family: str = "yolo",
                    ocr_family: str = "yolo_chars",
                    ocr_charset: Optional[Sequence[str]] = None,
                    rec_height: int = 48, rec_width: int = 320,
                    backend: Optional[str] = None) -> None:
        """Monte les deux etages via le registre.

        Le backend n'est plus deduit d'`edge_type` : il vient du chemin du
        modele, ou de `backend` si on veut le forcer. C'est ce qui permet de
        faire tourner D-FINE en ONNX sur la meme carte que le reste, et de
        comparer deux moteurs sans toucher au pipeline.

        `edge_type` ne sert plus qu'a choisir un defaut quand le chemin ne
        tranche pas.
        """
        from src.common.preprocess import OCR_FAMILIES
        from src.wrappers.registry import build_model

        if ocr_family not in OCR_FAMILIES:
            raise ValueError(
                f"ocr_family={ocr_family!r} n'est pas un etage OCR. "
                f"Attendu : {OCR_FAMILIES}")

        # Le charset fixe le nombre de classes de l'etage OCR. Pour
        # yolo_chars, une classe par caractere. Pour paddle_rec, une de plus
        # (le blanc CTC), ajoutee par le decodeur et non par le modele.
        if ocr_family == "paddle_rec":
            if not ocr_charset:
                raise ValueError(
                    "ocr_family='paddle_rec' exige `ocr_charset` : sans lui, "
                    "les indices CTC ne peuvent pas etre traduits en texte.")
            charset = list(ocr_charset)
            n_ocr_classes = len(charset) + 1
        else:
            if not idx_to_char:
                raise ValueError(
                    "idx_to_char est vide : le nombre de classes OCR ne peut "
                    "pas etre determine.")
            charset = [idx_to_char[k] for k in sorted(idx_to_char)]
            n_ocr_classes = len(idx_to_char)

        if edge_type not in ("rpi", "jetson") and backend is None:
            raise ValueError(
                f"edge_type inconnu : {edge_type!r} (attendu 'rpi' ou "
                f"'jetson'), et aucun `backend` explicite fourni.")
        defaut = {"rpi": "ncnn", "jetson": "tensorrt"}.get(edge_type)

        t0 = time.perf_counter()
        self.detector = build_model(
            detector_dir, family=detector_family, imgsz=detector_imgsz,
            num_classes=num_detector_classes,
            backend=backend or self._backend_ou_defaut(detector_dir, defaut),
            num_threads=num_threads)
        self.ocr = build_model(
            ocr_dir, family=ocr_family, imgsz=ocr_imgsz,
            num_classes=n_ocr_classes, charset=charset,
            rec_height=rec_height, rec_width=rec_width,
            backend=backend or self._backend_ou_defaut(ocr_dir, defaut),
            num_threads=num_threads)

        self.detector_family = detector_family
        self.ocr_family = ocr_family

        log.info("Detecteur : %s (%s/%s, imgsz=%d, nc=%d)", detector_dir,
                 detector_family, self.detector.backend_name, detector_imgsz,
                 num_detector_classes)
        log.info("OCR : %s (%s/%s, imgsz=%d, nc=%d)", ocr_dir, ocr_family,
                 self.ocr.backend_name, ocr_imgsz, n_ocr_classes)
        log.info("Chargement en %.2f s", time.perf_counter() - t0)

        self.warmup(detector_imgsz, ocr_imgsz)

    @staticmethod
    def _backend_ou_defaut(chemin: str, defaut: Optional[str]) -> Optional[str]:
        """Laisse le registre deviner depuis le chemin ; ne force le defaut de
        la plateforme que si le chemin ne tranche pas."""
        from src.common.base import deviner_backend
        try:
            return deviner_backend(chemin)
        except (ValueError, FileNotFoundError):
            return defaut

    def warmup(self, detector_imgsz: int, ocr_imgsz: int, runs: int = 2) -> None:
        """La 1re inference alloue les buffers : sans warmup, le FPS mesure au
        demarrage est faux et la 1re frame reelle est 3-4x plus lente.

        L'etage OCR se reveille par `read()` ou par `detect()` selon sa
        famille -- appeler `detect()` sur un paddle_rec leverait un TypeError.
        """
        dummy = np.zeros((detector_imgsz, detector_imgsz, 3), dtype=np.uint8)
        small = np.zeros((ocr_imgsz // 4, ocr_imgsz // 2, 3), dtype=np.uint8)
        for _ in range(runs):
            try:
                self.detector.detect(dummy, conf_thres=0.99)
                if self.ocr_family == "paddle_rec":
                    self.ocr.read(small)
                else:
                    self.ocr.detect(small, conf_thres=0.99,
                                    class_agnostic_nms=True)
            except Exception:
                log.debug("Warmup ignore (backend non pret)", exc_info=True)
                return
        log.info("Warmup termine (%d passes)", runs)

    # ------------------------------------------------------------------
    # Mode image
    # ------------------------------------------------------------------
    def process_image(self, frame: np.ndarray
                      ) -> Tuple[List[PlateResult], List[Detection]]:
        t0 = time.perf_counter()
        dets = self.detector.detect(frame, conf_thres=self.detector_conf_thres,
                                    iou_thres=self.detector_iou_thres,
                                    class_agnostic_nms=False)
        det_ms = (time.perf_counter() - t0) * 1000.0
        self.timer.record("detect", det_ms)

        vehicle_dets = [d for d in dets if d.cls != self.plate_cls_id]
        plate_dets = [d for d in dets if d.cls == self.plate_cls_id]
        log.info("Detections : %d (%d vehicule(s), %d plaque(s)) en %.1f ms",
                 len(dets), len(vehicle_dets), len(plate_dets), det_ms)

        results: List[PlateResult] = []
        for pd in plate_dets:
            plate_box = pd.box
            # Le vehicule porteur : le plus petit dont la boite contient le
            # centre de la plaque.
            candidates = [vd for vd in vehicle_dets
                          if box_center_inside(plate_box, vd.box)]
            vehicle_cls = min(candidates, key=lambda vd: vd.area).cls \
                if candidates else -1

            t1 = time.perf_counter()
            plate_text, crop = self._read_plate(frame, plate_box)
            ocr_ms = (time.perf_counter() - t1) * 1000.0
            self.timer.record("ocr", ocr_ms)

            results.append(PlateResult(
                vehicle_track_id=None,
                vehicle_cls=vehicle_cls,
                plate_box=plate_box,
                plate_text=plate_text,
                conf=pd.conf,
                plate_crop=crop,
                det_ms=det_ms,
                ocr_ms=ocr_ms,
            ))
        return results, dets

    # ------------------------------------------------------------------
    # Mode video / CSI
    # ------------------------------------------------------------------
    def process_video_frame(self, frame: np.ndarray, frame_idx: int
                            ) -> Tuple[List[PlateResult], List[Detection]]:
        run_detector = (frame_idx % self.frame_skip == 0)
        dets: List[Detection] = []
        det_ms = 0.0

        if run_detector:
            t0 = time.perf_counter()
            dets = self.detector.detect(frame,
                                        conf_thres=self.detector_conf_thres,
                                        iou_thres=self.detector_iou_thres,
                                        class_agnostic_nms=False)
            det_ms = (time.perf_counter() - t0) * 1000.0
            self.timer.record("detect", det_ms)

            vehicle_dets = [(d.x1, d.y1, d.x2, d.y2, d.conf, d.cls)
                            for d in dets if d.cls != self.plate_cls_id]
            plate_dets = [(d.x1, d.y1, d.x2, d.y2, d.conf, d.cls)
                          for d in dets if d.cls == self.plate_cls_id]
            vehicle_tracks = self.vehicle_tracker.update(vehicle_dets)
            plate_tracks = self.plate_tracker.update(plate_dets)
        else:
            vehicle_tracks = self.vehicle_tracker.predict_only()
            plate_tracks = self.plate_tracker.predict_only()

        self._purge_lost_votes(vehicle_tracks)

        assoc = associate_plates_to_vehicles(plate_tracks, vehicle_tracks)
        vehicle_by_id = {vt.track_id: vt for vt in vehicle_tracks}

        results: List[PlateResult] = []
        for pt in plate_tracks:
            vehicle_id = assoc.get(pt.track_id)
            if vehicle_id is None or vehicle_id not in vehicle_by_id:
                continue

            already_locked = vehicle_id in self.voter.locked
            crop = None
            ocr_ms = 0.0

            # OCR seulement si le detecteur a tourne sur cette frame ET que la
            # plaque n'est pas deja verrouillee.
            if run_detector and not already_locked:
                t1 = time.perf_counter()
                plate_text, crop = self._read_plate(frame, pt.box)
                ocr_ms = (time.perf_counter() - t1) * 1000.0
                self.timer.record("ocr", ocr_ms)
                if plate_text:
                    self.voter.add_reading(vehicle_id, plate_text)

            guess = self.voter.best_guess(vehicle_id)
            if guess is None:
                continue

            is_locked = vehicle_id in self.voter.locked
            just_locked = is_locked and vehicle_id not in self._announced_locks
            if just_locked:
                self._announced_locks.add(vehicle_id)
                if crop is None:
                    crop = self._crop(frame, pt.box)   # image du verrou
                log.info("Plaque verrouillee — vehicle_track=%s plate=%s",
                         vehicle_id, guess)

            results.append(PlateResult(
                vehicle_track_id=vehicle_id,
                vehicle_cls=vehicle_by_id[vehicle_id].cls,
                plate_box=np.asarray(pt.box, dtype=np.float32),
                plate_text=guess,
                conf=float(getattr(pt, "score", 0.0) or 0.0),
                plate_crop=crop,
                is_locked=is_locked,
                just_locked=just_locked,
                det_ms=det_ms,
                ocr_ms=ocr_ms,
            ))

        return results, dets

    # ------------------------------------------------------------------
    def _crop(self, frame: np.ndarray, plate_box) -> Optional[np.ndarray]:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in plate_box]
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, w), min(y2, h)
        if x2 - x1 < 2 or y2 - y1 < 2:
            return None
        return frame[y1:y2, x1:x2]

    def _read_plate(self, frame: np.ndarray, plate_box
                    ) -> Tuple[str, Optional[np.ndarray]]:
        """Lit une plaque, quel que soit le type d'etage OCR monte.

        Deux familles, deux facons de produire du texte :

          yolo_chars  detecte des boites de caracteres, qu'il faut trier par
                      abscisse puis traduire par idx_to_char ;
          paddle_rec  rend directement une sequence, decodee par CTC dans le
                      wrapper.

        Le pipeline ne voit que du texte : c'est ici que la difference est
        absorbee, et nulle part ailleurs.
        """
        crop = self._crop(frame, plate_box)
        if crop is None or crop.size == 0:
            return "", None
        if crop.shape[1] < self.min_plate_width:
            return "", crop        # trop petite : illisible, on ne paie pas l'OCR
        if self.min_sharpness > 0 and sharpness(crop) < self.min_sharpness:
            return "", crop        # trop floue

        if self.ocr_family == "paddle_rec":
            texte, conf, _ = self.ocr.read(crop)
            if conf < self.ocr_conf_thres:
                return "", crop    # lecture trop peu sure
            return texte, crop

        char_dets = self.ocr.detect(crop, conf_thres=self.ocr_conf_thres,
                                    iou_thres=self.ocr_iou_thres,
                                    class_agnostic_nms=True)
        char_dets = sort_ocr_detections(char_dets)
        text = "".join(self.idx_to_char.get(c.cls, "?") for c in char_dets)
        return text, crop

    def _purge_lost_votes(self, vehicle_tracks: List[Track]) -> None:
        current_ids = {t.track_id for t in vehicle_tracks}
        for lost_id in self._active_vehicle_ids - current_ids:
            self.voter.forget(lost_id)
            self._announced_locks.discard(lost_id)
        self._active_vehicle_ids = current_ids

    def stats(self) -> Dict[str, Any]:
        return {"det_ms": round(self.timer.ms("detect"), 1),
                "ocr_ms": round(self.timer.ms("ocr"), 1),
                "locked_plates": len(self.voter.locked),
                "tracked_vehicles": len(self._active_vehicle_ids),
                "detector_family": self.detector_family,
                "ocr_family": self.ocr_family,
                "backend": getattr(self.detector, "backend_name", "?")}