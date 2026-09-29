#!/usr/bin/env python3
"""
Source de frames unifiée (image / vidéo / webcam / caméra CSI Raspberry Pi).
"""
from __future__ import annotations

import glob as _glob
import threading
from pathlib import Path
from typing import Iterator, List, Optional

import cv2
import numpy as np

from src.utils.logger import get_logger

log = get_logger("capture")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class ImageSource:
    """Accepte un fichier, un dossier, ou un motif glob ('data/*.png')."""

    def __init__(self, path: str):
        p = Path(path)
        if p.is_dir():
            self.paths: List[Path] = sorted(
                f for f in p.iterdir()
                if f.is_file() and f.suffix.lower() in IMAGE_EXTS)
            if not self.paths:
                raise FileNotFoundError(f"Aucune image dans le dossier : {p}")
            log.info("%d image(s) trouvée(s) dans %s", len(self.paths), p)
        elif any(c in str(path) for c in "*?["):
            self.paths = sorted(Path(f) for f in _glob.glob(str(path)))
            if not self.paths:
                raise FileNotFoundError(f"Aucune image ne correspond à : {path}")
            log.info("%d image(s) correspondent à %s", len(self.paths), path)
        else:
            if not p.is_file():
                raise FileNotFoundError(f"Image introuvable : {p}")
            self.paths = [p]
        self.current: Optional[Path] = None

    def frames(self) -> Iterator[np.ndarray]:
        for p in self.paths:
            img = cv2.imread(str(p))
            if img is None:
                log.warning("Image illisible, ignorée : %s", p)
                continue
            self.current = p
            yield img

    def release(self) -> None:
        pass


class VideoSource:
    """Fichier vidéo (imageio-ffmpeg) ou webcam locale (index numérique, V4L2)."""

    def __init__(self, path: str, width: int = 1280, height: int = 720):
        self.is_live = str(path).isdigit()
        if self.is_live:
            self._mode = "webcam"
            self.cap = cv2.VideoCapture(int(path))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # limite le retard côté driver
            if not self.cap.isOpened():
                raise RuntimeError(f"Impossible d'ouvrir la webcam : {path}")
        else:
            self._mode = "file"
            import imageio_ffmpeg
            self._reader = imageio_ffmpeg.read_frames(path)
            meta = next(self._reader)
            self._size = meta["size"]

    def frames(self) -> Iterator[np.ndarray]:
        if self._mode == "webcam":
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    break
                yield frame
        else:
            w, h = self._size
            for frame_bytes in self._reader:
                frame_rgb = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((h, w, 3))
                yield cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    def release(self) -> None:
        if self._mode == "webcam":
            self.cap.release()
        else:
            self._reader.close()


class CSISource:
    """Caméra Raspberry Pi (IMX708) via rpicam-vid (libcamera), sans picamera2."""

    is_live = True
    _SOI = b"\xff\xd8"
    _EOI = b"\xff\xd9"

    def __init__(self, width: int = 1280, height: int = 720, framerate: int = 30):
        import subprocess
        cmd = ["rpicam-vid", "--width", str(width), "--height", str(height),
               "--framerate", str(framerate), "--codec", "mjpeg",
               "--timeout", "0", "--nopreview", "-o", "-"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10 ** 7)
        self._buffer = bytearray()

    def frames(self) -> Iterator[np.ndarray]:
        search_from = 0
        while True:
            chunk = self.proc.stdout.read(65536)
            if not chunk:
                break
            self._buffer += chunk
            while True:
                start = self._buffer.find(self._SOI, search_from)
                if start == -1:
                    # rien d'exploitable : on ne garde pas un buffer infini
                    search_from = max(0, len(self._buffer) - 1)
                    break
                end = self._buffer.find(self._EOI, start + 2)
                if end == -1:
                    search_from = start
                    break
                jpg = bytes(self._buffer[start:end + 2])
                del self._buffer[:end + 2]
                search_from = 0
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    yield frame

    def release(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(timeout=2)
        except Exception:
            self.proc.kill()


class LatestFrameSource:
    """
    Enveloppe une source temps réel : un thread consomme le flux en continu et
    ne conserve que la frame la plus récente. Le pipeline lit toujours du
    présent, jamais du retard accumulé.
    """

    def __init__(self, inner):
        self._inner = inner
        self._frame: Optional[np.ndarray] = None
        self._new = threading.Event()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.dropped = 0
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        try:
            for frame in self._inner.frames():
                if self._stop.is_set():
                    break
                with self._lock:
                    if self._frame is not None and not self._new.is_set():
                        pass
                    elif self._frame is not None:
                        self.dropped += 1
                    self._frame = frame
                self._new.set()
        except Exception:
            log.exception("Thread de capture arrêté sur erreur")
        finally:
            self._new.set()

    def frames(self) -> Iterator[np.ndarray]:
        while not self._stop.is_set():
            if not self._new.wait(timeout=5.0):
                log.warning("Aucune frame depuis 5 s — la caméra répond-elle ?")
                continue
            with self._lock:
                frame = self._frame
                self._new.clear()
            if frame is None:
                break
            yield frame

    def release(self) -> None:
        self._stop.set()
        self._new.set()
        self._inner.release()
        self._thread.join(timeout=2.0)
        if self.dropped:
            log.info("Frames sautées par le thread de capture : %d", self.dropped)


def build_source(source_type: str, input_path: Optional[str],
                 width: int = 1280, height: int = 720,
                 framerate: int = 30, drop_late: Optional[bool] = None):
    """
    drop_late=None -> automatique : True pour csi/webcam, False pour un fichier.
    """
    if source_type == "image":
        if not input_path:
            raise ValueError("--input est requis pour --source image")
        return ImageSource(input_path)

    if source_type == "video":
        if not input_path:
            raise ValueError("--input est requis pour --source video "
                             "(fichier, index webcam, ou URL rtsp/http)")
        src = VideoSource(input_path, width, height)
    elif source_type == "csi":
        src = CSISource(width, height, framerate)
    else:
        raise ValueError(f"source_type inconnu : {source_type}")

    if drop_late is None:
        drop_late = getattr(src, "is_live", False)
    return LatestFrameSource(src) if drop_late else src