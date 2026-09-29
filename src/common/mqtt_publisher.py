#!/usr/bin/env python3
"""
Publisher MQTT pour les événements ALPR : texte de plaque + métadonnées, crop
compressé en base64, aperçu vidéo annoté et statistiques (FPS, latences).

Topics publiés :
    <base>/<device>/status    (retain)  online / offline
    <base>/<device>/events              une plaque lue / verrouillée
    <base>/<device>/stats               FPS + latences, ~1 Hz
    <base>/<device>/preview             JPEG annoté basse résolution, ~2 Hz

Test rapide :
    

"""
from __future__ import annotations

import base64
import json
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, Optional, Tuple

import cv2
import numpy as np

from src.utils.logger import get_logger

log = get_logger("mqtt_publisher")

_Msg = Tuple[str, str, bool]  # (topic, payload, retain)


def _make_client(client_id: str):
    """paho >= 2.0 exige CallbackAPIVersion ; paho 1.x ne le connaît pas."""
    import paho.mqtt.client as mqtt

    api = getattr(mqtt, "CallbackAPIVersion", None)
    if api is not None:
        return mqtt.Client(api.VERSION1, client_id=client_id)
    return mqtt.Client(client_id=client_id)


def _mqtt_cfg(cfg: dict) -> dict:
    """Accepte la section sous `runtime.mqtt` (config.yaml actuel) ou `mqtt`."""
    return (cfg.get("runtime", {}) or {}).get("mqtt") or cfg.get("mqtt") or {}


class MQTTPublisher:
    """
    Ne lève jamais : si le broker est absent, le pipeline continue et les
    messages sont mis en file (bornée) puis rejoués à la reconnexion.
    """

    def __init__(self, cfg: dict):
        m = _mqtt_cfg(cfg)

        self.enabled: bool = bool(m.get("enabled", True))
        self.base_topic: str = m.get("topic", "alpr-edge")
        self.device_id: str = m.get("client_id", "raspi01")
        # `broker` et `host` coexistaient dans le YAML : on accepte les deux.
        self.host: str = m.get("broker") or m.get("host") or "localhost"
        self.port: int = int(m.get("port", 1883))
        self.keepalive: int = int(m.get("keepalive", 60))
        self.qos: int = int(m.get("qos", 1))
        self.username: Optional[str] = m.get("username")
        self.password: Optional[str] = m.get("password")
        self.use_tls: bool = bool(m.get("use_tls", False))

        self.image_max_side: int = int(m.get("image_max_side", 240))
        self.image_jpeg_quality: int = int(m.get("image_jpeg_quality", 60))
        self.preview_max_side: int = int(m.get("preview_max_side", 640))
        self.preview_jpeg_quality: int = int(m.get("preview_jpeg_quality", 55))

        self.max_pending: int = int(m.get("max_pending_messages", 500))
        self._pending: Deque[_Msg] = deque(maxlen=self.max_pending)

        self._connected = threading.Event()
        self._lock = threading.Lock()
        self.client: Optional[Any] = None
        self.dropped = 0
        self.sent = 0

        self.topic_events = f"{self.base_topic}/{self.device_id}/events"
        self.topic_stats = f"{self.base_topic}/{self.device_id}/stats"
        self.topic_status = f"{self.base_topic}/{self.device_id}/status"
        self.topic_preview = f"{self.base_topic}/{self.device_id}/preview"

    # ------------------------------------------------------------------
    @property
    def connected(self) -> bool:
        return self._connected.is_set()

    def connect(self, timeout: float = 5.0) -> bool:
        """Ouvre la connexion en arrière-plan. Retourne False sans bloquer le pipeline."""
        if not self.enabled:
            log.info("MQTT désactivé par la configuration.")
            return False
        try:
            self.client = _make_client(self.device_id)
        except ImportError:
            log.warning("paho-mqtt absent : publication désactivée.")
            self.enabled = False
            return False

        if self.username:
            self.client.username_pw_set(self.username, self.password)
        if self.use_tls:
            self.client.tls_set()

        # Testament : le broker publie ce message si le Pi disparaît brutalement.
        self.client.will_set(
            self.topic_status,
            json.dumps({"device_id": self.device_id, "status": "offline", "ts": time.time()}),
            qos=self.qos, retain=True,
        )
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect

        try:
            self.client.connect_async(self.host, self.port, keepalive=self.keepalive)
            self.client.loop_start()          # thread réseau paho
        except Exception:
            log.exception("MQTT: connexion impossible à %s:%s — le pipeline continue.",
                          self.host, self.port)
            return False

        return self._connected.wait(timeout=timeout)

    # ------------------------------------------------------------------
    def _on_connect(self, client, userdata, flags, rc, *args) -> None:
        if rc == 0:
            self._connected.set()
            log.info("MQTT connecté à %s:%s (client_id=%s)", self.host, self.port, self.device_id)
            client.publish(
                self.topic_status,
                json.dumps({"device_id": self.device_id, "status": "online", "ts": time.time()}),
                qos=self.qos, retain=True,
            )
            self._flush_pending()
        else:
            self._connected.clear()
            log.warning("MQTT connexion refusée (code=%s)", rc)

    def _on_disconnect(self, client, userdata, rc, *args) -> None:
        self._connected.clear()
        if rc != 0:
            log.warning("MQTT déconnecté (code=%s) — paho retentera seul.", rc)

    # ------------------------------------------------------------------
    def _encode_jpeg(self, img_bgr: np.ndarray, max_side: int, quality: int) -> str:
        h, w = img_bgr.shape[:2]
        scale = min(1.0, max_side / float(max(h, w)))
        if scale < 1.0:
            img_bgr = cv2.resize(img_bgr, (max(1, int(w * scale)), max(1, int(h * scale))),
                                 interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            return ""
        return base64.b64encode(buf.tobytes()).decode("ascii")

    def _publish(self, topic: str, payload: str, retain: bool = False) -> None:
        if not self.enabled:
            return
        if self.client is None or not self.connected:
            with self._lock:
                if len(self._pending) == self._pending.maxlen:
                    self.dropped += 1          # deque bornée : le plus ancien saute
                self._pending.append((topic, payload, retain))
            return
        try:
            info = self.client.publish(topic, payload, qos=self.qos, retain=retain)
            if getattr(info, "rc", 0) != 0:
                with self._lock:
                    self._pending.append((topic, payload, retain))
            else:
                self.sent += 1
        except Exception:
            log.exception("MQTT: échec de publication (non bloquant)")

    def _flush_pending(self) -> None:
        with self._lock:
            waiting = list(self._pending)
            self._pending.clear()
        if not waiting:
            return
        log.info("MQTT : rejeu de %d message(s) en attente", len(waiting))
        for topic, payload, retain in waiting:
            try:
                self.client.publish(topic, payload, qos=self.qos, retain=retain)
                self.sent += 1
            except Exception:
                log.exception("MQTT: échec du rejeu")
                break

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------
    def publish_event(self, payload: dict, crop_bgr: Optional[np.ndarray] = None) -> None:
        """Signature compatible avec ResultStorage(mqtt_hook=...)."""
        try:
            full = dict(payload)
            full.setdefault("device_id", self.device_id)
            full.setdefault("ts", time.time())
            if crop_bgr is not None and getattr(crop_bgr, "size", 0) > 0:
                full["crop_b64"] = self._encode_jpeg(crop_bgr, self.image_max_side,
                                                     self.image_jpeg_quality)
            self._publish(self.topic_events, json.dumps(full, ensure_ascii=False))
        except Exception:
            log.exception("MQTT: publish_event a échoué (non bloquant)")

    def publish_stats(self, stats: Dict[str, Any]) -> None:
        payload = dict(stats)
        payload["device_id"] = self.device_id
        payload["ts"] = time.time()
        payload["mqtt_pending"] = len(self._pending)
        payload["mqtt_dropped"] = self.dropped
        self._publish(self.topic_stats, json.dumps(payload, ensure_ascii=False))

    def publish_preview(self, frame_bgr: np.ndarray, meta: Optional[dict] = None) -> None:
        """Aperçu annoté pour le dashboard. Jamais mis en file : une image
        périmée n'a aucune valeur, on la laisse tomber si on est hors ligne."""
        if not self.enabled or not self.connected or frame_bgr is None:
            return
        try:
            payload = {"device_id": self.device_id, "ts": time.time(),
                       "jpeg_b64": self._encode_jpeg(frame_bgr, self.preview_max_side,
                                                     self.preview_jpeg_quality)}
            if meta:
                payload.update(meta)
            self.client.publish(self.topic_preview, json.dumps(payload), qos=0, retain=False)
        except Exception:
            log.exception("MQTT: publish_preview a échoué (non bloquant)")

    def close(self, timeout: float = 2.0) -> None:
        if self.client is None:
            return
        try:
            if self.connected:
                info = self.client.publish(
                    self.topic_status,
                    json.dumps({"device_id": self.device_id, "status": "offline",
                                "ts": time.time()}),
                    qos=self.qos, retain=True,
                )
                try:
                    info.wait_for_publish(timeout)
                except (TypeError, ValueError, AttributeError):
                    time.sleep(0.1)
            self.client.disconnect()
        except Exception as e:
            log.warning("MQTT: erreur à la fermeture (%s)", e)
        finally:
            try:
                self.client.loop_stop()
            except Exception:
                pass
            self._connected.clear()
            log.info("MQTT fermé (envoyés=%d, perdus=%d)", self.sent, self.dropped)


def make_mqtt_hook(cfg: dict) -> Tuple[Any, MQTTPublisher]:
    """
    Retourne (hook, publisher). `hook(payload, crop_bgr=None)` est à passer à
    ResultStorage(mqtt_hook=hook). Appeler publisher.close() en fin de programme.
    """
    publisher = MQTTPublisher(cfg)
    publisher.connect()
    return publisher.publish_event, publisher