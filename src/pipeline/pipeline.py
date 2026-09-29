#!/usr/bin/env python3
"""
Point d'entree du pipeline complet (detection vehicules+plaque, OCR, tracking,
association, vote temporel, publication MQTT, mesure du FPS).
Compteur FPS (capture + traitement), publication des stats a 1 Hz, apercu
JPEG annote pour le dashboard, overlay FPS a l'ecran.

Seul `build_session` connait la configuration. Les boucles `_run_image` et
`_run_video` ne voient que du texte dans PlateResult : elles sont identiques
quels que soient les modeles et les moteurs montes.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import yaml

from src.common.capture import build_source
from src.common.inference_session import InferenceSession
from src.common.mqtt_publisher import MQTTPublisher
from src.utils.logger import get_logger
from src.utils.metrics import FPSMeter, StageTimer, Throttle
from src.utils.storage import ResultStorage

log = get_logger("pipeline")

# Couleurs BGR par role
COLOR_VEHICLE = (86, 180, 233)
COLOR_PLATE = (60, 220, 120)
COLOR_LOCKED = (40, 200, 255)


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration introuvable : {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _lire_metadata(dossier: Path) -> dict:
    """Lit le metadata.yaml qu'Ultralytics depose a cote d'un export.

    Il porte la taille d'entree REELLE du modele exporte. C'est une verite
    terrain : la taille est inscrite dans le .param sous forme de nombre
    d'ancres, et faire tourner un modele exporte en 640 avec imgsz=416 ne
    redimensionne rien -- le Reshape final reclame 8400 valeurs dans un
    tampon qui en contient 3549, d'ou une lecture hors limites et un segfault.
    """
    fichier = dossier / "metadata.yaml" if dossier.is_dir() else \
        dossier.parent / "metadata.yaml"
    if not fichier.exists():
        return {}
    try:
        with open(fichier, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:                      # YAML casse : on continue
        log.warning("metadata.yaml illisible (%s) : %s", fichier, exc)
        return {}


def _imgsz_du_metadata(meta: dict) -> Optional[int]:
    """Ultralytics ecrit `imgsz: [h, w]` ; certaines versions un entier."""
    valeur = meta.get("imgsz")
    if isinstance(valeur, (list, tuple)) and valeur:
        return int(valeur[0])
    if isinstance(valeur, (int, float)):
        return int(valeur)
    return None


PRECISIONS_CONNUES = ("fp32", "fp16", "int8")


def _verifier_existence(chemin: str, gabarits: dict, cible: str,
                        precision: str, role: str) -> str:
    """Refuse tot un chemin absent, en disant ce qui EXISTE reellement.

    La configuration declare des gabarits ; le disque, lui, ne contient pas
    forcement toutes les precisions pour tous les moteurs. Plutot que de
    maintenir une table des precisions disponibles -- qui divergerait du
    disque des le premier export -- on regarde.
    """
    if Path(chemin).exists():
        return chemin

    # Quelles precisions existent pour CE moteur ?
    gabarit = str(gabarits.get(cible) or "")
    autres_precisions = [p for p in PRECISIONS_CONNUES
                         if p != precision and "{precision}" in gabarit
                         and Path(gabarit.format(precision=p)).exists()]
    # Quels moteurs ont CETTE precision ?
    autres_moteurs = [b for b, g in gabarits.items()
                      if b != cible and g
                      and Path(str(g).format(precision=precision)).exists()]

    pistes = []
    if autres_precisions:
        pistes.append(f"  --precision {' | '.join(autres_precisions)} "
                      f"(existe en {cible})")
    if autres_moteurs:
        pistes.append(f"  --backend {' | '.join(autres_moteurs)} "
                      f"(existe en {precision})")
    if not pistes:
        pistes.append("  aucun autre export trouve sur le disque pour ce "
                      "modele : il faut le produire.")

    raise FileNotFoundError(
        f"models.{role} : rien a {chemin!r} "
        f"(moteur {cible!r}, precision {precision!r}).\n"
        f"Ce qui existe :\n" + "\n".join(pistes))


def resoudre_chemin(section_cfg: dict, role: str, precision: str,
                    edge_type: str, backend: Optional[str],
                    cfg_edge: dict) -> Tuple[str, Optional[str]]:
    """Rend (chemin du modele, backend deduit ou None).

    Trois facons de declarer un modele, de la plus explicite a la plus
    ancienne :

      1. `paths:` par backend, avec {precision} substitue. C'est la forme a
         privilegier : elle dit explicitement quels moteurs existent pour ce
         modele, et `null` dit qu'il n'y en a pas.
      2. un chemin direct (`ncnn_param`, `engine`, `onnx`, `path`).
      3. `exported_models` de la config de cible (schema historique).
    """
    # --- 1. table par backend --------------------------------------------
    paths = section_cfg.get("paths")
    if paths:
        # Un backend EXPLICITE (--backend, ou `backend:` dans la section) doit
        # etre honore ou echouer. Un backend simplement DEDUIT de la cible
        # peut ceder la place s'il n'existe qu'un seul export : le choix est
        # alors sans ambiguite.
        explicite = backend or section_cfg.get("backend")
        cible = explicite or \
            {"rpi": "ncnn", "jetson": "tensorrt"}.get(edge_type, "ncnn")
        disponibles = sorted(k for k, v in paths.items() if v)

        if cible not in paths or paths[cible] is None:
            if explicite:
                raise KeyError(
                    f"Aucun export {cible!r} pour models.{role} "
                    f"(demande explicitement).\n"
                    f"Moteurs disponibles pour ce modele : {disponibles}.\n"
                    f"Relance avec --backend "
                    f"{disponibles[0] if disponibles else '<aucun>'}, ou "
                    f"produis l'export manquant.")
            if len(disponibles) != 1:
                raise KeyError(
                    f"Pas d'export {cible!r} pour models.{role} sur la cible "
                    f"{edge_type!r}, et {len(disponibles)} moteurs possibles "
                    f"({disponibles}) : impossible de choisir.\n"
                    f"Precise --backend.")
            cible = disponibles[0]
            log.info("models.%s n'a pas d'export pour la cible %s ; un seul "
                     "moteur disponible, on prend %r.", role, edge_type, cible)

        chemin = str(paths[cible]).format(precision=precision)
        return _verifier_existence(chemin, paths, cible, precision, role), cible

    # --- 2. chemin direct -------------------------------------------------
    for cle in ("ncnn_param", "engine", "onnx", "path"):
        if section_cfg.get(cle):
            return str(section_cfg[cle]).format(precision=precision), None

    # --- 3. exported_models (historique) ----------------------------------
    exported = (cfg_edge.get("exported_models") or {})
    cle_backend = section_cfg.get("backend") or backend or \
        {"rpi": "ncnn", "jetson": "tensorrt"}.get(edge_type, "ncnn")
    par_backend = exported.get(cle_backend) or exported.get("ncnn") or {}
    if precision not in par_backend:
        raise KeyError(
            f"Precision {precision!r} absente de exported_models"
            f"[{cle_backend!r}] (disponibles : {sorted(par_backend)}).\n"
            f"Autre possibilite : declarer `paths:` dans models.{role}.")
    if role not in par_backend[precision]:
        raise KeyError(
            f"Role {role!r} absent de exported_models[{cle_backend!r}]"
            f"[{precision!r}] (presents : {sorted(par_backend[precision])}).")
    return str(par_backend[precision][role]), cle_backend


def imgsz_effectif(chemin: str, imgsz_config: int, role: str) -> int:
    """La taille d'entree vient du metadata.yaml s'il existe, sinon du YAML.

    Le metadata fait foi : il decrit l'export reel. Une divergence est
    signalee fort, parce qu'elle produit soit un segfault, soit -- pire --
    des detections silencieusement fausses.
    """
    meta = _lire_metadata(Path(chemin))
    reel = _imgsz_du_metadata(meta)
    if reel is None:
        return imgsz_config
    if reel != imgsz_config:
        log.warning(
            "models.%s declare imgsz=%d, mais le metadata.yaml de l'export "
            "indique %d. On suit le METADATA : la taille est figee dans le "
            ".param (nombre d'ancres), la changer ne redimensionne rien.",
            role, imgsz_config, reel)
    return reel


def build_session(cfg: dict, cfg_edge: dict, precision: str,
                  edge_type: str,
                  detector_section: str = "detector",
                  ocr_section: str = "ocr",
                  backend: Optional[str] = None) -> InferenceSession:
    """Construit la session depuis la configuration.

    C'est ICI, et nulle part ailleurs, que le YAML devient des familles, des
    chemins et des backends. `InferenceSession` et les boucles de traitement
    restent agnostiques : elles ne voient que `detect()` / `read()` et du
    texte.

    Deux garde-fous que la configuration seule ne donne pas :

      - un modele sans export pour le backend demande leve une erreur qui
        NOMME les moteurs disponibles, au lieu d'un FileNotFoundError opaque ;
      - la taille d'entree vient du metadata.yaml de l'export quand il existe,
        pas du YAML : c'est l'export qui fait foi.
    """
    models_cfg = cfg["models"]
    det_cfg = models_cfg[detector_section]
    ocr_cfg = models_cfg[ocr_section]

    detector_family = det_cfg.get("family", "yolo")
    ocr_family = ocr_cfg.get("family", "yolo_chars")

    detector_dir, det_backend = resoudre_chemin(
        det_cfg, detector_section, precision, edge_type, backend, cfg_edge)
    ocr_dir, ocr_backend = resoudre_chemin(
        ocr_cfg, ocr_section, precision, edge_type, backend, cfg_edge)

    detector_imgsz = imgsz_effectif(detector_dir, int(det_cfg["imgsz"]),
                                    detector_section)
    # Pour paddle_rec la taille vient de height/width : pas de metadata a lire.
    ocr_imgsz = int(ocr_cfg["imgsz"]) if ocr_family == "paddle_rec" else \
        imgsz_effectif(ocr_dir, int(ocr_cfg["imgsz"]), ocr_section)

    # --- charset de l'etage OCR -------------------------------------------
    # Les deux familles ne lisent PAS la meme liste. `charset` de la section
    # yolo vient de dataset_ocr.yaml ; `charset_file` de paddle_rec vient du
    # character_dict_path de l'entrainement PaddleOCR. Memes symboles a `gr`
    # pres, mais ordre different : les confondre laisse les chiffres justes
    # et fausse TOUTES les lettres.
    if ocr_cfg.get("charset_file"):
        from src.models.paddle_rec import load_charset
        ocr_charset = load_charset(ocr_cfg["charset_file"])
        idx_to_char = {i: c for i, c in enumerate(ocr_charset)}
    else:
        charset_dict = ocr_cfg.get("charset_dict", {})
        raw_charset = ocr_cfg["charset"]
        idx_to_char = {i: charset_dict.get(c, c)
                       for i, c in enumerate(raw_charset)}
        ocr_charset = [str(c) for c in raw_charset]

    # --- parametres de pipeline -------------------------------------------
    pipeline_cfg = dict(cfg.get("pipeline", {}))
    pipeline_cfg.update(cfg_edge.get("pipeline", {}))

    return InferenceSession(
        edge_type=edge_type,
        detector_dir=detector_dir,
        ocr_dir=ocr_dir,
        detector_imgsz=detector_imgsz,
        ocr_imgsz=ocr_imgsz,
        num_detector_classes=det_cfg["num_classes"],
        idx_to_char=idx_to_char,
        plate_cls_id=det_cfg["plate_cls_id"],
        detector_family=detector_family,
        ocr_family=ocr_family,
        ocr_charset=ocr_charset,
        rec_height=ocr_cfg.get("height", 48),
        rec_width=ocr_cfg.get("width", 320),
        backend=backend,
        frame_skip=pipeline_cfg.get("frame_skip", 3),
        vehicle_track_buffer=pipeline_cfg.get("vehicle_track_buffer", 30),
        plate_track_buffer=pipeline_cfg.get("plate_track_buffer", 10),
        min_votes_before_lock=pipeline_cfg.get("min_votes_before_lock", 5),
        vote_window=cfg.get("voting", {}).get("window", 15),
        # `conf` / `iou` dans le YAML, `conf_thres` accepte aussi
        detector_conf_thres=det_cfg.get("conf_thres", det_cfg.get("conf", 0.35)),
        detector_iou_thres=det_cfg.get("iou_thres", det_cfg.get("iou", 0.45)),
        ocr_conf_thres=ocr_cfg.get("conf_thres", ocr_cfg.get("conf", 0.30)),
        ocr_iou_thres=ocr_cfg.get("iou_thres", ocr_cfg.get("iou", 0.45)),
        min_plate_width=pipeline_cfg.get("min_plate_width", 24),
        min_sharpness=pipeline_cfg.get("min_sharpness", 0.0),
        num_threads=pipeline_cfg.get("num_threads", 4),
    )


def build_storage(cfg: dict, mqtt: Optional[MQTTPublisher],
                  force_full_frames: bool = False) -> ResultStorage:
    st_cfg = cfg.get("runtime", {}).get("storage", {})
    return ResultStorage(
        save=st_cfg.get("save", True),
        output_root=st_cfg.get("output_root", "outputs/"),
        # --store_frame en ligne de commande prime sur le YAML
        save_full_frames=st_cfg.get("save_full_frames", False) or force_full_frames,
        frame_save_every_n=st_cfg.get("frame_save_every_n", 30),
        retention_days=st_cfg.get("retention_days", 3),
        mqtt_hook=mqtt.publish_event if (mqtt and mqtt.enabled) else None,
    )


# --------------------------------------------------------------------------
# Dessin
# --------------------------------------------------------------------------
def _draw_box(frame, box, label: str, color=(0, 255, 0)) -> None:
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if not label:
        return
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, max(y1 - th - 6, 0)), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, max(y1 - 5, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def _cls_name(cls_names: Dict[int, str], cls: int) -> str:
    return cls_names.get(cls, "inconnu") if cls is not None and cls >= 0 else "inconnu"


def _draw_hud(frame, fps: float, det_ms: float, ocr_ms: float,
              frame_idx: Optional[int] = None, etiquette: str = "") -> None:
    """Bandeau FPS en haut a gauche.

    `etiquette` rappelle les modeles montes : sur une session de comparaison,
    une capture d'ecran sans cette mention ne dit pas ce qui tournait.
    """
    txt = f"{fps:5.1f} FPS | det {det_ms:4.0f} ms | ocr {ocr_ms:4.0f} ms"
    if frame_idx is not None:
        txt += f" | #{frame_idx}"
    if etiquette:
        txt += f" | {etiquette}"
    cv2.rectangle(frame, (0, 0), (10 + 9 * len(txt), 28), (20, 20, 20), -1)
    cv2.putText(frame, txt, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (240, 240, 240), 1, cv2.LINE_AA)


def _etiquette_session(session: InferenceSession) -> str:
    """« dfine/onnx + paddle_rec/ncnn » — ce qui tourne reellement."""
    return (f"{session.detector_family}/{session.detector.backend_name}"
            f" + {session.ocr_family}/{session.ocr.backend_name}")


# --------------------------------------------------------------------------
# Boucle principale
# --------------------------------------------------------------------------
def run(source_type: str,
        input_path: Optional[str],
        config_path: str,
        precision: str,
        edge_type: str,
        config_jetson: str,
        config_rpi: str,
        detector_section: str = "detector",
        ocr_section: str = "ocr",
        backend: Optional[str] = None,
        display: bool = False,
        store_frame: bool = False,
        store_crop: bool = False) -> None:

    cfg = load_config(config_path)
    cfg_edge = load_config(config_jetson if edge_type == "jetson" else config_rpi)

    mqtt = MQTTPublisher(cfg)
    mqtt.connect()

    session = build_session(cfg, cfg_edge, precision, edge_type,
                            detector_section=detector_section,
                            ocr_section=ocr_section, backend=backend)
    storage = build_storage(cfg, mqtt, force_full_frames=store_frame)
    source = build_source(source_type, input_path,
                          **(cfg.get("video", {}).get("capture", {}) or {}))

    # Les noms de classes viennent de la SECTION MONTEE, pas d'un chemin fige :
    # D-FINE et YOLO peuvent avoir des indices differents.
    det_cfg = cfg["models"][detector_section]
    cls_names = det_cfg["class_names"]
    plate_cls_id = det_cfg["plate_cls_id"]
    etiquette = _etiquette_session(session)

    fps_meter = FPSMeter(window=60)
    timer = StageTimer()
    stats_throttle = Throttle(cfg.get("runtime", {}).get("stats_hz", 1.0))
    preview_throttle = Throttle(cfg.get("runtime", {}).get("preview_hz", 2.0))

    log.info("Demarrage pipeline — source=%s input=%s edge=%s precision=%s",
             source_type, input_path, edge_type, precision)
    log.info("Modeles montes : %s", etiquette)

    try:
        if source_type == "image":
            _run_image(session, storage, source, mqtt, cls_names, plate_cls_id,
                       display, store_frame, store_crop, fps_meter, etiquette)
        else:
            _run_video(session, storage, source, mqtt, cls_names, plate_cls_id,
                       display, store_frame, store_crop,
                       fps_meter, timer, stats_throttle, preview_throttle,
                       etiquette)
    except KeyboardInterrupt:
        log.info("Interruption clavier — arret demande.")
    finally:
        source.release()
        storage.close()
        mqtt.close()
        if display:
            cv2.destroyAllWindows()
        log.info("Pipeline arrete proprement. %d frame(s), %.1f FPS moyen (%s).",
                 fps_meter.frames, fps_meter.fps_avg, etiquette)


def _run_image(session, storage, source, mqtt, cls_names, plate_cls_id,
               display: bool, store_frame: bool, store_crop: bool,
               fps_meter: FPSMeter, etiquette: str = "") -> None:
    total_plates = 0
    n_images = 0

    for frame in source.frames():
        n_images += 1
        fps_meter.tick()
        name = getattr(source, "current", None)
        name = name.name if name is not None else f"image_{n_images}"

        t0 = time.perf_counter()
        results, dets = session.process_image(frame)
        total_ms = (time.perf_counter() - t0) * 1000.0

        for d in dets:
            color = COLOR_PLATE if d.cls == plate_cls_id else COLOR_VEHICLE
            _draw_box(frame, d.box, f"{_cls_name(cls_names, d.cls)} {d.conf:.2f}", color)

        for r in results:
            if r.plate_text:
                _draw_box(frame, r.plate_box, r.plate_text, COLOR_LOCKED)

        frame_path = storage.save_frame(frame) if store_frame else None

        if not results:
            log.info("%s — aucune plaque (%d detection(s), %.0f ms)",
                     name, len(dets), total_ms)

        for r in results:
            total_plates += 1
            log.info("%s — vehicule=%s | plaque=%s | det=%.0f ms | ocr=%.0f ms",
                     name, _cls_name(cls_names, r.vehicle_cls),
                     r.plate_text or "(vide)", r.det_ms, r.ocr_ms)

            crop_path = None
            if store_crop and r.plate_crop is not None:
                crop_path = storage.save_crop(r.plate_crop, vehicle_track_id=None)

            mqtt.publish_event({
                "event": "plate_read",
                "source_image": name,
                "vehicle_class": _cls_name(cls_names, r.vehicle_cls),
                "plate_text": r.plate_text,
                "conf": round(r.conf, 3),
                "models": etiquette,
            }, r.plate_crop)

            storage.save_event(
                plate_text=r.plate_text, vehicle_cls=r.vehicle_cls,
                vehicle_track_id=None, is_locked=False,
                frame_path=frame_path, crop_path=crop_path,
                crop=None,
            )

        mqtt.publish_preview(frame, {"mode": "image", "source_image": name})

        if display:
            cv2.imshow("ALPR", frame)
            # n'importe quelle touche pour l'image suivante, 'q' pour arreter
            if (cv2.waitKey(0) & 0xFF) == ord("q"):
                break

    mqtt.publish_stats({"mode": "image", "images": n_images,
                        "plates": total_plates, **session.stats()})
    log.info("Lot termine : %d image(s), %d plaque(s) lue(s).",
             n_images, total_plates)


def _run_video(session, storage, source, mqtt, cls_names, plate_cls_id,
               display: bool, store_frame: bool, store_crop: bool,
               fps_meter: FPSMeter, timer: StageTimer,
               stats_throttle: Throttle, preview_throttle: Throttle,
               etiquette: str = "") -> None:

    for frame_idx, frame in enumerate(source.frames()):
        with timer("process"):
            results, dets = session.process_video_frame(frame, frame_idx)

        fps = fps_meter.tick()

        for d in dets:
            color = COLOR_PLATE if d.cls == plate_cls_id else COLOR_VEHICLE
            _draw_box(frame, d.box, f"{_cls_name(cls_names, d.cls)} {d.conf:.2f}", color)

        frame_path = storage.save_frame(frame, frame_idx) if store_frame else None

        for r in results:
            _draw_box(frame, r.plate_box,
                      f"{r.plate_text}{' OK' if r.is_locked else ''}",
                      COLOR_LOCKED if r.is_locked else COLOR_PLATE)

            crop_path = None
            if store_crop and r.plate_crop is not None:
                crop_path = storage.save_crop(r.plate_crop, r.vehicle_track_id,
                                              frame_idx)

            # Un evenement MQTT par frame et par plaque saturerait le broker :
            # on ne publie qu'au moment du verrouillage.
            if r.just_locked:
                mqtt.publish_event({
                    "event": "plate_locked",
                    "frame": frame_idx,
                    "vehicle_track_id": r.vehicle_track_id,
                    "vehicle_class": _cls_name(cls_names, r.vehicle_cls),
                    "plate_text": r.plate_text,
                    "conf": round(r.conf, 3),
                    "fps": round(fps, 1),
                    "models": etiquette,
                }, r.plate_crop)

            storage.save_event(
                plate_text=r.plate_text, vehicle_cls=r.vehicle_cls,
                vehicle_track_id=r.vehicle_track_id, is_locked=r.is_locked,
                frame_idx=frame_idx, frame_path=frame_path, crop_path=crop_path,
                crop=None,          # le crop part deja via publish_event
            )

        _draw_hud(frame, fps, session.timer.ms("detect"), session.timer.ms("ocr"),
                  frame_idx, etiquette)

        if stats_throttle.ready():
            mqtt.publish_stats({"mode": "video", "frame": frame_idx,
                                "process_ms": round(timer.ms("process"), 1),
                                **fps_meter.snapshot(), **session.stats()})

        if preview_throttle.ready():
            mqtt.publish_preview(frame, {"frame": frame_idx,
                                         "fps": round(fps, 1),
                                         "models": etiquette})

        if display:
            cv2.imshow("ALPR", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break