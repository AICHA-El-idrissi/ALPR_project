def run_images(pipe, chemins, mqtt: MQTTPublisher, save_dir, display):
    print(f"\n{'image':<26} {'vehicule':<13} {'plaque':<14}")
    print("-" * 74)
    t0 = time.perf_counter()
    for i, f in enumerate(chemins, 1):
        frame = cv2.imread(str(f))
        if frame is None:
            print(f"  {f.name:<24} [illisible]")
            continue
        t1 = time.perf_counter()
        lectures = pipe.process(frame, avec_suivi=False)
        ms = (time.perf_counter() - t1) * 1000.0

        if not lectures:
            print(f"  {f.name:<24} aucune lecture ({ms:.0f} ms)")
        for l in lectures:
            print(f"  {f.name:<24} {l.vehicule:<13} {l.texte:<14} "
                  f"conf={l.conf_ocr:.3f}  ({ms:.0f} ms)")
            mqtt.event({"event": "plate_read", "source": f.name,
                        "vehicle_class": l.vehicule, "plate_text": l.texte,
                        "conf": round(l.conf_ocr, 3), "frame": i,
                        "vehicle_track_id": None}, l.crop)

        vue = annoter(frame, lectures, 1000.0 / max(ms, 1e-6))
        mqtt.preview(vue)
        if save_dir:
            cv2.imwrite(str(save_dir / f"annote_{f.name}"), vue)
        if display:
            cv2.imshow("bench ALPR", vue)
            if (cv2.waitKey(0) & 0xFF) == ord("q"):
                break
    total = time.perf_counter() - t0
    mqtt.status({"mode": "image", "images": len(chemins),
                 "det_ms": round(pipe.det.infer_ms, 1),
                 "ocr_ms": round(pipe.ocr.infer_ms, 1)})
    resume(pipe, len(chemins), total)


def run_video(pipe, chemin, mqtt: MQTTPublisher, save_dir, display, max_frames):
    cap = cv2.VideoCapture(str(chemin))
    if not cap.isOpened():
        sys.exit(f"Vidéo illisible : {chemin}")
    print(f"\nvidéo : {chemin}  (suivi + vote actifs)")
    print("-" * 74)

    n, t0, annonces, dernier_statut = 0, time.perf_counter(), {}, 0.0
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and n >= max_frames):
            break
        t1 = time.perf_counter()
        lectures = pipe.process(frame, avec_suivi=True)
        ms = (time.perf_counter() - t1) * 1000.0
        n += 1
        fps = n / max(time.perf_counter() - t0, 1e-6)

        for l in lectures:
            # Un événement par image et par plaque saturerait le broker :
            # on ne publie qu'au verrouillage.
            if l.verrouillee and annonces.get(l.track_id) != l.texte:
                annonces[l.track_id] = l.texte
                print(f"  image {n:>5} : piste {l.track_id} verrouillée -> "
                      f"{l.texte}  ({l.vehicule})")
                mqtt.event({"event": "plate_locked", "frame": n,
                            "vehicle_track_id": l.track_id,
                            "track_id": l.track_id,
                            "vehicle_class": l.vehicule,
                            "plate_text": l.texte, "conf": round(l.conf_ocr, 3),
                            "fps": round(fps, 2)}, l.crop)

        vue = annoter(frame, lectures, fps)
        mqtt.preview(vue)
        if time.time() - dernier_statut > 1.0:
            dernier_statut = time.time()
            mqtt.status({"mode": "video", "frame": n, "fps": round(fps, 2),
                         "det_ms": round(pipe.det.infer_ms, 1),
                         "ocr_ms": round(pipe.ocr.infer_ms, 1),
                         "locked": len(pipe.voter.locked)})
        if save_dir:
            cv2.imwrite(str(save_dir / f"frame_{n:05d}.jpg"), vue)
        if display:
            cv2.imshow("bench ALPR", vue)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    cap.release()
    resume(pipe, n, time.perf_counter() - t0)

    for tid in list(pipe.voter._hist):
        if tid not in pipe.voter.locked:
            c = pipe.voter.candidates(tid)
            if c:
                print(f"  piste {tid} non verrouillée — candidats : {c[:4]}")


# ===========================================================================
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detector", type=Path, required=True,
                    help=".onnx (ONNX Runtime) ou .engine (TensorRT)")
    ap.add_argument("--recognizer", type=Path, required=True)
    ap.add_argument("--det-family", choices=["yolo", "dfine"], default="yolo")
    ap.add_argument("--ocr-family", choices=["yolo_chars", "paddle_rec"],
                    default="yolo_chars")
    ap.add_argument("--config", type=Path, default=None)

    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", type=Path)
    src.add_argument("--dir", type=Path)
    src.add_argument("--video", type=Path)
    src.add_argument("--camera", type=int,
                     help="indice de la caméra (suivi + vote, comme --video)")

    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--ocr-imgsz", type=int, default=608,
                    help="taille d'entrée de l'OCR yolo_chars")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--ocr-conf", type=float, default=0.25,
                    help="seuil du détecteur de caractères (yolo_chars) ou "
                         "seuil de rejet sur la confiance CTC (paddle_rec)")
    ap.add_argument("--min-plate-width", type=int, default=24)
    ap.add_argument("--min-votes", type=int, default=3)
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--sizes-hw", action="store_true")
    ap.add_argument("--labels-offset", type=int, default=0)

    ap.add_argument("--mqtt-broker", default=None)
    ap.add_argument("--mqtt-port", type=int, default=1883)
    ap.add_argument("--mqtt-base", default="alpr-edge",
                    help="doit correspondre à BASE_TOPIC du dashboard")
    ap.add_argument("--mqtt-mode", choices=("json", "binary"), default="json",
                    help="json : images base64 dans le JSON (dashboard de "
                         "référence) ; binary : JPEG brut sur crops/<id>")
    ap.add_argument("--device-id", default="bench")
    ap.add_argument("--preview-hz", type=float, default=2.0)

    ap.add_argument("--save-dir", type=Path, default=None)
    ap.add_argument("--display", action="store_true")
    args = ap.parse_args()

    class_names, charset, plate_cls = dict(CLASS_NAMES), list(CHARSET), PLATE_CLS_ID
    if args.config and args.config.exists():
        import yaml
        cfg = yaml.safe_load(args.config.read_text())
        d = cfg.get("models", {}).get("detector", {})
        o = cfg.get("models", {}).get("ocr", {})
        class_names = {int(k): str(v) for k, v in d.get("class_names", {}).items()} or class_names
        plate_cls = int(d.get("plate_cls_id", plate_cls))
        charset = [str(c) for c in o.get("charset", charset)]
        print(f"config lue : {args.config}")

    print(f"détecteur     : {args.detector}  (famille {args.det_family})")
    e_det = make_engine(args.detector)
    print(f"  moteur {e_det.kind} | entrées {list(e_det.inputs)} "
          f"-> sorties {e_det.outputs}")
    print(f"reconnaissance: {args.recognizer}  (famille {args.ocr_family})")
    e_ocr = make_engine(args.recognizer)
    print(f"  moteur {e_ocr.kind} | entrées {list(e_ocr.inputs)} "
          f"-> sorties {e_ocr.outputs}")

    det = Detector(e_det, args.det_family, args.imgsz, len(class_names),
                   args.iou, args.sizes_hw, args.labels_offset)
    ocr = Recognizer(e_ocr, args.ocr_family, charset, args.ocr_imgsz,
                     args.ocr_conf, args.iou)

    seuil_rejet = args.ocr_conf if args.ocr_family == "paddle_rec" else 0.0
    pipe = Pipeline(det, ocr, class_names, plate_cls, args.conf, seuil_rejet,
                    args.min_plate_width, args.min_votes)

    mqtt = MQTTPublisher(args.mqtt_broker, args.mqtt_port, args.mqtt_base,
                         args.device_id, args.preview_hz,
                         mode=args.mqtt_mode)
    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)

    try:
        if args.video or args.camera is not None:
            # cv2.VideoCapture accepte indifféremment un chemin ou un indice.
            source = args.video if args.video else args.camera
            run_video(pipe, source, mqtt, args.save_dir, args.display, args.max)
        else:
            chemins = ([args.image] if args.image else
                       sorted(f for f in args.dir.iterdir()
                              if f.suffix.lower() in (".jpg", ".jpeg", ".png")))
            if args.max and not args.image:
                chemins = chemins[:args.max]
            run_images(pipe, chemins, mqtt, args.save_dir, args.display)
    except KeyboardInterrupt:
        print("\ninterrompu")
    finally:
        mqtt.close()
        if args.display:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())