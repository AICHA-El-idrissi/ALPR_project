#!/usr/bin/env python3
"""
Point d'entree CLI. Aucune logique metier ici.

CHOISIR CE QU'ON TESTE
----------------------
    --edge_type   la CIBLE            (rpi | jetson)
    --detector    la section de models: a monter en detection
    --ocr         la section de models: a monter en OCR
    --backend     force le moteur, sinon il se deduit du chemin du modele

Usage:
    # pipeline de reference : YOLO + YOLO-OCR, ncnn sur Pi
    python main.py --edge_type rpi --source video --input clip.mp4

    # meme detecteur, OCR PaddleOCR
    python main.py --edge_type rpi --source video --input clip.mp4 \
        --ocr ocr_paddle

    # D-FINE en ONNX + PaddleOCR en ncnn, dans le meme pipeline
    python main.py --edge_type rpi --source image --input photo.jpg \
        --detector detector_dfine --ocr ocr_paddle

    # comparer deux moteurs sur les memes modeles
    python main.py --edge_type jetson --source video --input clip.mp4 \
        --precision fp16 --backend tensorrt

    # lister les combinaisons famille x moteur possibles
    python main.py --list-backends
"""
import argparse
import sys

from src.pipeline.pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ALPR edge — detection + OCR de plaques",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument("--list-backends", action="store_true",
                        help="affiche la matrice famille x moteur et quitte")
    parser.add_argument("--source", choices=["image", "video", "csi"])
    parser.add_argument("--input", default=None,
                        help="chemin du fichier ou index webcam "
                             "(requis pour image/video)")
    parser.add_argument("--edge_type", choices=["jetson", "rpi"])
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--config_rpi", default="configs/rpi.yaml")
    parser.add_argument("--config_jetson", default="configs/jetson.yaml")
    parser.add_argument("--precision", default="fp32",
                        choices=["fp32", "fp16", "int8"])

    # --- quels modeles monter -------------------------------------------
    parser.add_argument("--detector", default="detector", metavar="SECTION",
                        help="section de models: pour la detection "
                             "(defaut: detector)")
    parser.add_argument("--ocr", default="ocr", metavar="SECTION",
                        help="section de models: pour l'OCR (defaut: ocr)")
    parser.add_argument("--backend", default=None,
                        choices=["ncnn", "tensorrt", "onnx"],
                        help="force le moteur ; sinon deduit du chemin du "
                             "modele declare dans la config")

    parser.add_argument("--display", action="store_true",
                        help="fenetre OpenCV locale (necessite un affichage)")
    parser.add_argument("--store_frame", action="store_true",
                        help="ecrire les frames annotees sur disque (lourd)")
    parser.add_argument("--store_crop", action="store_true",
                        help="ecrire les crops de plaque sur disque")
    args = parser.parse_args()

    if args.list_backends:
        from src.wrappers.registry import afficher_matrice
        print(afficher_matrice())
        sys.exit(0)

    # Ces deux options ne sont obligatoires que hors --list-backends, d'ou le
    # controle ici plutot que `required=True` (qui casserait --list-backends).
    if not args.source or not args.edge_type:
        parser.error("--source et --edge_type sont requis "
                     "(sauf avec --list-backends)")
    if args.source in ("image", "video") and not args.input:
        parser.error(f"--input est requis pour --source {args.source}")

    run(
        source_type=args.source,
        input_path=args.input,
        config_path=args.config,
        precision=args.precision,
        edge_type=args.edge_type,
        config_jetson=args.config_jetson,
        config_rpi=args.config_rpi,
        detector_section=args.detector,
        ocr_section=args.ocr,
        backend=args.backend,
        display=args.display,
        store_frame=args.store_frame,
        store_crop=args.store_crop,
    )


if __name__ == "__main__":
    main()