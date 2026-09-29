#!/usr/bin/env python3
"""
Export d'un modèle YOLO (Ultralytics) vers le format NCNN.

Ce script fait UNE seule chose : .pt -> dossier *_ncnn_model/ (FP32 par défaut,
ou FP16 avec --half). C'est ce dossier (model.ncnn.param / model.ncnn.bin) qui
sert ensuite d'entrée à ncnn_int8_calib.py pour la quantification INT8.

Usage:
    python scripts/export_ncnn.py --weights runs/detect/train/weights/best.pt \
        --imgsz 608

    # Variante FP16 (indépendante du pipeline INT8, sert surtout pour
    # accélérateurs qui supportent le calcul demi-précision) :
    python scripts/export_ncnn.py --weights best.pt --imgsz 608 --half
"""
import argparse
import shutil
from pathlib import Path
from ultralytics import YOLO
import yaml
import os

def export_ncnn(weights: str, imgsz: int, half: bool, dynamic: bool, out_dir: str | None, name: str):
    weights_path = Path(weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Poids introuvables : {weights_path}")

    model = YOLO(str(weights_path))

    export_path = model.export(
        format="ncnn",
        imgsz=imgsz,
        quantize=16 if half else None,   # remplace half= (déprécié depuis 8.4.x)
        dynamic=dynamic,
        simplify=True,
    )
    export_path = Path(export_path)  # <- déjà le dossier complet du modèle exporté

    if out_dir:
        dest = Path(out_dir) / name           # <- un sous-dossier par modèle
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(export_path), str(dest))   # <- déplace export_path, pas export_path/name
        export_path = dest

    param = export_path / "model.ncnn.param"
    bin_ = export_path / "model.ncnn.bin"
    ok = param.exists() and bin_.exists()
    ...
    return export_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    #p.add_argument("--weights", required=True, help="chemin vers le .pt entraîné")
    #p.add_argument("--imgsz", type=int, default=608,
    #                help="doit être identique à la calibration et à l'inférence")
    p.add_argument("--half", action="store_true",
                    help="export en FP16 (indépendant de l'INT8, à ne PAS combiner "
                         "avec le pipeline ncnn2table/ncnn2int8 qui part du FP32)")
    p.add_argument("--dynamic", action="store_true",
                    help="shapes dynamiques (déconseillé si vous visez l'INT8)")
    p.add_argument("--out_dir", default=None,
                    help="dossier de destination (sinon: <weights>_ncnn_model/ à côté du .pt)")
    p.add_argument("--config" , default="configs/config.yaml")
    args = p.parse_args()


    results = {}
    cfg = yaml.safe_load(open(args.config))

    for key in ["detector" , "ocr"] :

        m =cfg["models"][key]
        print(f"\n=== Export NCNN : {key} ({m['weights']}) ===")

        out = export_ncnn( m["weights"] , m["imgsz"] ,args.half, args.dynamic, args.out_dir  , key)
        
        results[key] = out

    print("==RESUME ==")

    for k,v in results.items():

        size_mb = sum(
            os.path.getsize(os.path.join(v ,f))for f in os.listdir(v) ) / 1e6 
        

        print(f"{k:10s} : {v}  ({size_mb:.1f} Mo)")