#!/usr/bin/env python3
"""
Quantification INT8 manuelle NCNN : ncnn2table + ncnn2int8, piloté par config.yaml.

Boucle sur models.detector / models.ocr : imgsz et calib_dir sont lus depuis
la config (plus besoin de les répéter en CLI par modèle), n/method depuis
quantization:. Le dossier du modèle exporté attendu est
<export.out_dir>/<key>/ (celui produit par export_ncnn1.py).

Usage :
    python scripts/ncnn_int8.py --config configs/config.yaml
    python scripts/ncnn_int8.py --config configs/config.yaml --only detector
"""
import argparse
import os
import random
import shutil
import subprocess

from click import Tuple
import cv2
import numpy as np
import yaml

# Mêmes constantes que l'entraînement Ultralytics (normalisation 0-1, pas de
# soustraction de moyenne) :
MEAN = [0.0, 0.0, 0.0]
NORM = [1 / 255.0, 1 / 255.0, 1 / 255.0]
PAD_COLOR = (114, 114, 114)  # gris Ultralytics


def letterbox(img: np.ndarray, imgsz: int, color=PAD_COLOR) -> np.ndarray:
    """Resize + padding en conservant le ratio d'aspect (comme Ultralytics)."""
    h, w = img.shape[:2]
    r = min(imgsz / h, imgsz / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((imgsz, imgsz, 3), color, dtype=np.uint8)
    top = (imgsz - new_h) // 2
    left = (imgsz - new_w) // 2  #padding
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas




def prepare_calib_images(calib_dir: str, out_dir: str, imgsz: int, n: int, use_letterbox: bool):
    """Prétraite les images de calibration comme l'inférence, et génère l'imagelist.txt."""
    if not os.path.isdir(calib_dir):
        raise FileNotFoundError(f"calib_dir introuvable : {calib_dir}")

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    files = [f for f in os.listdir(calib_dir)
              if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    if not files:
        raise FileNotFoundError(f"Aucune image (.jpg/.png/.jpeg) dans {calib_dir}")

    random.seed(42)
    random.shuffle(files)
    files = files[:n]

    written = []
    for i, f in enumerate(files):
        img = cv2.imread(os.path.join(calib_dir, f))  # reste en BGR natif
        if img is None:
            continue
        img = letterbox(img, imgsz) if use_letterbox else cv2.resize(img, (imgsz, imgsz))
        out_path = os.path.join(out_dir, f"calib_{i:04d}.png")
        cv2.imwrite(out_path, img)  # PAS de cvtColor ici : ncnn2table s'en charge via pixel=RGB
        written.append(os.path.abspath(out_path))

    imagelist_path = os.path.abspath(os.path.join(out_dir, "..", f"{os.path.basename(out_dir)}_imagelist.txt"))
    with open(imagelist_path, "w") as f:
        f.write("\n".join(written))

    print(f"[OK] {len(written)} images de calibration -> {out_dir}")
    print(f"[OK] Liste d'images -> {imagelist_path}")
    return imagelist_path


def find_tool(name: str):
    """Cherche ncnn2table/ncnn2int8 dans le PATH ou build/ncnn."""
    candidates = [
        shutil.which(name),
        f"ncnn/build/tools/quantize/{name}",
        f"ncnn/build/{name}",
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    raise FileNotFoundError(
        f"{name} introuvable. Compile ncnn avec -DNCNN_BUILD_TOOLS=ON")


def quantize(model_dir: str,out_dir:str, imagelist_path: str, imgsz: int, method: str = "kl"):
    param = os.path.join(model_dir, "model.ncnn.param")
    bin_ = os.path.join(model_dir, "model.ncnn.bin")
    table = os.path.join(model_dir, "model.table")

    if not (os.path.exists(param) and os.path.exists(bin_)):
        raise FileNotFoundError(
            f"{param} / {bin_} introuvables — as-tu bien lancé export_ncnn1.py avant ?")

    # ── 1. Table de calibration ──
    #une commande terminal complète.
    cmd_table = [
        find_tool("ncnn2table"),
        param, bin_, imagelist_path, table,
        "mean=[" + ",".join(str(m) for m in MEAN) + "]",
        "norm=[" + ",".join(str(n) for n in NORM) + "]",
        f"shape=[{imgsz},{imgsz},3]",
        "pixel=RGB",
        "thread=4",
        f"method={method}",   
    ]
    print("[1/2] ncnn2table :", " ".join(cmd_table))
    subprocess.run(cmd_table, check=True)

    # ── 2. Quantification ──
    os.makedirs(out_dir , exist_ok= True)
    out_param = os.path.join(out_dir, "model-int8.param")
    out_bin = os.path.join(out_dir, "model-int8.bin")
    cmd_int8 = [
        find_tool("ncnn2int8"),
        param, bin_, out_param, out_bin, table,
    ]
    print("[2/2] ncnn2int8 :", " ".join(cmd_int8))
    subprocess.run(cmd_int8, check=True)

    s = lambda p: os.path.getsize(p) / 1e6 if os.path.exists(p) else 0
    print(f"\nFP32 (original) : {s(bin_):.2f} Mo  ->  INT8 : {s(out_bin):.2f} Mo")
    print(f"[OK] {out_param}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--only", choices=["detector", "ocr"], default=None,
                    help="ne traiter qu'un seul modèle (sinon: les deux)")
    p.add_argument("--no-letterbox", action="store_true",
                    help="resize simple au lieu du letterbox (si l'inférence embarquée fait pareil)")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    export_root = cfg.get("export", {}).get("out_dir", "models/exported")
    quant_cfg = cfg.get("quantization", {})
    n = quant_cfg.get("n", 800)
    method = quant_cfg.get("method", "kl")

    keys = [args.only] if args.only else ["detector", "ocr"]

    for key in keys:
        if key not in cfg["models"]:
            print(f"[SKIP] pas de config pour '{key}'")
            continue

        m = cfg["models"][key]
        calib_dir = m.get("calib_dir")
        imgsz = m["imgsz"]
        if not calib_dir:
            print(f"[SKIP] pas de 'calib_dir' défini pour '{key}' dans {args.config}")
            continue

        model_dir = os.path.join(export_root,"fp32", key)
        out_dir= os.path.join(export_root , "int_8" , key )
        print(f"\n=== INT8 : {key}  (imgsz={imgsz}, calib_dir={calib_dir}) ===")

        imagelist = prepare_calib_images(
            calib_dir, f"calib_images_{key}", imgsz, n,
            use_letterbox=not args.no_letterbox,
        )
        quantize(model_dir,out_dir, imagelist, imgsz, method=method)