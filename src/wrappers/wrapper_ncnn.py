#!/usr/bin/env python3
"""Backend ncnn, pour les familles 'yolo' et 'paddle_rec'.

Ce qui change par rapport a la version « YOLO seulement » :

  - DEUX FAMILLES. Le portage ncnn de PaddleOCR est valide sur ce projet :
    ecart relatif de 3,9e-04 contre le graphe ONNX, et lectures correctes a
    0,94-0,997 de confiance sur 11 imagettes reelles.
  - PRETRAITEMENT IMPORTE, jamais recopie. Il vient de
    src/models/preprocess.py, partage avec les backends TensorRT et ONNX.
   
  - D-FINE EST REFUSE, avec sa raison (voir base.INCOMPATIBILITES).

Le module `ncnn` est importe paresseusement : ce fichier reste importable sur
un PC de dev sans ncnn.
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

import numpy as np

from src.models.postprocess import Detection, decode_detections
from src.models.postprocess_paddle_rec import ctc_greedy_decode
from src.common.preprocess import (REC_HEIGHT_DEFAUT, REC_WIDTH_DEFAUT,
                                      build_feeds, yolo_geometry)
from src.common.base import InferenceBackend
from src.utils.logger import get_logger

log = get_logger("wrapper_ncnn")


class NCNNModel(InferenceBackend):
    backend_name = "ncnn"

    def __init__(self, model_dir: Union[str, Path], imgsz: int,
                 num_classes: int, family: str = "yolo",
                 charset: Optional[Sequence[str]] = None,
                 rec_height: int = REC_HEIGHT_DEFAUT,
                 rec_width: int = REC_WIDTH_DEFAUT,
                 use_int8: bool = True, num_threads: int = 4,
                 use_vulkan: bool = False, fp16_arithmetic: bool = True,
                 input_blob: Optional[str] = None,
                 output_blob: Optional[str] = None):
        self.family = family
        self.imgsz = imgsz
        self.num_classes = num_classes
        self.charset = list(charset) if charset else []
        self.rec_height = rec_height
        self.rec_width = rec_width
        self._valider()

        try:
            import ncnn
        except ImportError as exc:
            raise ImportError(
                "Le module 'ncnn' n'est pas installe — requis pour "
                "l'inference reelle.\n\n    pip install ncnn\n\n"
                "Le pretraitement et le decodage restent utilisables sans lui."
            ) from exc

        param, bin_ = self._trouver_fichiers(model_dir, use_int8)

        self._ncnn = ncnn
        self._mat_cls: Any = self._resolve(ncnn, "Mat")
        net_cls: Any = self._resolve(ncnn, "Net")

        self.net = net_cls()
        opt = self.net.opt
        opt.num_threads = max(1, min(num_threads, os.cpu_count() or 4))
        opt.use_vulkan_compute = use_vulkan
        opt.use_winograd_convolution = True
        opt.use_sgemm_convolution = True
        self.using_int8 = param.name.endswith("int8.param")
        # fp16 arithmetique et INT8 ne se combinent pas : le modele INT8 a ete
        # calibre pour un calcul entier.
        actif_fp16 = bool(fp16_arithmetic) and not self.using_int8
        opt.use_fp16_packed = actif_fp16
        opt.use_fp16_storage = actif_fp16
        opt.use_fp16_arithmetic = actif_fp16
        opt.use_packing_layout = True
        try:
            opt.lightmode = True
        except AttributeError:
            pass

        if self.net.load_param(str(param)) != 0:
            raise RuntimeError(f"ncnn n'a pas pu lire le .param : {param}")
        if self.net.load_model(str(bin_)) != 0:
            raise RuntimeError(f"ncnn n'a pas pu lire le .bin : {bin_}")

        self.input_name = input_blob or self._premier_blob("input")
        self.output_name = output_blob or self._premier_blob("output")

        log.info("ncnn charge : %s (famille=%s, int8=%s, threads=%d, "
                 "%s -> %s)", param.name, family, self.using_int8,
                 opt.num_threads, self.input_name, self.output_name)

    # ------------------------------------------------------------------
    @staticmethod
    def _trouver_fichiers(model_dir: Union[str, Path],
                          use_int8: bool) -> Tuple[Path, Path]:
        """Accepte un dossier ou directement le chemin d'un .param."""
        p = Path(model_dir)
        if p.is_file() and p.suffix == ".param":
            bin_ = p.with_suffix(".bin")
            if not bin_.exists():
                raise FileNotFoundError(f"Fichier .bin absent : {bin_}")
            return p, bin_

        if not p.exists():
            raise FileNotFoundError(f"Dossier ncnn introuvable : {p}")
        if use_int8 and (p / "model-int8.param").exists():
            param, bin_ = p / "model-int8.param", p / "model-int8.bin"
        else:
            param, bin_ = p / "model.ncnn.param", p / "model.ncnn.bin"
        if not param.exists() or not bin_.exists():
            disponibles = sorted(f.name for f in p.glob("*.param"))
            raise FileNotFoundError(
                f"Modele ncnn introuvable dans {p}.\n"
                f"Attendu {param.name} + {bin_.name}. "
                f"Presents : {disponibles or 'aucun .param'}")
        return param, bin_

    def _premier_blob(self, sens: str) -> str:
        """Nom du premier blob d'entree ou de sortie.

        pnnx nomme 'in0'/'out0', l'ancien onnx2ncnn garde les noms du graphe.
        On interroge le reseau plutot que de supposer.
        """
        methode = getattr(self.net, f"{sens}_names", None)
        if methode is None:
            defaut = "in0" if sens == "input" else "out0"
            log.warning("Cette build de ncnn n'expose pas %s_names() ; "
                        "repli sur %r.", sens, defaut)
            return defaut
        noms = list(methode())
        if not noms:
            raise RuntimeError(f"Aucun blob de {sens} dans le modele ncnn.")
        return noms[0]

    @staticmethod
    def _resolve(module, attr: str):
        """Certaines builds n'exposent pas Mat/Net comme attributs statiques."""
        obj = getattr(module, attr, None)
        if obj is None:
            obj = cast(Any, getattr(importlib.import_module("ncnn"), attr, None))
        if obj is None:
            raise ImportError(
                f"Le type 'ncnn.{attr}' est introuvable dans cette build.")
        return obj

    # ------------------------------------------------------------------
    def infer_raw(self, feeds: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Execute le reseau sur un tenseur NCHW deja prepare.

        DEUX PIEGES de l'API python de ncnn, tous deux silencieux :

        1. `ncnn.Mat(tableau)` ENVELOPPE le tampon numpy sans le posseder. Lui
           passer un temporaire fait lire de la memoire liberee et rend des
           valeurs aberrantes (de l'ordre de 1e28) qu'on prendrait pour une
           divergence de conversion. D'ou la reference vivante `tampon`.
        2. `np.array(mat)` rend une VUE, pas une copie. Si l'extracteur meurt
           avant la lecture, meme symptome. D'ou le `.copy()` explicite.
        """
        if len(feeds) != 1:
            raise ValueError(
                f"Le backend ncnn n'accepte qu'une entree, recu {len(feeds)} : "
                f"{sorted(feeds)}")
        tenseur = next(iter(feeds.values()))
        tampon = np.ascontiguousarray(tenseur[0], dtype=np.float32)

        ex = self.net.create_extractor()
        ex.input(self.input_name, self._mat_cls(tampon))
        code, sortie = ex.extract(self.output_name)
        if code != 0:
            raise RuntimeError(
                f"ncnn extract({self.output_name!r}) a echoue (code={code}).\n"
                f"Verifie les noms de blobs en tete et en fin du .param.")
        resultat = np.array(sortie).copy()
        del ex, tampon
        return {self.output_name: resultat}

    # ------------------------------------------------------------------
    def detect(self, img_bgr: np.ndarray, conf_thres: float = 0.35,
               iou_thres: float = 0.45,
               class_agnostic_nms: bool = False) -> List[Detection]:
        if not self.est_detecteur:
            self._refuser_detect()
        feeds = build_feeds(self.family, img_bgr, self.imgsz,
                            [self.input_name], self.rec_height, self.rec_width)
        sorties = self.infer_raw(feeds)
        ratio, pad = yolo_geometry(img_bgr, self.imgsz)
        return decode_detections(sorties[self.output_name], self.num_classes,
                                 ratio, pad, img_bgr.shape[:2],
                                 conf_thres, iou_thres, class_agnostic_nms)

    def read(self, img_bgr: np.ndarray) -> Tuple[str, float, List[str]]:
        if self.est_detecteur:
            self._refuser_read()
        feeds = build_feeds(self.family, img_bgr, self.imgsz,
                            [self.input_name], self.rec_height, self.rec_width)
        sorties = self.infer_raw(feeds)
        return ctc_greedy_decode(sorties[self.output_name], self.charset)
