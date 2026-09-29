#!/usr/bin/env python3
"""
Wrapper d'inférence TensorRT, pour TROIS familles de modèles.

    famille        entrées                sorties                 décodage
    -------------  ---------------------  ----------------------  -----------------
    yolo           images (1,3,S,S)       1 tête brute            decode_detections
    dfine          images + orig_sizes    labels/boxes/scores     filtrage du score
    paddle_rec     x (1,3,48,W)           logits CTC (1,T,C)      CTC glouton



IMPORTANT ( :
    - L'engine doit être construit SUR le Jetson cible : il dépend de
      l'archi GPU et des versions TensorRT/CUDA. Voir scripts/export_jetson.py.
    - Un engine YOLO doit être exporté SANS NMS intégrée.
    - Batch 1.

Dépendances sur Jetson : tensorrt, pycuda.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from src.models.postprocess_yolo import Detection, decode_detections
from src.models import dfine as dfine_family
from src.models.postprocess_paddle_rec import ctc_greedy_decode
from src.common.preprocess import (FAMILIES, REC_HEIGHT_DEFAUT,
                                      REC_WIDTH_DEFAUT, build_feeds,
                                      expected_input_shapes, yolo_geometry)
from src.common.base import InferenceBackend


# Le prétraitement, les shapes attendues et la géométrie du letterbox vivent
# désormais dans src/models/preprocessing.py, partagés avec les backends ncnn
# et ONNX. Les y laisser ici garantissait qu'ils divergeraient : un même
# modèle porté sur deux moteurs aurait fini par lire différemment.


# ---------------------------------------------------------------------------
class TensorRTModel(InferenceBackend):
    backend_name = "tensorrt"

    def __init__(self, model_dir: Union[str, Path], imgsz: int,
                 num_classes: int, family: str = "yolo",
                 charset: Optional[Sequence[str]] = None,
                 rec_height: int = REC_HEIGHT_DEFAUT,
                 rec_width: int = REC_WIDTH_DEFAUT,
                 use_fp16: bool = True):
        self.family = family
        self.imgsz = imgsz
        self.num_classes = num_classes
        self.charset = list(charset) if charset else []
        self.rec_height = rec_height
        self.rec_width = rec_width
        self.use_fp16 = use_fp16
        # Contrôles communs aux trois backends : famille connue, combinaison
        # supportée, charset présent pour une famille de reconnaissance.
        self._valider()

        try:
            import tensorrt as trt
        except ImportError as exc:
            raise ImportError(
                "Le module Python 'tensorrt' n'est pas installé. Ce wrapper "
                "doit être exécuté sur le Jetson avec TensorRT fourni par "
                "JetPack.") from exc
        try:
            import pycuda.autoinit  # noqa: F401
            import pycuda.driver as cuda
        except ImportError as exc:
            raise ImportError(
                "PyCUDA n'est pas installé. Installe/configure PyCUDA dans "
                "l'environnement Python utilisé par le pipeline TensorRT."
            ) from exc

        self._trt = trt
        self._cuda = cuda

        self.engine_path = self._find_engine(model_dir)
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        with open(self.engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(
                f"TensorRT n'a pas pu charger l'engine : {self.engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Impossible de créer le TensorRT execution context.")

        # --- Recensement des tenseurs : TOUTES les entrées, TOUTES les sorties
        self.input_names: List[str] = []
        self.output_names: List[str] = []
        for i in range(self.engine.num_io_tensors):
            nom = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(nom) == trt.TensorIOMode.INPUT:
                self.input_names.append(nom)
            else:
                self.output_names.append(nom)

        if not self.input_names:
            raise RuntimeError("Aucun tenseur d'entrée dans l'engine TensorRT.")
        if not self.output_names:
            raise RuntimeError("Aucun tenseur de sortie dans l'engine TensorRT.")

        attendues = expected_input_shapes(family, imgsz, rec_height, rec_width)
        self._verifier_familie(attendues)

        # --- Résolution des shapes dynamiques, entrée par entrée
        self.input_specs: Dict[str, Dict[str, Any]] = {}
        for nom in self.input_names:
            shape = tuple(int(x) for x in self.engine.get_tensor_shape(nom))
            if -1 in shape:
                if nom not in attendues:
                    raise RuntimeError(
                        f"L'entrée {nom!r} a une shape dynamique {shape} et la "
                        f"famille {family!r} ne sait pas quoi lui donner. "
                        f"Reconstruis l'engine avec un profil figé "
                        f"(scripts/export_jetson.py --shape {nom}:...).")
                self.context.set_input_shape(nom, attendues[nom])
                shape = tuple(int(x) for x in self.context.get_tensor_shape(nom))

            dtype = trt.nptype(self.engine.get_tensor_dtype(nom))
            host = np.empty(int(np.prod(shape)), dtype=dtype)
            device = cuda.mem_alloc(host.nbytes)
            self.context.set_tensor_address(nom, int(device))
            self.input_specs[nom] = {"shape": shape, "dtype": dtype,
                                     "device": device}

            if nom in attendues and shape != attendues[nom]:
                raise RuntimeError(
                    f"Shape inattendue pour l'entrée {nom!r} : {shape}, "
                    f"attendu {attendues[nom]} pour la famille {family!r}.")

        # --- Sorties : allouées après résolution des entrées
        self.output_buffers: Dict[str, Dict[str, Any]] = {}
        for nom in self.output_names:
            shape = tuple(int(x) for x in self.context.get_tensor_shape(nom))
            if -1 in shape:
                raise RuntimeError(
                    f"Shape de sortie dynamique non résolue pour {nom!r} : "
                    f"{shape}")
            dtype = trt.nptype(self.engine.get_tensor_dtype(nom))
            host = np.empty(int(np.prod(shape)), dtype=dtype)
            device = cuda.mem_alloc(host.nbytes)
            self.context.set_tensor_address(nom, int(device))
            self.output_buffers[nom] = {"host": host, "device": device,
                                        "shape": shape, "dtype": dtype}

        self.stream = cuda.Stream()

    # ------------------------------------------------------------------
    def _verifier_familie(self, attendues: Dict[str, Tuple[int, ...]]) -> None:
        """Détecte tôt un engine qui ne correspond pas à la famille annoncée.

        Sans ce contrôle, un engine D-FINE chargé avec family="yolo" irait
        jusqu'au décodage et produirait des détections absurdes plutôt
        qu'une erreur.
        """
        if self.family == "dfine":
            manquants = [n for n in (dfine_family.INPUT_IMAGES,
                                     dfine_family.INPUT_SIZES)
                         if n not in self.input_names]
            if manquants:
                raise RuntimeError(
                    f"family='dfine' attend les entrées "
                    f"{dfine_family.INPUT_IMAGES} et {dfine_family.INPUT_SIZES}, "
                    f"absentes : {manquants}. Entrées de l'engine : "
                    f"{self.input_names}")
        elif len(self.input_names) != 1:
            raise RuntimeError(
                f"family={self.family!r} attend une seule entrée, l'engine en "
                f"a {len(self.input_names)} : {self.input_names}. "
                f"S'agit-il d'un engine D-FINE ?")

    @staticmethod
    def _find_engine(model_dir: Union[str, Path]) -> Path:
        path = Path(model_dir)
        if path.is_file():
            if path.suffix != ".engine":
                raise ValueError(
                    f"Le fichier fourni n'est pas un TensorRT engine : {path}")
            return path
        if not path.exists():
            raise FileNotFoundError(f"Dossier TensorRT introuvable : {path}")
        engines = sorted(path.glob("*.engine"))
        if not engines:
            raise FileNotFoundError(f"Aucun fichier .engine trouvé dans {path}")
        modele = path / "model.engine"
        return modele if modele.exists() else engines[0]

    # ------------------------------------------------------------------
    def infer_raw(self, feeds: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Exécute l'engine sur un dictionnaire d'entrées -> dict de sorties.

        Générique : ne suppose ni le nombre d'entrées, ni celui de sorties.
        """
        manquantes = [n for n in self.input_names if n not in feeds]
        if manquantes:
            raise ValueError(f"Entrées non fournies : {manquantes}")

        for nom, tableau in feeds.items():
            spec = self.input_specs[nom]
            tableau = np.ascontiguousarray(tableau, dtype=spec["dtype"])
            if tableau.shape != spec["shape"]:
                raise ValueError(
                    f"Entrée {nom!r} : shape {tableau.shape}, "
                    f"attendu {spec['shape']}")
            self._cuda.memcpy_htod_async(spec["device"], tableau, self.stream)

        if not self.context.execute_async_v3(stream_handle=self.stream.handle):
            raise RuntimeError("TensorRT execute_async_v3() a échoué.")

        for buf in self.output_buffers.values():
            self._cuda.memcpy_dtoh_async(buf["host"], buf["device"], self.stream)
        self.stream.synchronize()

        return {nom: buf["host"].reshape(buf["shape"]).copy()
                for nom, buf in self.output_buffers.items()}

    # ------------------------------------------------------------------
    def detect(self, img_bgr: np.ndarray, conf_thres: float = 0.35,
               iou_thres: float = 0.45,
               class_agnostic_nms: bool = False) -> List[Detection]:
        """Détection -- familles 'yolo' et 'dfine' uniquement."""
        if not self.est_detecteur:
            self._refuser_detect()

        feeds = build_feeds(self.family, img_bgr, self.imgsz, self.input_names,
                            self.rec_height, self.rec_width)
        sorties = self.infer_raw(feeds)

        if self.family == "dfine":
            labels, boxes, scores = dfine_family.outputs_in_order(sorties)
            # Ni NMS ni decode_detections : DETR rend un ensemble déjà
            # apparié, dans le repère de l'image d'origine.
            return dfine_family.decode_dfine(labels, boxes, scores, conf_thres)

        ratio, pad = yolo_geometry(img_bgr, self.imgsz)
        brute = sorties[self.output_names[0]]
        return decode_detections(brute, self.num_classes, ratio, pad,
                                 img_bgr.shape[:2], conf_thres, iou_thres,
                                 class_agnostic_nms)

    def read(self, img_bgr: np.ndarray) -> Tuple[str, float, List[str]]:
        """Lecture de texte -- famille 'paddle_rec' uniquement."""
        if self.est_detecteur:
            self._refuser_read()
        feeds = build_feeds(self.family, img_bgr, self.imgsz, self.input_names,
                            self.rec_height, self.rec_width)
        sorties = self.infer_raw(feeds)
        return ctc_greedy_decode(sorties[self.output_names[0]], self.charset)

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Libère les buffers CUDA. Le contexte part au ramasse-miettes."""
        for spec in list(self.input_specs.values()) + \
                list(self.output_buffers.values()):
            try:
                spec["device"].free()
            except Exception:
                pass