#!/usr/bin/env python3
"""Backend ONNX Runtime, pour les trois familles.

C'est la voie retenue pour D-FINE-N, dont le portage ncnn n'est pas validé
(voir src/wrappers/base.py, matrice INCOMPATIBILITES). C'est aussi le backend
de référence sur PC de dev : il tourne partout, il sert de vérité quand on
valide un portage ncnn ou TensorRT.

Il n'y a aucune logique de prétraitement ici. Tout vient de
src/models/preprocessing.py, partagé avec les deux autres backends -- c'est
la seule façon de garantir qu'un même modèle lit pareil quel que soit le
moteur.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from src.models.postprocess_yolo import Detection, decode_detections
from src.models import postprocess_dfine as dfine_family
from src.models.postprocess_paddle_rec import ctc_greedy_decode
from src.common.preprocess import (REC_HEIGHT_DEFAUT, REC_WIDTH_DEFAUT,
                                      build_feeds, expected_input_shapes,
                                      yolo_geometry)
from src.common.base import InferenceBackend
from src.utils.logger import get_logger

log = get_logger("wrapper_onnx")


class ONNXModel(InferenceBackend):
    backend_name = "onnx"

    def __init__(self, model_path: Union[str, Path], imgsz: int,
                 num_classes: int, family: str = "yolo",
                 charset: Optional[Sequence[str]] = None,
                 rec_height: int = REC_HEIGHT_DEFAUT,
                 rec_width: int = REC_WIDTH_DEFAUT,
                 num_threads: int = 4,
                 providers: Optional[Sequence[str]] = None):
        self.family = family
        self.imgsz = imgsz
        self.num_classes = num_classes
        self.charset = list(charset) if charset else []
        self.rec_height = rec_height
        self.rec_width = rec_width
        self._valider()

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "Le module 'onnxruntime' n'est pas installé.\n\n"
                "    pip install onnxruntime\n") from exc

        self.model_path = self._trouver_modele(model_path)

        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, num_threads)
        options.graph_optimization_level = \
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(self.model_path), sess_options=options,
            providers=list(providers) if providers else ["CPUExecutionProvider"])

        self.input_names = [e.name for e in self.session.get_inputs()]
        self.output_names = [s.name for s in self.session.get_outputs()]
        self.input_shapes = {e.name: tuple(e.shape)
                             for e in self.session.get_inputs()}

        self._verifier_graphe()
        log.info("ONNX chargé : %s (famille=%s, entrées=%s -> sorties=%s)",
                 self.model_path.name, family, self.input_names,
                 self.output_names)

    # ------------------------------------------------------------------
    @staticmethod
    def _trouver_modele(model_path: Union[str, Path]) -> Path:
        p = Path(model_path)
        if p.is_file():
            if p.suffix != ".onnx":
                raise ValueError(f"Le fichier fourni n'est pas un .onnx : {p}")
            return p
        if not p.exists():
            raise FileNotFoundError(f"Modèle ONNX introuvable : {p}")
        candidats = sorted(p.glob("*.onnx"))
        if not candidats:
            raise FileNotFoundError(f"Aucun .onnx trouvé dans {p}")
        prefere = p / "model.onnx"
        return prefere if prefere.exists() else candidats[0]

    def _verifier_graphe(self) -> None:
        """Détecte tôt un modèle qui ne correspond pas à la famille annoncée.

        Sans ce contrôle, un graphe D-FINE chargé avec family='yolo' irait
        jusqu'au décodage et produirait des détections absurdes plutôt qu'une
        erreur lisible.
        """
        if self.family == "dfine":
            manquants = [n for n in (dfine_family.INPUT_IMAGES,
                                     dfine_family.INPUT_SIZES)
                         if n not in self.input_names]
            if manquants:
                raise RuntimeError(
                    f"family='dfine' attend les entrées "
                    f"{dfine_family.INPUT_IMAGES} et "
                    f"{dfine_family.INPUT_SIZES}, absentes : {manquants}.\n"
                    f"Entrées du graphe : {self.input_names}")
            if len(self.output_names) < 3:
                raise RuntimeError(
                    f"family='dfine' attend trois sorties "
                    f"(labels/boxes/scores), le graphe en a "
                    f"{len(self.output_names)} : {self.output_names}")
        elif len(self.input_names) != 1:
            raise RuntimeError(
                f"family={self.family!r} attend une seule entrée, le graphe "
                f"en a {len(self.input_names)} : {self.input_names}.\n"
                f"S'agit-il d'un modèle D-FINE ?")

        if self.family == "paddle_rec":
            self._verifier_charset_contre_graphe()
            self._lire_hauteur_du_graphe()

    def _verifier_charset_contre_graphe(self) -> None:
        """Le nombre de classes du graphe doit valoir len(charset) + 1.

        Un écart de 1 est la panne la plus insidieuse de toute la chaîne :
        elle décale chaque caractère et produit un texte plausible et faux.
        """
        forme = tuple(self.session.get_outputs()[0].shape)
        n_classes = forme[-1] if isinstance(forme[-1], int) else None
        if n_classes is None:
            log.warning("Nombre de classes non statique dans le graphe : "
                        "la concordance avec le charset n'est pas vérifiable.")
            return
        if len(self.charset) + 1 != n_classes:
            raise ValueError(
                f"Le charset compte {len(self.charset)} caractères, soit "
                f"{len(self.charset) + 1} classes avec le blanc CTC, mais le "
                f"modèle en sort {n_classes}.\n"
                f"Écart de {n_classes - len(self.charset) - 1}. Causes "
                f"habituelles : une ligne vide du dictionnaire filtrée à tort, "
                f"ou `use_space_char` actif à l'entraînement.")

    def _lire_hauteur_du_graphe(self) -> None:
        """La hauteur est imposée par l'architecture ; on la lit plutôt que de
        la supposer. La largeur, elle, reste un choix."""
        forme = self.input_shapes[self.input_names[0]]
        if len(forme) == 4 and isinstance(forme[2], int) and forme[2] > 0:
            if forme[2] != self.rec_height:
                log.warning("rec_height=%d demandé, mais le graphe impose %d "
                            "— on suit le graphe.", self.rec_height, forme[2])
                self.rec_height = int(forme[2])

    # ------------------------------------------------------------------
    def infer_raw(self, feeds: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        manquantes = [n for n in self.input_names if n not in feeds]
        if manquantes:
            raise ValueError(f"Entrées non fournies : {manquantes}")
        brut = self.session.run(None, {n: feeds[n] for n in self.input_names})
        return dict(zip(self.output_names, brut))

    def detect(self, img_bgr: np.ndarray, conf_thres: float = 0.35,
               iou_thres: float = 0.45,
               class_agnostic_nms: bool = False) -> List[Detection]:
        if not self.est_detecteur:
            self._refuser_detect()

        feeds = build_feeds(self.family, img_bgr, self.imgsz, self.input_names,
                            self.rec_height, self.rec_width)
        sorties = self.infer_raw(feeds)

        if self.family == "dfine":
            labels, boxes, scores = dfine_family.outputs_in_order(sorties)
            # Ni NMS ni remise à l'échelle : DETR rend un ensemble déjà
            # apparié, dans le repère de l'image d'origine.
            return dfine_family.decode_dfine(labels, boxes, scores, conf_thres)

        ratio, pad = yolo_geometry(img_bgr, self.imgsz)
        return decode_detections(sorties[self.output_names[0]],
                                 self.num_classes, ratio, pad,
                                 img_bgr.shape[:2], conf_thres, iou_thres,
                                 class_agnostic_nms)

    def read(self, img_bgr: np.ndarray) -> Tuple[str, float, List[str]]:
        if self.est_detecteur:
            self._refuser_read()
        feeds = build_feeds(self.family, img_bgr, self.imgsz, self.input_names,
                            self.rec_height, self.rec_width)
        sorties = self.infer_raw(feeds)
        return ctc_greedy_decode(sorties[self.output_names[0]], self.charset)