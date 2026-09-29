#!/usr/bin/env python3
"""Fabrique de backends : choisit le moteur et la famille depuis la config.

Le pipeline ne doit connaître ni ncnn, ni TensorRT, ni ONNX Runtime. Il
demande un modèle, il reçoit un objet qui expose `detect()` ou `read()`.
C'est ce qui permet de changer de moteur ou de modèle en modifiant le seul
fichier de configuration, et de tester au banc n'importe quelle combinaison.

Le moteur se déduit du chemin quand il n'est pas donné :

    models/exported/ncnn/detector/         -> ncnn   (contient model*.param)
    models/exported/ncnn/paddle_rec/model.ncnn.param -> ncnn
    engines/dfine.engine                   -> tensorrt
    models/dfine_n/best_stg2_single.onnx   -> onnx

"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

from src.common.preprocess import (REC_HEIGHT_DEFAUT, REC_WIDTH_DEFAUT,
                                      verify_model)
from src.common.base import (BACKENDS, InferenceBackend,
                               supported_combinations, deviner_backend,
                               verify_compatibilities)


def build_model(path: Union[str, Path],
                family: str,
                imgsz: int,
                num_classes: int = 1,
                backend: Optional[str] = None,
                charset: Optional[Sequence[str]] = None,
                rec_height: int = REC_HEIGHT_DEFAUT,
                rec_width: int = REC_WIDTH_DEFAUT,
                **options: Any) -> InferenceBackend:
    """Instancie le backend adapté. `backend=None` le déduit du chemin.

    Les `options` supplémentaires sont passées telles quelles au wrapper
    (use_int8, num_threads, use_vulkan pour ncnn ; providers pour ONNX...).
    Une option inconnue du wrapper visé lève un TypeError explicite plutôt
    que d'être ignorée en silence.
    """
    verify_model(family)
    backend = backend or deviner_backend(path)
    verify_compatibilities(family, backend)

    communs = dict(imgsz=imgsz, num_classes=num_classes, family=family,
                   charset=charset, rec_height=rec_height,
                   rec_width=rec_width)

    if backend == "ncnn":
        from src.wrappers.wrapper_ncnn import NCNNModel
        return NCNNModel(path, **communs, **options)
    if backend == "tensorrt":
        from src.wrappers.wrapper_tensorrt import TensorRTModel
        return TensorRTModel(path, **communs, **options)
    from src.wrappers.wrapper_onnx import ONNXModel
    return ONNXModel(path, **communs, **options)


def build_from_config(cfg: Dict[str, Any], section: str,
                      backend: Optional[str] = None) -> InferenceBackend:
    """Instancie un modèle depuis une section `models.<section>` de la config.

    Clés lues, dans l'ordre de priorité pour le chemin : `ncnn_param`,
    `engine`, `onnx`, `path`. Le charset vient de `charset_file` s'il est
    présent, sinon de `charset`.
    """
    bloc = (cfg.get("models") or {}).get(section)
    if bloc is None:
        disponibles = sorted((cfg.get("models") or {}).keys())
        raise KeyError(
            f"Section models.{section} absente de la configuration.\n"
            f"Sections présentes : {disponibles}")

    chemin = None
    for cle in ("ncnn_param", "engine", "onnx", "path"):
        if bloc.get(cle):
            chemin = bloc[cle]
            break
    if chemin is None:
        raise KeyError(
            f"models.{section} ne déclare aucun chemin de modèle "
            f"(ncnn_param, engine, onnx ou path).")

    charset = None
    if bloc.get("charset_file"):
        from src.models.postprocess_paddle_rec import load_charset
        charset = load_charset(bloc["charset_file"])
    elif bloc.get("charset"):
        charset = [str(c) for c in bloc["charset"]]

    noms = bloc.get("class_names") or {}
    num_classes = int(bloc.get("num_classes") or len(noms) or 1)

    return build_model(
        chemin,
        family=bloc.get("family", "yolo"),
        imgsz=int(bloc.get("imgsz", 640)),
        num_classes=num_classes,
        backend=backend or bloc.get("backend"),
        charset=charset,
        rec_height=int(bloc.get("height", REC_HEIGHT_DEFAUT)),
        rec_width=int(bloc.get("width", REC_WIDTH_DEFAUT)),
    )


def afficher_matrice() -> str:
    """Tableau des combinaisons famille x backend, pour l'aide en ligne."""
    from src.common.preprocess import FAMILIES
    from src.common.base import INCOMPATIBILTIES

    supportees = set(supported_combinations())
    lignes = [f"{'famille':<12}" + "".join(f"{b:>11}" for b in BACKENDS)]
    lignes.append("-" * len(lignes[0]))
    for f in FAMILIES:
        cases = "".join(f"{'oui' if (f, b) in supportees else 'non':>11}"
                        for b in BACKENDS)
        lignes.append(f"{f:<12}{cases}")
    for (f, b), raison in INCOMPATIBILTIES.items():
        lignes.append(f"\n{f} x {b} : {raison}")
    return "\n".join(lignes)


if __name__ == "__main__":
    print(afficher_matrice())