#!/usr/bin/env python3
"""
logger.py -- Factory de logger centralisée avec flags on/off.

Contrôlable de 2 façons (priorité : config > env > défauts) :
  - variable d'env : ALPR_LOG_LEVEL=DEBUG, ALPR_LOG_CONSOLE=0, ALPR_LOG_FILE=/tmp/alpr.log
  - code/config    : get_logger("x", enabled=False, console=False, log_file="...", file_level="WARNING")
"""
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

_CONFIGURED: set[str] = set()

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str,
               enabled: bool | None = None,      # None = logger actif
               console: bool | None = None,      # None = handler console ON
               log_file: str | None = None,      # fichier rotatif optionnel
               level: str | None = None,
               file_level: str | None = None,
               max_bytes: int = 5 * 1024 * 1024,
               backup_count: int = 3) -> logging.Logger:
    """
    - enabled=False  -> logger muet (no-op), zéro coût
    - console=False  -> pas de sortie terminal (utile en production headless)
    - log_file       -> RotatingFileHandler (5 Mo x 3 par défaut, pas de SD saturée)
    - file_level     -> niveau distinct pour le fichier (ex: WARNING en prod, DEBUG en dev)
    """
    logger = logging.getLogger(name)

    if name in _CONFIGURED:
        return logger

    # résolution : argument > variable d'env > défaut
    if enabled is None:
        enabled = os.environ.get("ALPR_LOG_ENABLED", "1") != "0"
    if console is None:
        console = os.environ.get("ALPR_LOG_CONSOLE", "1") != "0"
    if log_file is None:
        log_file = os.environ.get("ALPR_LOG_FILE") or None
    if level is None:
        level = os.environ.get("ALPR_LOG_LEVEL", "INFO")
    if file_level is None:
        file_level = os.environ.get("ALPR_LOG_FILE_LEVEL", level)

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    if not enabled:
        logger.addHandler(logging.NullHandler())
        _CONFIGURED.add(name)
        return logger

    if console:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
        logger.addHandler(h)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
        fh.setLevel(getattr(logging, file_level.upper(), logging.INFO))
        fh.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
        logger.addHandler(fh)

    _CONFIGURED.add(name)
    return logger
