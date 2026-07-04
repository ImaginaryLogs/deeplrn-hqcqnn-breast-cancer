"""
utils/logging_setup.py
----------------------
Centralised logging configuration.

Usage:
    from utils.logging_setup import setup_logging
    setup_logging()          # INFO to console + rotating file
    setup_logging(debug=True) # DEBUG everywhere
"""

import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(
    debug: bool = False,
    log_dir: str = "logs",
    prefix: str = "collector",
) -> None:
    log_level = logging.DEBUG if debug else logging.INFO
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"{prefix}_{timestamp}.log"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG) 

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # Rotating file handler — keeps last 5 × 10 MB files
    fh = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)

    logging.info(f"[Logging] Initialised — level={logging.getLevelName(log_level)} file={log_file}")
