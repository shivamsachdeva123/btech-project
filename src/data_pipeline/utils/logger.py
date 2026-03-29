from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path


def build_logger(log_dir: str) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(log_dir) / f"pipeline_{timestamp}.log"

    logger = logging.getLogger("data_pipeline")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if script is re-run in the same process.
    if logger.handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)

    file_handler = logging.FileHandler(log_path)
    stream_handler = logging.StreamHandler()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.info("Logger initialized. Log file: %s", log_path)
    return logger
