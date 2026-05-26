import logging
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_FILE = os.path.join(BASE_DIR, "app.log")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
RESOURCES_DIR = os.path.join(BASE_DIR, "resources")


def get_resource_path(filename):
    return os.path.join(RESOURCES_DIR, filename)


def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger
