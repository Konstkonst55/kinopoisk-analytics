import pandas as pd

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.aspect_extractor import KMeansAspectExtractor
from backend.config import setup_logger, DATA_DIR

logger = setup_logger("TrainWord2Vec")

if __name__ == "__main__":
    force_train = os.environ.get("FORCE_TRAIN", "false").lower() == "true"
    model_path = os.path.join(DATA_DIR, "custom_word2vec.model")

    if not force_train and os.path.exists(model_path):
        logger.info("Word2Vec model already exists. Skipping training.")
        sys.exit(0)

    ds1_path = os.path.join(DATA_DIR, "dataset1.parquet")

    if not os.path.exists(ds1_path):
        raise FileNotFoundError(f"Run data_pipeline/preprocess.py first to create {ds1_path}")

    logger.info("Training Word2Vec via KMeansAspectExtractor...")
    extractor = KMeansAspectExtractor(n_clusters=4)
    logger.info("Word2Vec model is ready.")
