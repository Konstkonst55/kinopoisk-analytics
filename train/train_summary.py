import pandas as pd

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.config import DATA_DIR, setup_logger
from models.summary_model import ReviewSummarizer

logger = setup_logger("TrainSummary")

if __name__ == "__main__":
    force_train = os.environ.get("FORCE_TRAIN", "false").lower() == "true"
    weights_dir = os.path.join(DATA_DIR, "movie_summary_model")

    if not force_train and os.path.exists(os.path.join(weights_dir, "config.json")):
        logger.info("Summary model already exists. Skipping training.")
        sys.exit(0)

    ds2_path = os.path.join(DATA_DIR, "dataset2.parquet")

    if not os.path.exists(ds2_path):
        logger.error(f"Dataset not found at {ds2_path}. Run preprocess.py first.")
        raise FileNotFoundError(f"Missing {ds2_path}")

    logger.info("Loading dataset2.parquet for summarization training")

    df = pd.read_parquet(ds2_path)

    if df.empty:
        logger.error("dataset2.parquet is empty. Check preprocessing logic.")
        raise ValueError("Empty dataset")

    train_data = df.to_dict("records")
    logger.info(f"Loaded {len(train_data)} samples for fine-tuning")

    model = ReviewSummarizer(load_weights=False)
    model.train(train_data=train_data, epochs=3, batch_size=1, lr=1e-4, accumulation_steps=4)
