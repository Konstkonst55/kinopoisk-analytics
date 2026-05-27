import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.absa_model import AspectSentimentAnalyzer

if __name__ == "__main__":
    from backend.config import DATA_DIR, setup_logger

    logger = setup_logger("TrainABSA")

    force_train = os.environ.get("FORCE_TRAIN", "false").lower() == "true"
    weights_path = os.path.join(DATA_DIR, "custom_bilstm_weights.pth")

    if not force_train and os.path.exists(weights_path):
        logger.info("ABSA model already exists. Skipping training.")
        sys.exit(0)

    model = AspectSentimentAnalyzer(load_weights=False)
    model.train()

    if not model.model_weights_path:
        raise RuntimeError("Weights path missing")
