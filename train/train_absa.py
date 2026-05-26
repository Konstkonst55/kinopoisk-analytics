import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.absa_model import AspectSentimentAnalyzer

if __name__ == "__main__":
    model = AspectSentimentAnalyzer(load_weights=False)
    model.train()

    if not model.model_weights_path:
        raise RuntimeError("Weights path missing")
