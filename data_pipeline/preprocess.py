import os
import re
import logging
import pandas as pd
from tqdm import tqdm
from backend.config import get_resource_path

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
LOG_FILE = os.path.join(BASE_DIR, "app.log")

FOLDERS = {"neg": 0, "neu": 1, "pos": 2}


def _load_markers():
    path = get_resource_path("markers.csv")

    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


MARKERS = _load_markers()
MARKERS_PATTERN = re.compile(r"(?i)(" + "|".join(re.escape(m) for m in MARKERS) + r")\s*(.*)", re.DOTALL)
WHITESPACE_PATTERN = re.compile(r"\s+")
RATING_PATTERN = re.compile(r"(\d+(?:[.,]\d+)?)\s*из\s*(\d+)", re.IGNORECASE)


class DataPreprocessor:
    def __init__(self):
        self.logger = self._setup_logger()
        self.data1 = []
        self.data2 = []
        self.stats = {"found": 0, "read": 0, "skipped": 0, "summaries_extracted": 0}

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("DataPipeline")
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

    def _normalize_text(self, text: str) -> str:
        text = WHITESPACE_PATTERN.sub(" ", text)

        return text.strip()

    def _extract_summary(self, text: str) -> str | None:
        match = MARKERS_PATTERN.search(text)

        if match:
            summary = self._normalize_text(match.group(2))

            if len(summary) > 15:
                return summary

        return None

    def _extract_rating(self, text: str) -> float:
        matches = RATING_PATTERN.findall(text)

        if matches:
            score_str = matches[-1][0].replace(",", ".")

            try:
                return float(score_str)
            except ValueError:
                return -1.0

        return -1.0

    def _process_file(self, file_path: str, movie_id: str, label: int):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_text = f.read()

            if not raw_text.strip():
                self.stats["skipped"] += 1
                return

            normalized_text = self._normalize_text(raw_text)
            user_rating = self._extract_rating(raw_text)

            self.data1.append(
                {
                    "movie_id": movie_id,
                    "review_text": normalized_text,
                    "sentiment": label,
                    "user_rating": user_rating,
                }
            )

            summary_text = self._extract_summary(raw_text)

            if summary_text:
                self.data2.append(
                    {
                        "movie_id": movie_id,
                        "text": normalized_text,
                        "summary": summary_text,
                    }
                )

                self.stats["summaries_extracted"] += 1

            self.stats["read"] += 1
        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {e}")

            self.stats["skipped"] += 1

    def _process_folder(self, folder_name: str, label: int):
        folder_path = os.path.join(DATA_DIR, folder_name)

        if not os.path.exists(folder_path):
            self.logger.error(f"Directory not found: {folder_path}")

            return

        files = os.listdir(folder_path)
        self.stats["found"] += len(files)

        for filename in tqdm(files, desc=f"Processing {folder_name}"):
            if not filename.endswith(".txt"):
                self.stats["skipped"] += 1
                continue

            parts = filename.replace(".txt", "").split("-")

            if len(parts) < 2:
                self.stats["skipped"] += 1
                continue

            self._process_file(os.path.join(folder_path, filename), parts[0], label)

    def _log_dataframe_stats(self, df: pd.DataFrame, name: str):
        self.logger.info(f"{name} shape: {df.shape}")

        mem_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)

        self.logger.info(f"Memory usage - {name}: {mem_usage:.2f} MB")

        missing = df.isna().sum().sum()

        self.logger.info(f"NaN values - {name}: {missing}")

    def run(self):
        self.logger.info("Starting data preprocessing pipeline")

        os.makedirs(PROCESSED_DIR, exist_ok=True)

        for folder, label in FOLDERS.items():
            self._process_folder(folder, label)

        self.logger.info(
            f'Stats - Found: {self.stats["found"]}, Read: {self.stats["read"]}, '
            f'Skipped: {self.stats["skipped"]}, '
            f'Summaries: {self.stats["summaries_extracted"]}'
        )

        df1 = pd.DataFrame(self.data1)
        df2 = pd.DataFrame(self.data2)

        self._log_dataframe_stats(df1, "Dataset 1")
        self._log_dataframe_stats(df2, "Dataset 2")

        ds1_path = os.path.join(PROCESSED_DIR, "dataset1.parquet")
        ds2_path = os.path.join(PROCESSED_DIR, "dataset2.parquet")

        df1.to_parquet(ds1_path, index=False)
        df2.to_parquet(ds2_path, index=False)

        self.logger.info(f"Datasets saved to {PROCESSED_DIR} in Parquet format")


if __name__ == "__main__":
    processor = DataPreprocessor()
    processor.run()
