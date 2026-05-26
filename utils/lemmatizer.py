import os
import re

from backend.config import get_resource_path


class CustomLemmatizer:
    def __init__(self):
        self.stop_words = self._load_stop_words()
        self.suffixes = self._load_suffixes()

    def _load_stop_words(self):
        path = get_resource_path("stop_words.csv")

        with open(path, "r", encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}

    def _load_suffixes(self):
        path = get_resource_path("suffixes.csv")

        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def lemmatize(self, text):
        words = re.findall(r"[а-яё]+", text.lower())
        lemmas = []

        for word in words:
            if word in self.stop_words or len(word) <= 2:
                continue

            stem = word

            for suffix in self.suffixes:
                if word.endswith(suffix):
                    stem = word[: -len(suffix)]
                    break

            if len(stem) > 2:
                lemmas.append(stem)

        return lemmas
