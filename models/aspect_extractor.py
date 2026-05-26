import os
import time
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from gensim.models import Word2Vec
from tqdm import tqdm

from backend.config import DATA_DIR, get_resource_path, setup_logger
from utils.lemmatizer import CustomLemmatizer


class CustomSeededKMeans:
    def __init__(self, n_clusters, max_iter=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.labels_ = None
        self.cluster_centers_ = None

    def fit(self, X, seed_vectors=None):
        n_samples, n_features = X.shape

        if seed_vectors is not None and len(seed_vectors) > 0:
            seed_vectors = np.array(seed_vectors)

            if len(seed_vectors) < self.n_clusters:
                n_missing = self.n_clusters - len(seed_vectors)
                random_indices = np.random.choice(n_samples, n_missing, replace=False)
                random_points = X[random_indices]
                self.centroids = np.vstack([seed_vectors, random_points])
            else:
                self.centroids = seed_vectors[: self.n_clusters].copy()

        else:
            random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
            self.centroids = X[random_indices].copy()

        for i in range(self.max_iter):
            distances = cdist(X, self.centroids, metric="euclidean")
            labels = np.argmin(distances, axis=1)
            new_centroids = np.zeros((self.n_clusters, n_features))

            for k in range(self.n_clusters):
                cluster_points = X[labels == k]

                if len(cluster_points) > 0:
                    new_centroids[k] = np.mean(cluster_points, axis=0)
                else:
                    new_centroids[k] = X[np.random.choice(n_samples)]

            shift = np.linalg.norm(self.centroids - new_centroids)
            self.centroids = new_centroids

            if shift < self.tol:
                break

        self.labels_ = labels
        self.cluster_centers_ = self.centroids

        return self


class KMeansAspectExtractor:
    def __init__(self, n_clusters=4):
        self.logger = setup_logger("KMeansAspectExtractor")

        self.logger.info("Initializing KMeansAspectExtractor")

        start_time = time.time()
        self.lemmatizer = CustomLemmatizer()

        self.logger.info("Custom lemmatizer loaded")

        self.model_path = os.path.join(DATA_DIR, "custom_word2vec.model")
        self.n_clusters = n_clusters
        self.cinema_aspects = self._load_csv_list("cinema_aspects.csv")
        self.anchor_aspects = self._load_csv_list("anchor_aspects.csv")
        self._load_or_train_w2v()

        self.logger.info(f"Initialization completed in {time.time() - start_time:.2f}s")

    def _load_csv_list(self, filename):
        path = get_resource_path(filename)

        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def _load_or_train_w2v(self):
        if os.path.exists(self.model_path):
            self.logger.info(f"Loading existing Word2Vec model from {self.model_path}")

            self.word2vec_model = Word2Vec.load(self.model_path)
        else:
            self.logger.info("Word2Vec model not found. Starting training on dataset1.parquet")

            ds1_path = os.path.join(DATA_DIR, "dataset1.parquet")

            if not os.path.exists(ds1_path):
                self.logger.error("dataset1.parquet not found for Word2Vec training")

                raise FileNotFoundError("dataset1.parquet is required to train Word2Vec")

            df = pd.read_parquet(ds1_path)
            texts = df["review_text"].tolist()
            sentences = []

            for text in tqdm(texts, desc="Tokenizing texts for Word2Vec"):
                sentences.append(self.lemmatizer.lemmatize(text))

            self.logger.info("Training Word2Vec model")

            self.word2vec_model = Word2Vec(sentences=sentences, vector_size=300, window=5, min_count=2, workers=4)

            self.word2vec_model.save(self.model_path)

            self.logger.info(f"Word2Vec model trained and saved to {self.model_path}")

    def fit_predict(self, texts: list[str]) -> list[str]:
        self.logger.info(f"Starting aspect extraction. Input texts count: {len(texts)}")

        candidates = set()
        token_count = 0

        for text in texts:
            lemmas = self.lemmatizer.lemmatize(text)

            for lemma in lemmas:
                token_count += 1
                candidates.add(lemma)

        candidates_list = list(candidates)

        self.logger.info(f"Processed {token_count} tokens. Extracted {len(candidates_list)} unique candidates")

        valid_words = []
        embeddings = []

        for word in candidates_list:
            if word in self.word2vec_model.wv:
                valid_words.append(word)
                embeddings.append(self.word2vec_model.wv[word])

        actual_clusters = min(self.n_clusters, len(valid_words))

        if actual_clusters == 0:
            fallback_aspects = ["сюжет", "актер", "визуал", "атмосфер"]

            self.logger.warning(f"No valid embeddings found. Returning fallback aspects: {fallback_aspects}")

            return fallback_aspects

        embeddings_array = np.array(embeddings)
        seed_vectors = []

        for word in self.anchor_aspects:
            if word in self.word2vec_model.wv:
                seed_vectors.append(self.word2vec_model.wv[word])
            if len(seed_vectors) == actual_clusters:
                break

        kmeans = CustomSeededKMeans(n_clusters=actual_clusters)
        kmeans.fit(embeddings_array, seed_vectors=seed_vectors)
        aspect_vocab = []
        aspect_vecs = []

        for word in self.cinema_aspects:
            if word in self.word2vec_model.wv:
                aspect_vocab.append(word)
                aspect_vecs.append(self.word2vec_model.wv[word])

        aspect_vecs_array = np.array(aspect_vecs)
        distances = cdist(kmeans.cluster_centers_, aspect_vecs_array, metric="euclidean")

        final_aspects = []

        for i in range(actual_clusters):
            sorted_indices = np.argsort(distances[i])

            for idx in sorted_indices:
                word = aspect_vocab[idx]

                if word not in final_aspects:
                    final_aspects.append(word)
                    break

        self.logger.info(f"Final extracted aspects: {final_aspects}")

        return final_aspects
