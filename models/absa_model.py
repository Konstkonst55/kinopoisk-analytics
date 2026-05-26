import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from gensim.models import Word2Vec
from tqdm import tqdm

from backend.config import DATA_DIR, setup_logger
from utils.lemmatizer import CustomLemmatizer

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

MAX_SEQ_LEN = 50
BATCH_SIZE = 128
EPOCHS = 3
TRAIN_SAMPLES = 25000
SIMILARITY_THRESHOLD = 0.55


class TextDataset(Dataset):
    def __init__(self, texts, labels, w2v_model, lemmatizer):
        self.texts = texts
        self.labels = labels
        self.w2v_model = w2v_model
        self.lemmatizer = lemmatizer
        self.vocab = set(self.w2v_model.wv.index_to_key)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        lemmas = self.lemmatizer.lemmatize(text)
        vectorized = []

        for lemma in lemmas:
            if lemma in self.vocab:
                vectorized.append(self.w2v_model.wv[lemma])

                if len(vectorized) >= MAX_SEQ_LEN:
                    break

        if not vectorized:
            vectorized = [np.zeros(300)]

        pad_length = MAX_SEQ_LEN - len(vectorized)

        if pad_length > 0:
            vectorized.extend([np.zeros(300)] * pad_length)

        return (
            torch.from_numpy(np.asarray(vectorized, dtype=np.float32)),
            torch.tensor(label, dtype=torch.long),
        )


class SentimentBiLSTM(nn.Module):
    def __init__(self, input_dim=300, hidden_dim=64, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)
        hidden = self.dropout(hidden)

        return self.fc(hidden)


class AspectSentimentAnalyzer:
    def __init__(self, load_weights=True):
        self.logger = setup_logger("AspectSentimentAnalyzer")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.logger.info(f"Using device: {self.device}")

        self.model_weights_path = os.path.join(DATA_DIR, "custom_bilstm_weights.pth")
        self.w2v_path = os.path.join(DATA_DIR, "custom_word2vec.model")
        self.lemmatizer = CustomLemmatizer()
        self._load_w2v()
        self.classifier = None

        if load_weights:
            self._init_model()
        else:
            self.classifier = SentimentBiLSTM().to(self.device)

            self.logger.info("Model initialized without weights (training mode)")

    def _init_model(self):
        if not os.path.exists(self.model_weights_path):
            raise FileNotFoundError(f"ABSA weights not found: {self.model_weights_path}")

        self.logger.info("Loading pre-trained weights")

        self.classifier = SentimentBiLSTM().to(self.device)
        self.classifier.load_state_dict(torch.load(self.model_weights_path, map_location=self.device))
        self.classifier.eval()

    def _load_w2v(self):
        self.logger.info("Loading Word2Vec")

        self.w2v_model = Word2Vec.load(self.w2v_path)
        self.vocab = set(self.w2v_model.wv.index_to_key)

    def train(self):
        self.logger.info("Starting BiLSTM training")

        ds1_path = os.path.join(DATA_DIR, "dataset1.parquet")
        df = pd.read_parquet(ds1_path)

        if len(df) > TRAIN_SAMPLES:
            df = df.sample(n=TRAIN_SAMPLES, random_state=42).reset_index(drop=True)

        texts = df["review_text"].tolist()
        labels = df["sentiment"].tolist()
        dataset = TextDataset(texts, labels, self.w2v_model, self.lemmatizer)

        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.classifier.parameters(), lr=0.002)
        scaler = torch.cuda.amp.GradScaler()
        self.classifier.train()

        for epoch in range(EPOCHS):
            epoch_loss = 0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

            for batch_x, batch_y in progress_bar:
                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast():
                    outputs = self.classifier(batch_x)
                    loss = criterion(outputs, batch_y)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.item()
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            self.logger.info(f"Epoch {epoch + 1} completed. Loss: {epoch_loss / len(dataloader):.4f}")

        torch.save(self.classifier.state_dict(), self.model_weights_path)

        self.logger.info("BiLSTM training finished")

        self.classifier.eval()

    def _get_sentence_tensor(self, lemmas):
        vectorized = []

        for lemma in lemmas:
            if lemma in self.vocab:
                vectorized.append(self.w2v_model.wv[lemma])

                if len(vectorized) >= MAX_SEQ_LEN:
                    break

        if not vectorized:
            vectorized = [np.zeros(300)]

        pad_length = MAX_SEQ_LEN - len(vectorized)

        if pad_length > 0:
            vectorized.extend([np.zeros(300)] * pad_length)

        return torch.from_numpy(np.asarray([vectorized], dtype=np.float32)).to(self.device, non_blocking=True)

    def _cosine_sim(self, vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def predict(self, texts: list[str], aspects: list[str]) -> dict:
        aspect_stats = {a: {"pos": 0, "neg": 0, "neu": 0, "mentions": 0} for a in aspects}

        aspect_vecs = {a: self.w2v_model.wv[a] for a in aspects if a in self.vocab}
        self.classifier.eval()

        with torch.inference_mode():
            for text in texts:
                sentences = re.split(r"[.!?]+", text)

                for sentence in sentences:
                    if not sentence.strip():
                        continue

                    lemmas_list = self.lemmatizer.lemmatize(sentence)

                    if not lemmas_list:
                        continue

                    sentence_vectors = {
                        lemma: self.w2v_model.wv[lemma] for lemma in lemmas_list if lemma in self.vocab
                    }

                    if not sentence_vectors:
                        continue

                    mentioned_aspects = set()

                    for aspect, a_vec in aspect_vecs.items():
                        for _, s_vec in sentence_vectors.items():
                            sim = self._cosine_sim(a_vec, s_vec)

                            if sim > SIMILARITY_THRESHOLD:
                                mentioned_aspects.add(aspect)
                                break

                    if mentioned_aspects:
                        seq_tensor = self._get_sentence_tensor(lemmas_list)
                        pred_logits = self.classifier(seq_tensor)
                        pred_class = torch.argmax(pred_logits, dim=1).item()

                        for aspect in mentioned_aspects:
                            aspect_stats[aspect]["mentions"] += 1

                            if pred_class == 2:
                                aspect_stats[aspect]["pos"] += 1
                            elif pred_class == 0:
                                aspect_stats[aspect]["neg"] += 1
                            else:
                                aspect_stats[aspect]["neu"] += 1

        result = {}

        for aspect, stats in aspect_stats.items():
            total = stats["mentions"]

            if total > 0:
                result[aspect] = {
                    "pos": round((stats["pos"] / total) * 100, 1),
                    "neg": round((stats["neg"] / total) * 100, 1),
                    "neu": round((stats["neu"] / total) * 100, 1),
                    "mentions": total,
                }
            else:
                result[aspect] = {"pos": 0.0, "neg": 0.0, "neu": 0.0, "mentions": 0}

        return result
