import os
import re
import time
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, T5ForConditionalGeneration, GenerationConfig
from tqdm import tqdm

from backend.config import DATA_DIR, setup_logger

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


class MovieReviewsDataset(Dataset):
    def __init__(self, data, tokenizer, max_source_length=512, max_target_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = re.sub(r"\s+", " ", str(item.get("text", ""))).strip()
        summary = re.sub(r"\s+", " ", str(item.get("summary", ""))).strip()

        source = self.tokenizer(
            text,
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        target = self.tokenizer(
            summary,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = target["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": source["input_ids"].squeeze(),
            "attention_mask": source["attention_mask"].squeeze(),
            "labels": labels,
        }


class ReviewSummarizer:
    def __init__(self, load_weights=True):
        self.logger = setup_logger("ReviewSummarizer")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.logger.info(f"Using device: {self.device}")

        self.model_name = "IlyaGusev/rut5_base_sum_gazeta"
        self.weights_dir = os.path.join(DATA_DIR, "movie_summary_model")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, legacy=False)

        is_model_exist = os.path.exists(os.path.join(self.weights_dir, "config.json"))

        if load_weights and is_model_exist:
            self.logger.info("Loading fine-tuned movie summarization model")

            self.model = T5ForConditionalGeneration.from_pretrained(self.weights_dir).to(self.device)
        else:
            self.logger.info("Loading baseline pre-trained model")

            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name).to(self.device)

        self.model.eval()

    def train(self, train_data, epochs=3, batch_size=4, lr=3e-5):
        self.logger.info("Starting fine-tuning on custom movie reviews dataset")
        train_dataset = MovieReviewsDataset(train_data, self.tokenizer)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        scaler = torch.cuda.amp.GradScaler()
        self.model.train()
        torch.cuda.empty_cache()

        for epoch in range(epochs):
            epoch_loss = 0
            progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

            for batch in progress:
                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
                labels = batch["labels"].to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(dtype=torch.float16):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )

                    loss = outputs.loss

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.item()
                progress.set_postfix({"loss": f"{loss.item():.4f}"})

            self.logger.info(f"Epoch {epoch + 1} completion. Average Loss: {epoch_loss / len(train_loader):.4f}")

        os.makedirs(self.weights_dir, exist_ok=True)
        self.model.save_pretrained(self.weights_dir)
        self.tokenizer.save_pretrained(self.weights_dir)

        self.logger.info(f"Model successfully fine-tuned and saved to {self.weights_dir}")

        self.model.eval()

    def generate(self, texts: list[str]) -> str:
        start_time = time.time()

        if not texts:
            return ""

        cleaned_texts = []

        for t in texts:
            t_clean = re.sub(r"\s+", " ", str(t)).strip()

            if t_clean:
                cleaned_texts.append(t_clean)

        if not cleaned_texts:
            return ""

        combined_text = " ".join(cleaned_texts)
        tokens = self.tokenizer(combined_text, return_tensors="pt", truncation=False)["input_ids"].squeeze()

        if tokens.dim() == 0:
            tokens = tokens.unsqueeze(0)

        max_chunk = 600
        chunks = [tokens[i : i + max_chunk] for i in range(0, len(tokens), max_chunk)][:5]
        chunk_summaries = []

        generation_config = GenerationConfig(
            max_length=150,
            min_length=30,
            num_beams=4,
            repetition_penalty=2.5,
            no_repeat_ngram_size=3,
            encoder_no_repeat_ngram_size=3,
            early_stopping=True,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        with torch.inference_mode():
            for chunk in chunks:
                if chunk[-1] != self.tokenizer.eos_token_id:
                    chunk = torch.cat([chunk, torch.tensor([self.tokenizer.eos_token_id])])

                chunk_input = chunk.unsqueeze(0).to(self.device)

                output_ids = self.model.generate(input_ids=chunk_input, generation_config=generation_config)
                summary_part = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

                if summary_part.strip():
                    chunk_summaries.append(summary_part)

        if not chunk_summaries:
            return ""

        if len(chunk_summaries) > 1:
            final_combined = " ".join(chunk_summaries)
            final_input = self.tokenizer(final_combined, return_tensors="pt", max_length=1024, truncation=True).to(
                self.device
            )

            final_config = GenerationConfig(
                max_length=250,
                min_length=40,
                num_beams=4,
                repetition_penalty=2.5,
                no_repeat_ngram_size=3,
                encoder_no_repeat_ngram_size=3,
                early_stopping=True,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            with torch.inference_mode():
                final_output_ids = self.model.generate(
                    input_ids=final_input["input_ids"],
                    attention_mask=final_input["attention_mask"],
                    generation_config=final_config,
                )

            final_summary = self.tokenizer.decode(final_output_ids[0], skip_special_tokens=True)
        else:
            final_summary = chunk_summaries[0]

        self.logger.info(f"Generation completed in {time.time() - start_time:.4f}s")

        return final_summary
