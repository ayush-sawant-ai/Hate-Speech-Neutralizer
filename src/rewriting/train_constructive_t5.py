"""Fine-tunes Flan-T5 on the curated constructive neutralization dataset.
Uses Apple Silicon MPS / CUDA acceleration for fast, reliable convergence.
"""

from pathlib import Path
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
TRAIN_CSV = BASE_DIR / "data" / "constructive_train.csv"
VAL_CSV = BASE_DIR / "data" / "constructive_val.csv"
OUTPUT_DIR = BASE_DIR / "models" / "t5_neutralizer"
BASE_MODEL_NAME = "google/flan-t5-base"

MAX_SOURCE_LEN = 128
MAX_TARGET_LEN = 64


def train_model():
    print(f"Loading base model: {BASE_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_NAME)

    # Disable tied word embeddings warning in config
    model.config.tie_word_embeddings = False

    train_df = pd.read_csv(TRAIN_CSV).dropna()
    val_df = pd.read_csv(VAL_CSV).dropna()

    train_ds = Dataset.from_pandas(train_df, preserve_index=False)
    val_ds = Dataset.from_pandas(val_df, preserve_index=False)

    def preprocess_batch(batch):
        inputs = tokenizer(
            batch["input_text"],
            max_length=MAX_SOURCE_LEN,
            truncation=True,
            padding=False,
        )
        labels = tokenizer(
            text_target=batch["target_text"],
            max_length=MAX_TARGET_LEN,
            truncation=True,
            padding=False,
        )
        inputs["labels"] = labels["input_ids"]
        return inputs

    print("Tokenizing datasets...")
    tokenized_train = train_ds.map(preprocess_batch, batched=True, remove_columns=train_ds.column_names)
    tokenized_val = val_ds.map(preprocess_batch, batched=True, remove_columns=val_ds.column_names)

    use_cuda = torch.cuda.is_available()
    use_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    print(f"Compute device: {'CUDA' if use_cuda else ('Apple Silicon MPS' if use_mps else 'CPU')}")

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),
        learning_rate=2.5e-4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=2,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        save_total_limit=1,
        report_to="none",
        fp16=use_cuda,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
    )

    print("Starting fine-tuning...")
    trainer.train()

    print(f"Saving fine-tuned constructive neutralizer to {OUTPUT_DIR}...")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print("Fine-tuning completed successfully!")


if __name__ == "__main__":
    train_model()
