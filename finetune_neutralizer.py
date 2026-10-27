"""Fine-tuning script for training T5 neutralization model on constructive sentence pairs."""

from pathlib import Path
import pandas as pd
from datasets import Dataset
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from src.config import REWRITER_INSTRUCTION

BASE_DIR = Path(__file__).resolve().parent
TRAIN_DATA_PATH = BASE_DIR / "data" / "constructive_train.csv"
VAL_DATA_PATH = BASE_DIR / "data" / "constructive_val.csv"
OUTPUT_DIR = BASE_DIR / "models" / "t5_neutralizer"

MODEL_NAME = "google/flan-t5-base"
MAX_SOURCE_LEN = 128
MAX_TARGET_LEN = 64


def prepare_dataset():
    if not TRAIN_DATA_PATH.exists() or not VAL_DATA_PATH.exists():
        from src.rewriting.prepare_constructive_data import main as prep_main
        prep_main()

    train_df = pd.read_csv(TRAIN_DATA_PATH).dropna()
    val_df = pd.read_csv(VAL_DATA_PATH).dropna()

    train_ds = Dataset.from_pandas(train_df, preserve_index=False)
    val_ds = Dataset.from_pandas(val_df, preserve_index=False)
    return train_ds, val_ds


def train():
    train_ds, val_ds = prepare_dataset()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.config.tie_word_embeddings = False

    def preprocess(batch):
        model_inputs = tokenizer(batch["input_text"], max_length=MAX_SOURCE_LEN, truncation=True)
        labels = tokenizer(text_target=batch["target_text"], max_length=MAX_TARGET_LEN, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_train = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
    tokenized_val = val_ds.map(preprocess, batched=True, remove_columns=val_ds.column_names)

    use_cuda = torch.cuda.is_available()

    args = Seq2SeqTrainingArguments(
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
        save_total_limit=1,
        logging_steps=50,
        report_to="none",
        fp16=use_cuda,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )

    trainer.train()
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"Fine-tuned constructive T5 model successfully saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    train()
