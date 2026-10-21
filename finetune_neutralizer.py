# pip install transformers datasets accelerate torch pandas sentencepiece
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments,
)

MODEL = "google/flan-t5-base"   # use "google/flan-t5-small" if memory/time is tight
PREFIX = "Rewrite neutrally: "

# 1. Data: same CSV as before (columns: toxic, neutral)
df = pd.read_csv("data/rewriting_pairs.csv").dropna().drop_duplicates()
ds = Dataset.from_pandas(df, preserve_index=False).train_test_split(test_size=0.1, seed=42)

tok = AutoTokenizer.from_pretrained(MODEL)

def preprocess(batch):
    enc = tok([PREFIX + t for t in batch["toxic"]], max_length=64, truncation=True)
    enc["labels"] = tok(text_target=batch["neutral"], max_length=64, truncation=True)["input_ids"]
    return enc

tokenized = ds.map(preprocess, batched=True, remove_columns=ds["train"].column_names)

# 2. Model + training
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)

args = Seq2SeqTrainingArguments(
    output_dir="models/t5_neutralizer",
    learning_rate=3e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=4,
    eval_strategy="epoch",          # older transformers versions: evaluation_strategy="epoch"
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_total_limit=2,
    logging_steps=50,
    fp16=False,                     # set True on a CUDA GPU (not for T5 if you see NaN loss; then use bf16)
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    data_collator=DataCollatorForSeq2Seq(tok, model=model),
)
trainer.train()
trainer.save_model("models/t5_neutralizer")
tok.save_pretrained("models/t5_neutralizer")

# 3. Inference with sensible decoding settings
def neutralize(text):
    inputs = tok(PREFIX + text, return_tensors="pt", truncation=True, max_length=64).to(model.device)
    out = model.generate(
        **inputs,
        num_beams=5,
        no_repeat_ngram_size=3,
        max_new_tokens=64,
        early_stopping=True,
    )
    return tok.decode(out[0], skip_special_tokens=True)

if __name__ == "__main__":
    for s in ds["test"]["toxic"][:5]:
        print("IN :", s)
        print("OUT:", neutralize(s), "\n")