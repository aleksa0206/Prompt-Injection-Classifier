"""Fine-tune a small transformer. Usage: python train.py [hf_model_name]"""
import sys

import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from data import report

MODEL = sys.argv[1] if len(sys.argv) > 1 else "answerdotai/ModernBERT-base"
OUT = "model"

tok = AutoTokenizer.from_pretrained(MODEL)


def encode(df):
    ds = Dataset.from_pandas(df[["text", "label"]], preserve_index=False)
    return ds.map(lambda b: tok(b["text"], truncation=True, max_length=512), batched=True)


train = pd.read_parquet("data/train.parquet").sample(frac=1, random_state=42)
test = pd.read_parquet("data/test.parquet")
val, train = train.iloc[: len(train) // 10], train.iloc[len(train) // 10 :]

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, num_labels=2, id2label={0: "SAFE", 1: "INJECTION"}, label2id={"SAFE": 0, "INJECTION": 1}
)
bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
args = TrainingArguments(
    OUT,
    num_train_epochs=3,
    learning_rate=3e-5,
    warmup_steps=150,  # ~10% of ~1.5k total steps
    weight_decay=0.01,
    per_device_train_batch_size=8,  # sized for 6 GB VRAM; effective batch 16
    gradient_accumulation_steps=2,
    per_device_eval_batch_size=16,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    bf16=bf16,
    fp16=torch.cuda.is_available() and not bf16,
    report_to="none",
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=encode(train),
    eval_dataset=encode(val),
    processing_class=tok,
    data_collator=DataCollatorWithPadding(tok),
    compute_metrics=lambda p: {"f1": f1_score(p.label_ids, p.predictions.argmax(-1))},
)
trainer.train()
trainer.save_model(OUT)
report(test, trainer.predict(encode(test)).predictions.argmax(-1))
