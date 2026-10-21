"""Merge public prompt-injection datasets into one train/test set. label: 1 = injection, 0 = safe."""
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.metrics import classification_report, f1_score

SOURCES = {
    "deepset/prompt-injections": lambda df: df[["text", "label"]],
    "xTRam1/safe-guard-prompt-injection": lambda df: df[["text", "label"]],
    "jackhhao/jailbreak-classification": lambda df: pd.DataFrame(
        {"text": df["prompt"], "label": (df["type"] == "jailbreak").astype(int)}
    ),
}


def build(split):
    parts = [fix(load_dataset(name, split=split).to_pandas()).assign(source=name) for name, fix in SOURCES.items()]
    return pd.concat(parts, ignore_index=True).dropna().drop_duplicates("text")


def report(df, preds):
    print(classification_report(df.label, preds, target_names=["safe", "injection"], digits=3))
    for src, g in df.assign(pred=preds).groupby("source"):
        print(f"{src:40s} n={len(g):5d}  F1={f1_score(g.label, g.pred):.3f}")


if __name__ == "__main__":
    train, test = build("train"), build("test")
    test = test[~test.text.isin(train.text)]  # no test leakage into training
    assert not set(train.text) & set(test.text)

    Path("data").mkdir(exist_ok=True)
    train.to_parquet("data/train.parquet")
    test.to_parquet("data/test.parquet")
    for name, df in (("train", train), ("test", test)):
        print(f"\n{name}: {len(df)} rows")
        print(df.groupby(["source", "label"]).size().unstack(fill_value=0))
