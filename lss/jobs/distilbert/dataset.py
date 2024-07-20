import os
from collections import Counter

import pandas as pd
from datasets import ClassLabel, Dataset, Features, Value
from sklearn.utils import resample
from sqlalchemy import Engine, create_engine
from transformers import AutoTokenizer

POS = "POSITIVE"
NEG = "NEGATIVE"


def build_dataset_and_upload(training_input_path: str, test_input_path: str):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize(entries):
        return tokenizer(entries["title"], padding="max_length", truncation=True)

    engine: Engine = create_engine(url=os.environ["SQLALCHEMY_URL"])
    df = pd.read_sql("SELECT title,label FROM item WHERE label IS NOT NULL", engine)

    counts = Counter(df["label"])
    n_samples = max(counts.values())
    minority_label = POS if counts[POS] < counts[NEG] else NEG
    majority_label = int(not minority_label)
    minority_upsampled_df = resample(
        df[df.label == minority_label], replace=True, n_samples=n_samples
    )
    upsampled_df = pd.concat(
        [df[df.label == majority_label], minority_upsampled_df], ignore_index=True
    )

    features = Features(
        {"title": Value("string"), "label": ClassLabel(names=[NEG, POS])}
    )

    dataset = Dataset.from_pandas(upsampled_df, features=features).map(
        tokenize, batched=True
    )
    dataset = dataset.remove_columns(["title"])
    train_test_split = dataset.train_test_split(
        test_size=0.1, stratify_by_column="label"
    )
    train_dataset = train_test_split["train"]
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    test_dataset = train_test_split["test"]
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    train_dataset.save_to_disk(training_input_path)
    test_dataset.save_to_disk(test_input_path)
