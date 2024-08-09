import os

import pandas as pd
from datasets import ClassLabel, Dataset, Features, Value
from sklearn.model_selection import train_test_split
from sqlalchemy import Engine, create_engine
from transformers import AutoTokenizer

from ...lib.utils import upsample_dataframe_by_label


def build_dataset_and_upload(training_input_path: str, test_input_path: str):

    engine: Engine = create_engine(url=os.environ["SQLALCHEMY_URL"])
    df = pd.read_sql("SELECT title,label FROM item WHERE label IS NOT NULL", engine)
    train, test = train_test_split(
        df,
        test_size=0.1,
        stratify=df[["label"]],
    )
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize(entries):
        return tokenizer(entries["title"], padding="max_length", truncation=True)

    features = Features(
        {
            "title": Value("string"),
            "label": ClassLabel(names=["POOR", "AVERAGE", "GOOD"]),
        }
    )

    for df, path in [(train, training_input_path), (test, test_input_path)]:

        upsampled_df = upsample_dataframe_by_label(df)
        ds = (
            Dataset.from_pandas(upsampled_df, features=features)
            .map(tokenize, batched=True)
            .rename_column("label", "labels")
            .remove_columns("title")
        )
        ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        ds.save_to_disk(path)
