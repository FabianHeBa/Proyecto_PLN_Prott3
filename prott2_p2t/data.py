import os
from typing import Optional

import pandas as pd
from datasets import load_dataset
from huggingface_hub import login

from .config import Config


def login_huggingface(token: Optional[str] = None):
    token = token or os.getenv("HF_TOKEN")
    if token:
        login(token=token)
    else:
        print("HF_TOKEN no está definido. Se intentará cargar el dataset/modelos sin login.")


def extract_conversations(conversation):
    question = conversation[0]["value"]
    question = question.replace("<protein_sequence>\n", "")
    answer = conversation[1]["value"]
    return question, answer


def add_question_answer_columns(df: pd.DataFrame):
    df = df.reset_index(drop=True).copy()

    for i in range(len(df)):
        question, answer = extract_conversations(df.iloc[i]["conversations"])
        df.at[i, "question"] = question
        df.at[i, "answer"] = answer

    return df


def load_and_prepare_data(config: Config):
    p2tqa = load_dataset(config.dataset_id)

    source_data = p2tqa[config.source_split]
    min_samp = int(len(source_data) * config.sample_fraction)

    data_small = source_data.shuffle(seed=config.seed).select(range(min_samp))
    split = data_small.train_test_split(
        test_size=config.validation_size,
        seed=config.seed,
    )

    train_data_df = split["train"].to_pandas().reset_index(drop=True)
    val_data_df = split["test"].to_pandas().reset_index(drop=True)

    train_data_df = add_question_answer_columns(train_data_df)
    val_data_df = add_question_answer_columns(val_data_df)

    print(train_data_df.info())

    return train_data_df, val_data_df
