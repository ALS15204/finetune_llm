import itertools
import os

import pandas as pd
from datasets import Dataset

LABEL_LIST = [
    "O",
    "B-MISC",
    "I-MISC",
    "B-PER",
    "I-PER",
    "B-ORG",
    "I-ORG",
    "B-LOC",
    "I-LOC",
]
LABEL_ENCODING_DICT = {
    "I-PRG": 2,
    "I-I-MISC": 2,
    "I-OR": 6,
    "O": 0,
    "I-": 0,
    "VMISC": 0,
    "B-PER": 3,
    "I-PER": 4,
    "B-ORG": 5,
    "I-ORG": 6,
    "B-LOC": 7,
    "I-LOC": 8,
    "B-MISC": 1,
    "I-MISC": 2,
}


def get_all_tokens_and_ner_tags(directory):
    return (
        pd.concat(
            [
                get_tokens_and_ner_tags(os.path.join(directory, filename))
                for filename in os.listdir(directory)
            ]
        )
        .reset_index()
        .drop("index", axis=1)
    )


def get_tokens_and_ner_tags(filename):
    with open(filename, "r", encoding="utf8") as f:
        lines = f.readlines()
        split_list = [
            list(y) for x, y in itertools.groupby(lines, lambda z: z == "\n") if not x
        ]
        tokens = [[x.split("\t")[0] for x in y] for y in split_list]
        entities = [[x.split("\t")[1][:-1] for x in y] for y in split_list]
    return pd.DataFrame({"tokens": tokens, "ner_tags": entities})


def get_un_token_dataset(train_directory, test_directory):
    train_df = get_all_tokens_and_ner_tags(train_directory)
    test_df = get_all_tokens_and_ner_tags(test_directory)
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    return (train_dataset, test_dataset)


def tokenize_and_align_labels(examples, tokenizer, task):
    label_all_tokens = True
    tokenized_inputs = tokenizer(
        list(examples["tokens"]),
        truncation=True,
        is_split_into_words=True,
    )

    labels = []
    for i, label in enumerate(examples[f"{task}_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif label[word_idx] == "0":
                label_ids.append(0)
            elif word_idx != previous_word_idx:
                label_ids.append(LABEL_ENCODING_DICT[label[word_idx]])
            else:
                label_ids.append(
                    LABEL_ENCODING_DICT[label[word_idx]] if label_all_tokens else -100
                )
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs
