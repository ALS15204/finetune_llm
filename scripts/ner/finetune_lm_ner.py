from pathlib import Path

from transformers import TrainingArguments

from ner.dataset import (
    get_un_token_dataset,
    tokenize_and_align_labels,
)
from ner.model import BaseNerModel

TASK = "ner"
REPO_ROOT = Path(__file__).parent.parent


if __name__ == "__main__":
    model_checkpoint = "distilbert-base-uncased"
    # model_checkpoint = "gpt2"
    batch_size = 16

    train_dataset, test_dataset = get_un_token_dataset(
        REPO_ROOT / "UN-named-entity-recognition/tagged-training/",
        REPO_ROOT / "UN-named-entity-recognition/tagged-test/",
    )


    args = TrainingArguments(
        f"test-{TASK}",
        evaluation_strategy="epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=3,
        weight_decay=1e-5,
    )

    model = BaseNerModel(model_checkpoint=model_checkpoint)
    train_tokenized_datasets = train_dataset.map(
        tokenize_and_align_labels,
        batched=True,
        fn_kwargs={"tokenizer": model.tokenizer, "task": TASK},
    )
    test_tokenized_datasets = test_dataset.map(
        tokenize_and_align_labels,
        batched=True,
        fn_kwargs={"tokenizer": model.tokenizer, "task": TASK},
    )
    model.train(
        args=args,
        train_dataset=train_tokenized_datasets,
        test_dataset=test_tokenized_datasets,
    )
    model.save("un-ner-bert.model")
