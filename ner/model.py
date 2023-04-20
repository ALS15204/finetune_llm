from typing import List, Union

from transformers import (
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
    PreTrainedTokenizerFast,
)
from ner.dataset import LABEL_LIST
from ner.metrics import compute_metrics


class BaseNerModel:
    def __init__(self, model_checkpoint: str, label_list: List = LABEL_LIST):
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_checkpoint, num_labels=len(label_list)
        )
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
            model_checkpoint, add_prefix_space=True
        )
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self._trainer = None

    @property
    def trainer(self) -> Union[None, Trainer]:
        return self._trainer

    def train(self, args: TrainingArguments, train_dataset, test_dataset):
        data_collator = DataCollatorForTokenClassification(self.tokenizer)
        if not self._trainer:
            self._trainer = Trainer(
                self.model,
                args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                data_collator=data_collator,
                tokenizer=self.tokenizer,
                compute_metrics=compute_metrics,
            )
        self._trainer.train()
        self._trainer.evaluate()

    def save(self, output_path: str):
        if self._trainer:
            self.trainer.save_model(output_path)
        else:
            self.model.save_pretrained(output_path)
