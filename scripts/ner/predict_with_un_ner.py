import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification

from ner.dataset import LABEL_LIST

pretrained_model = "./un-ner-bert.model/"
tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

paragraph = """Before proceeding further, I should like to inform members that action on draft resolution iv, 
entitled situation of human rights of Rohingya Muslims and other minorities in Myanmar is postponed to a later date 
to allow time for the review of its programme budget implications by the fifth committee. The assembly will take 
action on draft resolution iv as soon as the report of the fifth committee on the programme budget implications is 
available. I now give the floor to delegations wishing to deliver explanations of vote or position before voting or 
adoption."""
tokens = tokenizer(paragraph)
torch.tensor(tokens["input_ids"]).unsqueeze(0).size()

model = AutoModelForTokenClassification.from_pretrained(
    pretrained_model, num_labels=len(LABEL_LIST)
)
predictions = model.forward(
    input_ids=torch.tensor(tokens["input_ids"]).unsqueeze(0),
    attention_mask=torch.tensor(tokens["attention_mask"]).unsqueeze(0),
)
predictions = torch.argmax(predictions.logits.squeeze(), axis=1)
predictions = [LABEL_LIST[i] for i in predictions]

words = tokenizer.batch_decode(tokens["input_ids"])
pd.DataFrame({"ner": predictions, "words": words}).to_csv("un_ner_bert.csv")
