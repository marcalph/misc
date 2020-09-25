## wordpiece


from transformers import AutoTokenizer, AutoModelForSequenceClassification
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
pt_model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


sequences = ["I love you so much", "i hate you"]
pt_batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
print(pt_batch)


for key, value in pt_batch.items():
    print(f"{key}: {value.numpy().tolist()}")

pt_outputs = pt_model(**pt_batch)

tokenizer.decode(pt_batch['input_ids'][1])

print(pt_outputs)

import torch.nn.functional as F
pt_predictions = F.softmax(pt_outputs[0], dim=-1)

print(pt_predictions)

import torch
pt_outputs = pt_model(**pt_batch, labels = torch.tensor([1,0]))


