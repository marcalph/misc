## wordpiece
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForSequenceClassification
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
pt_model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


sequences = ["I love you so much I will not trade you for a Titan GTX 24 Gio of VRAM", "i hate you"]
pt_batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
print(pt_batch)
print(tokenizer.tokenize(sequences[0]))

for key, value in pt_batch.items():
    print(f"{key}: {value.numpy().tolist()}")

pt_outputs = pt_model(**pt_batch)

tokenizer.decode(pt_batch['input_ids'][0])

print(pt_outputs)

pt_predictions = F.softmax(pt_outputs[0], dim=-1)

print(pt_predictions)

pt_outputs = pt_model(**pt_batch, labels = torch.tensor([1,0]))

pt_outputs = pt_model(**pt_batch, output_hidden_states=True, output_attentions=True)
all_hidden_states, all_attentions = pt_outputs[-2:]


from transformers import AutoModelWithLMHead, AutoTokenizer
import torch
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
model = AutoModelWithLMHead.from_pretrained("distilbert-base-cased")
sequence = f"Distilled models are smaller than the models they mimic. Using them instead of the large versions would  {tokenizer.mask_token} help limit our carbon footprint."
input = tokenizer.encode(sequence, return_tensors="pt")
print(input)
mask_token_index = torch.where(input == tokenizer.mask_token_id)[1]
print(mask_token_index)
token_logits = model(input)[0]
mask_token_logits = token_logits[0, mask_token_index, :]
top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

for token in top_5_tokens:
    print(sequence.replace(tokenizer.mask_token, tokenizer.decode([token])))


# for token in test_top_5_tokens:
#     print(sequence.replace("mimic", tokenizer.decode([token])))


from transformers import AutoModelWithLMHead, AutoTokenizer, top_k_top_p_filtering
import torch
from torch.nn import functional as F
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelWithLMHead.from_pretrained("gpt2")
sequence = f"Hugging Face is based in DUMBO, New York City, and "
input_ids = tokenizer.encode(sequence, return_tensors="pt")
# get logits of last hidden state
next_token_logits = model(input_ids)[0][:, -1, :]
first_token_logits = model(input_ids)[0][:,0,:]
# filter

filtered_next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=50, top_p=1.0)
filtered_first_token_logits = top_k_top_p_filtering(first_token_logits, top_k=50, top_p=1.0)

# sample
probs = F.softmax(filtered_next_token_logits, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)
tokenizer.decode(next_token)


first_probs = F.softmax(filtered_first_token_logits, dim=-1)
first_token = torch.multinomial(first_probs, num_samples=1)
tokenizer.decode(first_token)


generated = torch.cat([input_ids, next_token], dim=-1)
resulting_string = tokenizer.decode(generated.tolist()[0])
