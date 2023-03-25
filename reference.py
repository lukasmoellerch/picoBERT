import torch
from transformers import AutoModel, AutoTokenizer


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/paraphrase-TinyBERT-L6-v2"
)
model = AutoModel.from_pretrained("sentence-transformers/paraphrase-TinyBERT-L6-v2")

text = ["How many people live in London?", "test"]

tokenized_text = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
    output_hidden_state = model(**tokenized_text)
sentence_embeddings = mean_pooling(
    output_hidden_state, tokenized_text["attention_mask"]
)

print(sentence_embeddings)
