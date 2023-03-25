from transformers import BertModel, BertTokenizer

from bert import BertConfig, bert_model
from common import mean_pooling
from utils import Parameters

tokenizer = BertTokenizer.from_pretrained(
    "sentence-transformers/paraphrase-TinyBERT-L6-v2"
)
model = BertModel.from_pretrained("sentence-transformers/paraphrase-TinyBERT-L6-v2")

text = ["How many people live in London?", "test"]
tokenized_text = tokenizer(text, padding=True, truncation=True, return_tensors="np")


input_ids = tokenized_text["input_ids"]
token_type_ids = tokenized_text["token_type_ids"]
attention_mask = tokenized_text["attention_mask"]


# weights
config_dict = model.config.to_dict()  # type: ignore
filtered_config_dict = {
    k: v for k, v in config_dict.items() if k in BertConfig.__annotations__
}
config = BertConfig(**filtered_config_dict)
params = Parameters.from_model(model)

output_hidden_state = bert_model(
    config,
    params,
    input_ids,
    token_type_ids,
    attention_mask,
)
sentence_embedding = mean_pooling(output_hidden_state, attention_mask)
print(sentence_embedding)
