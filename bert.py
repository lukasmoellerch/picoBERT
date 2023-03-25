from dataclasses import dataclass

import numpy as np
from numpy import typing as npt

from common import gelu, layer_norm, linear, softmax
from utils import Parameters


@dataclass
class BertConfig:
    attention_probs_dropout_prob: float
    gradient_checkpointing: bool
    hidden_act: str
    hidden_dropout_prob: float
    hidden_size: int
    initializer_range: float
    intermediate_size: int
    layer_norm_eps: float
    max_position_embeddings: int
    model_type: str
    num_attention_heads: int
    num_hidden_layers: int
    pad_token_id: int
    position_embedding_type: str
    transformers_version: str
    type_vocab_size: int
    use_cache: int
    vocab_size: int


def bert_embedding_lookup(
    config: BertConfig,
    params: Parameters,
    input_ids: npt.NDArray,
    token_type_ids: npt.NDArray,
) -> npt.NDArray:
    embeds = (
        params.word_embeddings.weight.np[input_ids]
        + params.token_type_embeddings.weight.np[token_type_ids]
    )
    if config.position_embedding_type == "absolute":
        seq_length = input_ids.shape[1]
        position_ids = np.arange(seq_length, dtype=np.int64)[None, :]
        embeds += params.position_embeddings.weight.np[position_ids]
    embeds = layer_norm(params.LayerNorm, embeds)
    return embeds


def transpose_for_scores(
    x: npt.NDArray,
    num_attention_heads: int,
    head_size: int,
) -> npt.NDArray:
    new_x_shape = x.shape[:-1] + (num_attention_heads, head_size)
    x = np.reshape(x, new_x_shape)
    return np.transpose(x, [0, 2, 1, 3])


def bert_self_attention(
    config: BertConfig,
    params: Parameters,
    hidden_states: npt.NDArray,
    attention_mask: npt.NDArray,
) -> npt.NDArray:
    num_attention_heads = config.num_attention_heads
    attention_head_size = config.hidden_size // num_attention_heads
    all_head_size = num_attention_heads * attention_head_size

    query, key, value = (
        transpose_for_scores(
            linear(params.query, hidden_states),
            num_attention_heads,
            attention_head_size,
        ),
        transpose_for_scores(
            linear(params.key, hidden_states),
            num_attention_heads,
            attention_head_size,
        ),
        transpose_for_scores(
            linear(params.value, hidden_states),
            num_attention_heads,
            attention_head_size,
        ),
    )

    attention_scores = np.matmul(query, key.transpose([0, 1, 3, 2]))
    attention_scores = attention_scores / np.sqrt(attention_head_size)
    attention_scores = attention_scores + attention_mask

    attention_probs = softmax(attention_scores)

    context_layer = np.matmul(attention_probs, value)
    context_layer = np.transpose(context_layer, [0, 2, 1, 3])
    new_context_layer_shape = context_layer.shape[:-2] + (all_head_size,)
    context_layer = np.reshape(context_layer, new_context_layer_shape)

    return context_layer


def bert_encoder_layer(
    config: BertConfig,
    params: Parameters,
    hidden_states: npt.NDArray,
    attention_mask: npt.NDArray,
) -> npt.NDArray:
    input_tensor = bert_self_attention(
        config,
        params.attention.self,
        hidden_states,
        attention_mask,
    )
    # output
    attention_output_state = linear(params.attention.output.dense, input_tensor)
    hidden_states = layer_norm(
        params.attention.output.LayerNorm, attention_output_state + hidden_states
    )

    # intermediate
    intermediate_output = linear(params.intermediate.dense, hidden_states)
    intermediate_output = gelu(intermediate_output)

    # output
    output_hidden_states = linear(params.output.dense, intermediate_output)
    output_hidden_states = layer_norm(
        params.output.LayerNorm, output_hidden_states + hidden_states
    )

    return output_hidden_states


def bert_encoder(
    config: BertConfig,
    params: Parameters,
    hidden_states: npt.NDArray,
    attention_mask: npt.NDArray,
) -> npt.NDArray:
    current = hidden_states
    for i in range(config.num_hidden_layers):
        layer = params.layer[i]
        current = bert_encoder_layer(config, layer, current, attention_mask)
    return current


def bert_model(
    config: BertConfig,
    params: Parameters,
    input_ids: npt.NDArray,
    token_type_ids: npt.NDArray,
    attention_mask: npt.NDArray,
) -> npt.NDArray:
    input_hidden_state = bert_embedding_lookup(
        config, params.embeddings, input_ids, token_type_ids
    )
    # [batch_size, num_heads, seq_length, seq_length]
    extended_attention_mask = (1 - attention_mask[:, None, None, :]) * np.float32(-1e9)
    return bert_encoder(
        config, params.encoder, input_hidden_state, extended_attention_mask
    )
