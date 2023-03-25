import numpy as np
from numpy import typing as npt

from utils import Parameters


def layer_norm(params: Parameters, x: npt.NDArray, eps: float = 1e-5) -> npt.NDArray:
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    x = (x - mean) / np.sqrt(variance + eps)
    return params.weight.np * x + params.bias.np


def linear(
    param: Parameters,
    x: npt.NDArray,
):
    return np.matmul(x, param.weight.np.T) + param.bias.np


def softmax(x: npt.NDArray):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def gelu(x: npt.NDArray):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def mean_pooling(x: npt.NDArray, attention_mask: npt.NDArray):
    input_mask_expanded = attention_mask[:, :, None].astype(np.float32)
    return np.sum(x * input_mask_expanded, 1) / np.sum(input_mask_expanded, 1)
