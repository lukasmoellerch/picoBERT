# PicoBERT

A BERT implementation inspired by [picoGPT](https://github.com/jaymody/picoGPT).

`picoBERT` is a tiny implementation of [BERT](https://arxiv.org/pdf/1810.04805.pdf) for educational purposes. It is a NumPy implementation of BERT, and it is not optimized for speed. It is intended to be a simple and easy-to-understand implementation of BERT. It is not intended to be used for production purposes.

A quick breakdown of the code is as follows:

- `bert.py` contains the implementation of BERT.
- `common.py` contains some common core functions such as layer normalization, linear layers etc.
- `example.py` loads a model using huggingface transformers library and runs it on a sample input.
- `reference.py` is a file using the huggingface transformers library to verify the correctness of the implementation.
- `utils.py` contains utility functions for accessing weights in a structured manner.

## Dependencies

```sh
pip install -r requirements.txt
```
