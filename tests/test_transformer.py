from dataclasses import dataclass

import einops
import torch
from transformer import Transformer
from transformers import AutoTokenizer

vocab_size = 3000
d_model = 768
num_layers = 4
n_heads = 6


@dataclass
class BertConfig:
    vocab_size: int = 30522
    d_model: int = 768
    hidden_size: int = 3072
    max_length: int = 512
    dropout: float = 0.1
    n_heads: int = 12
    num_layers: int = 12


def build():
    model = Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        n_heads=n_heads,
    )
    return model


def test_build_transformer():
    model = build()
    print(model)


def test_transformer():
    model = build()
    src = torch.randint(0, 3000, (4, 128))
    tgt = torch.randint(0, 3000, (4, 128))
    out = model(src, tgt)
    assert out.size() == (4, 128, 768)


def test_transformer_decode():
    model = build()
    src = torch.randint(0, 3000, (4, 128))
    tgt = torch.zeros_like(src)
    tgt_mask = model.generate_square_subsequent_mask(tgt.size(1))
    tgt_mask = einops.repeat(tgt_mask, "i j -> b i j", b=tgt.size(0))
    out = model(src, tgt, tgt_mask=tgt_mask)
    assert out.size() == (4, 128, 768)


def test_transformer_with_bert_conf():
    config = BertConfig()
    model = Transformer(
        config.vocab_size,
        config.d_model,
        config.n_heads,
        config.num_layers,
        config.hidden_size,
        config.dropout,
        config.max_length,
    )
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    texts = ["I love the kids."]
    inputs = tokenizer.batch_encode_plus(
        texts,
        padding="max_length",
        truncation=True,
        max_length=config.max_length,
        return_tensors="pt,",
    )
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    out = model(input_ids)
