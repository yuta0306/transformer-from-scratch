import torch
from transformer import (
    CrossAttention,
    MultiHeadCrossAttention,
    MultiHeadSelfAttention,
    SelfAttention,
)


def test_self_attention():
    model = SelfAttention(768)
    inp = torch.randn((2, 128, 768))

    out = model(inp)

    assert out["outputs"].size() == (2, 128, 768)


def test_self_attention_with_mask():
    idx = 120
    model = SelfAttention(768)
    inp = torch.randn((2, 128, 768))
    mask = torch.zeros((2, 128), dtype=torch.long)
    mask[:, :idx] = 1

    out = model(inp, mask=mask)

    assert out["outputs"].size() == (2, 128, 768), out.size()
    assert (out["attention_weights"][:, :, idx:] == 0).all(), out["attention_weights"]


def test_multi_head_self_attention():
    model = MultiHeadSelfAttention(8, 768)
    inp = torch.randn((2, 128, 768))

    out = model(inp)

    assert out["outputs"].size() == (2, 128, 768)


def test_multi_head_self_attention_with_mask():
    idx = 120
    model = MultiHeadSelfAttention(8, 768)
    inp = torch.randn((2, 128, 768))
    mask = torch.zeros((2, 128), dtype=torch.long)
    mask[:, :idx] = 1

    out = model(inp, mask=mask)

    assert out["outputs"].size() == (2, 128, 768), out.size()
    assert (out["attention_weights"][:, :, :, idx:] == 0).all(), out[
        "attention_weights"
    ]


def test_cross_attention():
    model = CrossAttention(768, 1024, hidden_size=768)
    inp = torch.randn((2, 128, 768))
    context = torch.randn((2, 1, 1024))

    out = model(inp, context)

    assert out["outputs"].size() == (2, 128, 768)


def test_cross_attention_with_mask():
    idx = 120
    model = CrossAttention(768, 1024, hidden_size=768)
    inp = torch.randn((2, 128, 768))
    context = torch.randn((2, 2, 1024))
    mask = torch.zeros((2, 128), dtype=torch.long)
    mask[:, :idx] = 1

    out = model(inp, context, mask=mask)

    assert out["outputs"].size() == (2, 128, 768), out.size()
    assert (out["attention_weights"][:, :, idx:] == 0).all(), out["attention_weights"]


def test_multi_head_cross_attention():
    model = MultiHeadCrossAttention(8, 768, 1024, hidden_size=768)
    inp = torch.randn((2, 128, 768))
    context = torch.randn((2, 1, 1024))

    out = model(inp, context)

    assert out["outputs"].size() == (2, 128, 768)


def test_multi_head_cross_attention_with_mask():
    idx = 120
    model = MultiHeadCrossAttention(8, 768, 1024, hidden_size=768)
    inp = torch.randn((2, 128, 768))
    context = torch.randn((2, 1, 1024))
    mask = torch.zeros((2, 128), dtype=torch.long)
    mask[:, :idx] = 1

    out = model(inp, context, mask=mask)

    assert out["outputs"].size() == (2, 128, 768), out.size()
    assert (out["attention_weights"][:, :, :, idx:] == 0).all(), out[
        "attention_weights"
    ]
