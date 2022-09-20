import os
from typing import Optional, Tuple

import einops
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm, trange
from transformers import AutoConfig, AutoTokenizer

from transformer import Transformer


def load_txt(path: str) -> pd.DataFrame:
    return pd.read_table(path, names=["en", "ja", "license"])


def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
    train, test = train_test_split(df, test_size=1000)
    # train, valid = train_test_split(train, test_size=0.3)
    # return train, valid, test
    return train, test


def get_config(model_path: str):
    return AutoConfig.from_pretrained(model_path)


def get_tokenizer(model_path: str):
    return AutoTokenizer.from_pretrained(model_path)


def get_model(
    vocab_size: int,
    d_model: int = 256,
    n_heads: int = 4,
    num_layers: int = 4,
    hidden_size: Optional[int] = None,
    dropout: float = 0.1,
    max_length: int = 32,
) -> Transformer:
    model = Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        num_layers=num_layers,
        hidden_size=hidden_size,
        dropout=dropout,
        max_length=max_length,
    )
    return model


class MTDataset(data.Dataset):
    def __init__(self, df: pd.DataFrame) -> None:
        super(MTDataset, self).__init__()
        self.en = df["en"].tolist()
        self.ja = df["ja"].tolist()

    def __len__(self):
        return len(self.en)

    def __getitem__(self, index):
        src = "</s>" + self.en[index] + "</s>"
        tgt = "</s>" + self.ja[index] + "</s>"

        return src, tgt


class CollateFn:
    def __init__(self, tokenizer, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        src, tgt = zip(*batch)
        src = list(src)
        tgt = list(tgt)
        src = self.tokenizer.batch_encode_plus(
            src,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        tgt = self.tokenizer.batch_encode_plus(
            tgt,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return src, tgt


class MTModel(nn.Module):
    def __init__(
        self,
        transformer: Transformer,
        vocab_size: int,
        d_model: int,
    ) -> None:
        super(MTModel, self).__init__()
        self.transformer = transformer
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_mask, tgt_mask):
        out = self.transformer(src, tgt, src_mask, tgt_mask)
        out = self.linear(out)
        return out

    @torch.inference_mode()
    def greedy_search(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor,
        alpha: float = 0.001,
        eos_token: int = 0,
        pad_token: int = 60715,
    ):
        tgt = torch.zeros_like(src, device=src.device)
        tgt[:, 1:].fill_(pad_token)
        tgt_mask = torch.zeros_like(src_mask, device=src_mask.device)

        for i in trange(src.size(1) - 1, leave=False):
            tgt_mask[:, i] = 1
            pred = self(src, src_mask, tgt, tgt_mask)
            pred = pred[:, i + 1]
            logits = torch.softmax(pred, dim=-1)
            ids = logits.max(dim=-1)

            # threshold
            tgt[:, i + 1] = ids.indices.masked_fill(ids.values < alpha, pad_token)

        return tgt


def train(
    model_path: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    batch_size: int = 64,
    epoch: int = 10,
    device: Optional[str] = None,
):
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    config = get_config(model_path)
    tokenizer = get_tokenizer(model_path)
    transformer = get_model(config.vocab_size)
    model = MTModel(transformer, vocab_size=config.vocab_size, d_model=768)
    model = model.to(device)

    dataset = MTDataset(train)
    test_dataset = MTDataset(test)
    train_loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=CollateFn(tokenizer, max_length=32),
        drop_last=True,
        pin_memory=True,
    )
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        num_workers=2,
        collate_fn=CollateFn(tokenizer, max_length=32),
        pin_memory=True,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-7)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1000)

    for e in trange(epoch):
        step = 0
        total = 0.0
        for src, tgt in tqdm(train_loader, leave=False):
            step += 1
            outs = model(
                src["input_ids"].to(device),
                tgt["input_ids"].to(device),
                src["attention_mask"].to(device),
                einops.repeat(
                    Transformer.generate_square_subsequent_mask(128),
                    "t l -> b t l",
                    b=batch_size,
                ).to(device),
            )
            loss = criterion(
                outs[:, :-1, :].contiguous().view(-1, outs.size(-1)).to(device),
                tgt["input_ids"][:, 1:].contiguous().view(-1).to(device),
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            total += loss.item()

            if step % 200 == 0:
                print(f"Epoch {e} | Step {step} >> Loss {loss.item()}")
        print("=" * 30)
        print(f"Epoch {e} >> Loss {total / step}")
        print()
        print("Predict testset")
        preds = []
        srcs = []
        for src, tgt in tqdm(test_loader, leave=False):
            ids = model.greedy_search(
                src["input_ids"].to(device),
                src["attention_mask"].to(device),
                alpha=0,
            )
            text = tokenizer.batch_decode(ids.to("cpu"), skip_special_tokens=True)
            preds += text
            srcs += tokenizer.batch_decode(
                src["input_ids"].to("cpu"), skip_special_tokens=True
            )

        df = pd.DataFrame(data={"src": srcs, "tgt": preds})
        os.makedirs("results", exist_ok=True)
        df.to_csv(f"results/epoch_{e}.tsv", header=True, sep="\t", line_terminator="\n")


if __name__ == "__main__":
    df = load_txt("data/jpn.txt")
    traindf, testdf = split_data(df)
    train("Helsinki-NLP/opus-mt-ja-en", traindf, testdf, batch_size=128, epoch=30)
