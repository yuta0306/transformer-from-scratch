import os
from typing import Optional, Tuple

import einops
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm, trange
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
    tgt_vocab_size: Optional[int] = None,
    d_model: int = 256,
    n_heads: int = 8,
    num_layers: int = 4,
    hidden_size: Optional[int] = 512,
    dropout: float = 0.1,
    max_length: int = 32,
) -> Transformer:
    model = Transformer(
        vocab_size=vocab_size,
        tgt_vocab_size=tgt_vocab_size,
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
        src = self.en[index]
        tgt = self.ja[index]

        return src, tgt


class CollateFn:
    def __init__(self, tokenizer, max_length: int, tgt_tokenizer=None) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tgt_tokenizer = tgt_tokenizer

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
        if self.tgt_tokenizer is None:
            tgt = self.tokenizer.batch_encode_plus(
                tgt,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
        else:
            tgt = self.tgt_tokenizer.batch_encode_plus(
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
        pad_token: int = 32000,
    ):
        tgt = torch.zeros_like(src, device=src.device)
        tgt[:, 1:].fill_(pad_token)
        tgt_mask = torch.zeros_like(src_mask, device=src_mask.device)

        for i in trange(src.size(1) - 1, leave=False):
            tgt_mask[:, i] = 1
            # print(tgt_mask[0])
            # print(tgt[0])
            pred = self(src, src_mask, tgt, tgt_mask)
            pred = pred[:, i]
            # logits = torch.softmax(pred, dim=-1)
            ids = pred.max(dim=-1)
            print(torch.topk(pred, 5, dim=-1).indices)

            # threshold
            tgt[:, i + 1] = ids.indices.masked_fill(
                torch.sigmoid(ids.values) / 2 + 0.5 < alpha, pad_token
            )

        return tgt


class CosineAnnealingLR(optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: optim.Optimizer,
        T_0: int,
        T_warmup: int = 0,
        T_multi: int = 1,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        assert T_0 >= 0 and isinstance(T_0, int)
        assert T_multi > 0 and isinstance(T_multi, int)

        self.T_0 = T_0
        self.T_warmup = T_warmup
        self.T_multi = T_multi
        self.eta_min = eta_min
        self.cos_anneal = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=T_multi, eta_min=eta_min, last_epoch=last_epoch
        )
        super(CosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.T_warmup:
            return [
                self.eta_min
                + (base_lr - self.eta_min) * (self.last_epoch / self.T_warmup)
                for base_lr in self.base_lrs
            ]

        self.cos_anneal.step()
        return self.cos_anneal.get_lr()


def train(
    model_path: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    tgt_model_path: Optional[str] = None,
    batch_size: int = 64,
    epoch: int = 10,
    device: Optional[str] = None,
):
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    config = get_config(model_path)
    tokenizer = get_tokenizer(model_path)
    tgt_vocab_size = None
    tgt_tokenizer = None
    if tgt_model_path is not None:
        tgt_config = get_config(tgt_model_path)
        tgt_tokenizer = get_tokenizer(tgt_model_path)
        tgt_vocab_size = tgt_config.vocab_size
    transformer = get_model(config.vocab_size, tgt_vocab_size=tgt_vocab_size)
    vocab_size = tgt_vocab_size if tgt_vocab_size is not None else config.vocab_size
    model = MTModel(transformer, vocab_size=vocab_size, d_model=256)
    model = model.to(device)

    dataset = MTDataset(train)
    test_dataset = MTDataset(test)
    train_loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
        collate_fn=CollateFn(tokenizer, max_length=32, tgt_tokenizer=tgt_tokenizer),
        drop_last=True,
        pin_memory=True,
    )
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        num_workers=os.cpu_count(),
        collate_fn=CollateFn(tokenizer, max_length=32, tgt_tokenizer=tgt_tokenizer),
        pin_memory=True,
    )

    criterion = nn.CrossEntropyLoss(ignore_index=32000)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    # lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1000)
    lr_scheduler = CosineAnnealingLR(
        optimizer, T_0=10000, T_warmup=500, T_multi=1, eta_min=1e-5
    )

    for e in trange(epoch):
        step = 0
        total = 0.0
        model.train()
        for src, tgt in tqdm(train_loader, leave=False):
            step += 1
            outs = model(
                src["input_ids"].to(device),
                tgt["input_ids"].to(device),
                src["attention_mask"].to(device),
                einops.repeat(
                    Transformer.generate_square_subsequent_mask(32),
                    "t l -> b t l",
                    b=batch_size,
                ).to(device),
            )

            loss = criterion(
                outs[:, :-1].contiguous().view(-1, outs.size(-1)).to(device),
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
        print(f"Epoch {e} >> Loss {total / max(1, step)}")
        print()
        print("Predict testset")
        preds = []
        srcs = []
        model.eval()
        for src, tgt in tqdm(test_loader, leave=False):
            ids = model.greedy_search(
                src["input_ids"].to(device),
                src["attention_mask"].to(device),
                alpha=0,
            )
            if tgt_tokenizer is None:
                text = tokenizer.batch_decode(ids.to("cpu"), skip_special_tokens=True)
            else:
                text = tgt_tokenizer.batch_decode(
                    ids.to("cpu"), skip_special_tokens=True
                )
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
    train(
        "bert-base-uncased",
        traindf,
        testdf,
        batch_size=128,
        epoch=50,
        tgt_model_path="cl-tohoku/bert-base-japanese",
    )
