"""
CIS 4190/5190 Project B: News Headline Classifier
Training script for RoBERTa model

Run:
    python train_roberta.py

Saves: model_roberta.pt
"""

from __future__ import annotations

from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from roberta import RoBERTaModel, DEVICE
from preprocess import prepare_data

CSV_PATH    = Path("scraped_headlines_clean_latest.csv")
WEIGHTS_OUT = Path("model_roberta.pt")

BATCH_SIZE   = 16      # RoBERTa is larger, use smaller batch
EPOCHS       = 5
LR_ENCODER   = 1e-5   # slightly lower than DistilBERT due to larger model
LR_HEAD      = 1e-4
WEIGHT_DECAY = 1e-2
TEST_SIZE    = 0.20
SEED         = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


class HeadlineDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts  = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


def make_collate(model):
    def collate(batch):
        texts, labels = zip(*batch)
        enc = model._tokenize(list(texts))
        return enc["input_ids"], enc["attention_mask"], torch.stack(list(labels))
    return collate


def main():
    print(f"Loading: {CSV_PATH}")
    X, y = prepare_data(str(CSV_PATH))
    print(f"Total rows: {len(X)}  |  FoxNews: {y.sum().item()}  |  NBC: {(y==0).sum().item()}\n")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y.tolist(), test_size=TEST_SIZE, random_state=SEED, stratify=y.tolist()
    )
    y_tr  = torch.tensor(y_tr,  dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)
    print(f"Train: {len(X_tr)}  |  Val: {len(X_val)}\n")

    model   = RoBERTaModel()
    collate = make_collate(model)

    train_loader = DataLoader(HeadlineDataset(X_tr,  y_tr),  batch_size=BATCH_SIZE,   shuffle=True,  collate_fn=collate)
    val_loader   = DataLoader(HeadlineDataset(X_val, y_val), batch_size=BATCH_SIZE*2, shuffle=False, collate_fn=collate)

    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(),    "lr": LR_ENCODER},
        {"params": model.classifier.parameters(), "lr": LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1,
        total_iters=len(train_loader) * EPOCHS
    )

    criterion = nn.CrossEntropyLoss()
    best_acc  = 0.0

    for epoch in range(1, EPOCHS + 1):
        # train
        model.train()
        total_loss = 0.0
        for input_ids, attn_mask, labels in train_loader:
            input_ids, attn_mask, labels = (
                input_ids.to(DEVICE), attn_mask.to(DEVICE), labels.to(DEVICE)
            )
            optimizer.zero_grad()
            loss = criterion(model(input_ids, attn_mask), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        # validate
        model.eval()
        preds_all, labels_all = [], []
        with torch.no_grad():
            for input_ids, attn_mask, labels in val_loader:
                logits = model(input_ids.to(DEVICE), attn_mask.to(DEVICE))
                preds_all.extend(logits.argmax(-1).cpu().tolist())
                labels_all.extend(labels.tolist())

        val_acc = accuracy_score(labels_all, preds_all)
        print(f"Epoch {epoch}/{EPOCHS}  loss={total_loss/len(train_loader):.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), WEIGHTS_OUT)
            print(f"  Saved best model to {WEIGHTS_OUT}  (acc={best_acc:.4f})")

    # final report
    print(f"\nBest val accuracy: {best_acc:.4f}")
    model.load_state_dict(torch.load(WEIGHTS_OUT, map_location=DEVICE))
    model.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for input_ids, attn_mask, labels in val_loader:
            logits = model(input_ids.to(DEVICE), attn_mask.to(DEVICE))
            preds_all.extend(logits.argmax(-1).cpu().tolist())
            labels_all.extend(labels.tolist())

    print("\n── Classification Report ─────────────────────────────────────────")
    print(classification_report(labels_all, preds_all, target_names=["NBC (0)", "FoxNews (1)"]))
    print(f"\nIf RoBERTa beats DistilBERT:")
    print(f"  1. cp model_roberta.py model.py")
    print(f"  2. cp model_roberta.pt model.pt")
    print(f"  3. Submit model.py + preprocess.py + model.pt")


if __name__ == "__main__":
    main()