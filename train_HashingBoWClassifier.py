"""
Train the submission-compatible Project B news headline classifier.

Usage examples:
    python train.py --csv scraped_headlines_clean_latest.csv
    python train.py --csv url_data_only.csv --epochs 80

Output:
    model.pt    # PyTorch state_dict loaded by the backend
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from preprocess import prepare_data
from model_HashingBoWClassifier import Model


SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_default_csv(base_dir: Path) -> Path:
    candidates = [
        base_dir / "scraped_headlines_clean_latest.csv",
        base_dir / "url_data_only.csv",
        base_dir / "data.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    csvs = sorted(base_dir.glob("*.csv"))
    if csvs:
        return csvs[0]
    raise FileNotFoundError(
        "No CSV file found. Please run: python train.py --csv path/to/your_data.csv"
    )


def make_split_indices(y: torch.Tensor, test_size: float, seed: int):
    indices = np.arange(len(y))
    y_np = y.cpu().numpy()

    # Stratification only works when every class has at least 2 examples.
    unique, counts = np.unique(y_np, return_counts=True)
    stratify = y_np if len(unique) > 1 and np.all(counts >= 2) else None

    train_idx, val_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )
    return torch.tensor(train_idx, dtype=torch.long), torch.tensor(val_idx, dtype=torch.long)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None, help="Path to training CSV")
    parser.add_argument("--out", type=str, default="model.pt", help="Output state_dict path")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    set_seed(SEED)

    base_dir = Path(__file__).resolve().parent
    csv_path = Path(args.csv) if args.csv else find_default_csv(base_dir)
    out_path = Path(args.out)

    print(f"Loading data from: {csv_path}")
    X, y = prepare_data(str(csv_path))
    y = torch.as_tensor(y, dtype=torch.long)

    num_classes = int(y.max().item()) + 1
    if num_classes != 2:
        print(f"Warning: detected {num_classes} classes. Project B usually expects 2 classes.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Total examples: {len(X)}")
    print(f"Label counts: {torch.bincount(y).tolist()}")

    model = Model(num_classes=max(num_classes, 2)).to(device)

    # Precompute hashed features once; this makes training much faster.
    print("Featurizing headlines ...")
    with torch.no_grad():
        features = model.featurize(X).detach()

    train_idx, val_idx = make_split_indices(y, args.test_size, SEED)
    features = features.to(device)
    labels = y.to(device)
    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)

    # Class weights help when Fox/NBC counts are imbalanced.
    counts = torch.bincount(labels, minlength=model.num_classes).float()
    class_weights = counts.sum() / torch.clamp(counts, min=1.0)
    class_weights = class_weights / class_weights.mean()

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print(f"Train: {len(train_idx)} | Val: {len(val_idx)}")
    best_state = None
    best_val_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = train_idx[torch.randperm(len(train_idx), device=device)]
        total_loss = 0.0

        for start in range(0, len(perm), args.batch_size):
            idx = perm[start:start + args.batch_size]
            logits = model(features[idx])
            loss = criterion(logits, labels[idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(idx)

        model.eval()
        with torch.no_grad():
            val_logits = model(features[val_idx])
            val_preds = torch.argmax(val_logits, dim=1)
            val_acc = (val_preds == labels[val_idx]).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            avg_loss = total_loss / len(train_idx)
            print(f"Epoch {epoch:03d} | loss={avg_loss:.4f} | val_acc={val_acc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final validation report.
    model.eval()
    with torch.no_grad():
        final_preds = torch.argmax(model(features[val_idx]), dim=1).cpu().numpy()
    y_val = labels[val_idx].cpu().numpy()
    print(f"\nBest validation accuracy: {accuracy_score(y_val, final_preds):.4f}")
    print(classification_report(y_val, final_preds, zero_division=0))

    torch.save(model.state_dict(), out_path)
    print(f"\nSaved state_dict to: {out_path}")

    # Smoke test: verify the exact backend-style loading flow.
    reloaded = Model(num_classes=max(num_classes, 2))
    reloaded.load_state_dict(torch.load(out_path, map_location="cpu"))
    reloaded.eval()
    print("Smoke test predictions:", reloaded.predict(X[:5]))


if __name__ == "__main__":
    main()
