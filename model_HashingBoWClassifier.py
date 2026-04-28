"""
CIS 4190/5190 Project B: News Headline Classifier
Submission-compatible model.py

Interface required by backend:
  - get_model() -> model_instance
  - Model can be instantiated without arguments
  - predict(batch) -> List[int]
  - model.pt can be loaded as a PyTorch state_dict

Model design:
  - Deterministic hashing bag-of-words / bigrams
  - Linear classifier trained with CrossEntropyLoss
  - No fitted tokenizer/vectorizer object is required at inference time
"""

from __future__ import annotations

import re
import zlib
from typing import Iterable, List, Sequence, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_FEATURES = 16384
NUM_CLASSES = 2
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


def _normalize_text(x: Any) -> str:
    """Convert one input item into a clean string."""
    if x is None:
        return ""
    return str(x).lower().strip()


def _stable_hash(text: str) -> int:
    """Python's built-in hash is randomized, so use crc32 for stable features."""
    return zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF


def _tokens(text: str) -> List[str]:
    return TOKEN_RE.findall(_normalize_text(text))


class Model(nn.Module):
    """Hashing n-gram linear classifier for news headline classification."""

    def __init__(self, num_features: int = NUM_FEATURES, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.linear = nn.Linear(num_features, num_classes)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def featurize(self, batch: Iterable[Any]) -> torch.Tensor:
        """
        Convert a batch of headline strings into normalized hashed n-gram features.

        Features include:
          - unigram token counts
          - adjacent bigram token counts
          - a few simple style/length features
        """
        texts = list(batch)
        X = torch.zeros((len(texts), self.num_features), dtype=torch.float32, device=self.device)

        for i, text in enumerate(texts):
            toks = _tokens(text)
            if not toks:
                continue

            # Unigrams
            for tok in toks:
                j = _stable_hash("u:" + tok) % self.num_features
                X[i, j] += 1.0

            # Bigrams help capture short headline phrases
            for a, b in zip(toks, toks[1:]):
                j = _stable_hash("b:" + a + "_" + b) % self.num_features
                X[i, j] += 1.0

            # Lightweight numeric/style signals, hashed into fixed slots
            style_features = {
                "len_bin": min(len(text) // 20, 10),
                "tok_bin": min(len(toks), 20),
                "has_question": int("?" in str(text)),
                "has_exclaim": int("!" in str(text)),
                "has_number": int(any(ch.isdigit() for ch in str(text))),
            }
            for name, value in style_features.items():
                j = _stable_hash(f"s:{name}:{value}") % self.num_features
                X[i, j] += 1.0

        # log transform + L2 normalization makes counts more stable
        X = torch.log1p(X)
        X = F.normalize(X, p=2, dim=1)
        return X

    def forward(self, batch: Union[torch.Tensor, Sequence[Any], Iterable[Any]]) -> torch.Tensor:
        """
        Return logits with shape [batch_size, num_classes].

        The backend may call model(batch) directly. If batch is already a tensor,
        it is treated as a precomputed feature matrix. Otherwise, it is treated as
        a batch of headline strings.
        """
        if torch.is_tensor(batch):
            X = batch.to(device=self.device, dtype=torch.float32)
        else:
            X = self.featurize(batch)
        return self.linear(X)

    def predict(self, batch: Iterable[Any]) -> List[int]:
        """Return integer class predictions as a Python list."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(batch)
            return torch.argmax(logits, dim=1).cpu().tolist()


# Alternative class name accepted by some project checkers
NewsClassifier = Model


def get_model() -> Model:
    """Factory function required by the backend."""
    return Model()
