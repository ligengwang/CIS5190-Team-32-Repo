"""
CIS 4190/5190 Project B: News Headline Classifier
Model 2: RoBERTa-based classifier

RoBERTa is a stronger variant of BERT trained with more data and longer.

Run:
    python roberta.py
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Any, Iterable, List
from transformers import RobertaModel, RobertaTokenizerFast

PRETRAINED  = "roberta-base"
MAX_LEN     = 64
HIDDEN_DIM  = 768
DROPOUT     = 0.3
NUM_CLASSES = 2       # 0 = NBC, 1 = FoxNews
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RoBERTaModel(nn.Module):
    """
    RoBERTa encoder + two-layer classification head.
    Same interface as DistilBERT Model — drop-in replacement.

    Satisfies all leaderboard requirements:
      - Instantiable with no arguments
      - predict(batch) -> List[int]  (FoxNews=1, NBC=0)
      - eval() supported
      - Compatible with load_state_dict
    """

    def __init__(self) -> None:
        super().__init__()
        self.encoder   = RobertaModel.from_pretrained(PRETRAINED)
        self.tokenizer = RobertaTokenizerFast.from_pretrained(PRETRAINED)

        self.classifier = nn.Sequential(
            nn.LayerNorm(HIDDEN_DIM),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, 256),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(256, NUM_CLASSES),
        )

        self.to(DEVICE)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # RoBERTa returns (last_hidden_state, pooler_output)
        outputs   = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]   # [CLS] vector
        return self.classifier(cls_token)

    def _tokenize(self, texts: List[str]) -> dict:
        return self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=MAX_LEN, return_tensors="pt",
        )

    def eval(self) -> "RoBERTaModel":
        super().eval()
        return self

    def predict(self, batch: Iterable[Any]) -> List[int]:
        """
        Args:
            batch: iterable of cleaned headline strings
        Returns:
            list of int -- 1 = FoxNews, 0 = NBC
        """
        texts = list(batch)
        if not texts:
            return []

        self.eval()
        encoded        = self._tokenize(texts)
        input_ids      = encoded["input_ids"].to(DEVICE)
        attention_mask = encoded["attention_mask"].to(DEVICE)

        with torch.no_grad():
            preds = self.forward(input_ids, attention_mask).argmax(dim=-1).cpu().tolist()

        return preds


# alias so leaderboard evaluator can find it as "Model"
Model = RoBERTaModel


def get_model() -> RoBERTaModel:
    return RoBERTaModel()


if __name__ == "__main__":
    print(f"DEVICE: {DEVICE}")
    model = get_model()
    print(f"Model type: {type(model).__name__}")
    test = [
        "trump signs executive order on immigration",
        "nbc investigation into government fraud",
        "fox news exclusive border crisis report",
    ]
    preds = model.predict(test)
    labels = {0: "NBC", 1: "FoxNews"}
    print("\nSanity check (untrained weights):")
    for h, p in zip(test, preds):
        print(f"  [{labels[p]}] {h}")