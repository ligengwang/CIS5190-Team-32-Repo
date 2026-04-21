"""
CIS 4190/5190 Project B: News Headline Classifier
Submission model.py

Contains:
  - Model: DistilBERT-based classifier
  - get_model(): factory function required by evaluator
  - FallbackModel: TF-IDF + Logistic Regression 

Submission files:
  model.py      <- this file
  preprocess.py <- existing pipeline
  model.pt      <- weights from train.py

Labels: FoxNews = 1, NBC = 0
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Any, Iterable, List

# constants
PRETRAINED  = "distilbert-base-uncased"
MAX_LEN     = 64
HIDDEN_DIM  = 768
DROPOUT     = 0.3
NUM_CLASSES = 2       # 0 = NBC, 1 = FoxNews
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# check if transformers is available
try:
    from transformers import DistilBertModel, DistilBertTokenizerFast
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# primary model: DistilBERT
class Model(nn.Module):
    """
    DistilBERT encoder + two-layer classification head.

    Satisfies all leaderboard requirements:
      - Instantiable with no arguments
      - predict(batch) -> List[int]  (FoxNews=1, NBC=0)
      - eval() supported
      - Compatible with load_state_dict
    """

    def __init__(self) -> None:
        super().__init__()

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers is required for DistilBERT. "
                "Use get_model() which falls back to TF-IDF if unavailable."
            )

        self.encoder   = DistilBertModel.from_pretrained(PRETRAINED)
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(PRETRAINED)

        self.classifier = nn.Sequential(
            nn.LayerNorm(HIDDEN_DIM),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, 256),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(256, NUM_CLASSES),
        )

        self.to(DEVICE)

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs   = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_token)

    def _tokenize(self, texts: List[str]) -> dict:
        return self.tokenizer(
            texts,
            padding        = True,
            truncation     = True,
            max_length     = MAX_LEN,
            return_tensors = "pt",
        )

    def eval(self) -> "Model":
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
            logits = self.forward(input_ids, attention_mask)
            preds  = logits.argmax(dim=-1).cpu().tolist()

        return preds


# fallback model: TF-IDF + Logistic Regression
class FallbackModel:
    """
    TF-IDF + Logistic Regression fallback.
    Used automatically by get_model() if transformers is not available.
    Same predict(batch) interface as Model. Accuracy: ~67%.
    """

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        self.clf        = LogisticRegression(max_iter=1000, C=1.0)
        self._fitted    = False

    def fit(self, X: List[str], y: List[int]) -> None:
        X_tfidf = self.vectorizer.fit_transform(X)
        self.clf.fit(X_tfidf, y)
        self._fitted = True

    def eval(self) -> "FallbackModel":
        return self

    def predict(self, batch: Iterable[Any]) -> List[int]:
        texts = list(batch)
        if not texts:
            return []
        if not self._fitted:
            raise RuntimeError("FallbackModel must be fitted before predict.")
        X_tfidf = self.vectorizer.transform(texts)
        return self.clf.predict(X_tfidf).tolist()


# factory function required by evaluator
def get_model():
    """
    Returns DistilBERT Model if transformers is available,
    otherwise returns FallbackModel with a warning.
    """
    if TRANSFORMERS_AVAILABLE:
        return Model()
    else:
        print(
            "[WARNING] transformers not available. "
            "Falling back to TF-IDF + Logistic Regression (~67% accuracy)."
        )
        return FallbackModel()


# local sanity check
if __name__ == "__main__":
    print(f"transformers available : {TRANSFORMERS_AVAILABLE}")
    print(f"DEVICE                 : {DEVICE}")

    model = get_model()
    print(f"Model type             : {type(model).__name__}")

    test_headlines = [
        "trump signs executive order on immigration",
        "climate change threatens coastal communities",
        "nbc exclusive investigation into government fraud",
    ]

    if TRANSFORMERS_AVAILABLE:
        preds = model.predict(test_headlines)
        labels = {0: "NBC", 1: "FoxNews"}
        print("\nSanity check predictions (untrained weights):")
        for headline, pred in zip(test_headlines, preds):
            print(f"  [{labels[pred]}] {headline}")
    else:
        print("\nFallbackModel requires fit() before predict() -- skipping smoke test.")
