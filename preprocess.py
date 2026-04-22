from pathlib import Path
from collections import Counter

import pandas as pd
import re
import torch
from typing import Tuple, List

# pip install nltk
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords", quiet=True)
nltk.download("wordnet",   quiet=True)
nltk.download("omw-1.4",   quiet=True)

# paths
BASE_DIR        = Path(__file__).resolve().parent 
INPUT_CSV       = BASE_DIR / "scraped_headlines.csv"
TRAIN_READY_CSV = BASE_DIR / "scraped_headlines_clean_latest.csv"

# NLP helpers (initialised once) 
STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

#   label encoding
#   FoxNews = 1,  NBC = 0
def encode_label(source: str) -> int:
    """Binary label: 1 for FoxNews, 0 for NBC. Returns -1 if unknown."""
    if pd.isna(source):
        return -1
    s = str(source).strip().lower()
    if "fox" in s:
        return 1
    if "nbc" in s:
        return 0
    return -1


#  repair  (fix mojibake/encoding artefacts)
def repair_headline(text) -> str:
    """
    Fix common encoding / mojibake issues.
    Example: don?€?t ->  don't
    """
    if pd.isna(text):
        return ""
    text = str(text).strip()

    # apostrophe inside a word: don?€?t -> don't
    text = re.sub(r"(?<=\w)\?€\?(?=\w)",        "'", text)
    # possessive before punctuation/space: players?€? -> players'
    text = re.sub(r"(?<=s)\?€\?(?=[\s,.:;!?])", "'", text)
    # remove any remaining broken token
    text = text.replace("?€?", " ")

    # normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# clean (HTML, URLs, symbols, special characters)
def clean_headline(text: str) -> str:
    """Remove HTML tags, URLs, expand symbols, strip non-alpha characters."""
    # HTML tags  e.g. <b>, <span class="...">
    text = re.sub(r"<[^>]+>", " ", text)
    # HTML entities  e.g. &amp;  &nbsp;
    text = re.sub(r"&[a-z]+;", " ", text)

    # URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Expand common symbols before stripping punctuation
    text = text.replace("&", " and ")
    text = text.replace("/", " ")
    text = text.replace("-", " ")
    text = text.replace("_", " ")

    # Keep only letters, digits, apostrophes, spaces
    text = re.sub(r"[^a-zA-Z0-9'\s]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


#  normalize  (lowercase, stopwords, lemmatization)
def normalize_headline(text: str,
                        remove_stopwords: bool = True,
                        lemmatize: bool = True) -> str:
    """Lowercase, optionally remove stopwords and lemmatize."""
    text = text.lower()

    # Remove apostrophes inside contractions: don't -> dont
    text = re.sub(r"(?<=\w)'(?=\w)", "", text)

    tokens = text.split()

    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS]

    if lemmatize:
        tokens = [LEMMATIZER.lemmatize(t) for t in tokens]

    return " ".join(tokens)


#  full pipeline  (repair-> clean-> normalize)
def full_pipeline(raw_text,
                  remove_stopwords: bool = True,
                  lemmatize: bool = True) -> str:
    text = repair_headline(raw_text)
    text = clean_headline(text)
    text = normalize_headline(text,
                               remove_stopwords=remove_stopwords,
                               lemmatize=lemmatize)
    return text

#  IMPORTANT: LEADERBOARD ENTRY POINT!!!
def prepare_data(path: str) -> Tuple[List[str], torch.Tensor]:
    df = pd.read_csv(path, encoding="utf-8-sig")

    if "scrape_status" in df.columns:
        df = df[df["scrape_status"] == "success"].copy()

    # find headline column
    if "headline" in df.columns:
        df["headline_raw"] = df["headline"].fillna("")
    elif "headline_raw" in df.columns:
        df["headline_raw"] = df["headline_raw"].fillna("")
    else:
        raise ValueError(f"No headline column found. Columns: {list(df.columns)}")

    # apply full cleaning pipeline
    df["headline_clean"] = df["headline_raw"].apply(full_pipeline)
    df = df[df["headline_clean"].str.strip() != ""].copy()

    # infer label from URL if no source/label column ──
    label_col = None
    for col in ["source", "label", "Source", "Label"]:
        if col in df.columns:
            label_col = col
            break

    if label_col is not None:
        df["label"] = df[label_col].apply(encode_label)
    elif "url" in df.columns:
        # infer from URL: foxnews.com 1, nbcnews.com 0
        def label_from_url(url: str) -> int:
            url = str(url).lower()
            if "fox" in url:
                return 1
            if "nbc" in url:
                return 0
            return -1
        df["label"] = df["url"].apply(label_from_url)
    else:
        raise ValueError(f"No label column found. Columns: {list(df.columns)}")

    df = df[df["label"] != -1].copy()

    X: List[str] = df["headline_clean"].tolist()
    y = torch.tensor(df["label"].tolist(), dtype=torch.long)

    return X, y

def main():
    # Read
    df = pd.read_csv(INPUT_CSV, encoding="cp1252")
    print(f"Rows loaded                : {len(df)}")
    print(f"Columns                    : {list(df.columns)}\n")

    clean = df.copy()

    # Drop fully empty columns (e.g. "Unnamed: 6")
    empty_cols = [c for c in clean.columns if clean[c].isna().all()]
    if empty_cols:
        clean = clean.drop(columns=empty_cols)
        print(f"Dropped empty columns      : {empty_cols}")

    # Keep only successful scrapes
    if "scrape_status" in clean.columns:
        clean = clean[clean["scrape_status"] == "success"].copy()
        print(f"After scrape_status filter : {len(clean)} rows")

    # Handle missing raw headlines
    missing = clean["headline_raw"].isna().sum()
    print(f"Missing headlines          : {missing} (dropped)")
    clean = clean[clean["headline_raw"].notna()].copy()
    clean["headline_raw"] = clean["headline_raw"].astype(str)

    # Apply full pipeline 
    clean["headline_repaired"] = clean["headline_raw"].apply(repair_headline)
    clean["headline_clean"]    = clean["headline_raw"].apply(full_pipeline)

    # Drop empty cleaned headlines
    before = len(clean)
    clean = clean[clean["headline_clean"].str.strip() != ""].copy()
    print(f"Dropped after cleaning     : {before - len(clean)} rows")

    # Encode binary labels (FoxNews=1, NBC=0)
    clean["label"] = clean["source"].apply(encode_label)

    # Drop rows with unknown source
    before = len(clean)
    clean = clean[clean["label"] != -1].copy()
    print(f"Dropped unknown source     : {before - len(clean)} rows")

    # Class distribution
    fox_count = (clean["label"] == 1).sum()
    nbc_count = (clean["label"] == 0).sum()
    print(f"\nClass distribution:")
    print(f"  FoxNews (1)              : {fox_count}")
    print(f"  NBC     (0)              : {nbc_count}")

    # QA metrics 
    clean["char_count"] = clean["headline_repaired"].str.len()
    clean["word_count"] = clean["headline_clean"].str.split().str.len()

    print(f"\n── Word count stats ───────────────────────────────")
    print(clean["word_count"].describe().to_string())

    # Spot-check: 5 random rows
    print(f"\n── Random sample (raw -> clean) ────────────────────")
    sample = clean[["source", "label", "headline_raw", "headline_clean"]].sample(
        n=min(5, len(clean)), random_state=42
    )
    for _, row in sample.iterrows():
        print(f"  SOURCE : {row['source']}  (label={row['label']})")
        print(f"  RAW    : {row['headline_raw']}")
        print(f"  CLEAN  : {row['headline_clean']}")
        print()

    # word frequency distribution (top 20)
    all_words = " ".join(clean["headline_clean"]).split()
    freq = Counter(all_words).most_common(20)
    print("── Top 20 tokens ──────────────────────────────────")
    for word, count in freq:
        print(f"  {word:<25} {count}")

    # deduplicate 
    before_dedup = len(clean)
    train_ready = clean.drop_duplicates(subset=["source", "headline_clean"]).copy()
    print(f"\nDuplicates removed         : {before_dedup - len(train_ready)}")
    print(f"Final training rows        : {len(train_ready)}")

    # save
    out_cols = ["url", "source", "headline_raw", "headline_repaired",
                "headline_clean", "article_date", "label",
                "char_count", "word_count"]
    out_cols = [c for c in out_cols if c in train_ready.columns]
    train_ready[out_cols].to_csv(TRAIN_READY_CSV, index=False, encoding="utf-8-sig")

    print(f"\nSaved: {TRAIN_READY_CSV.name}")


if __name__ == "__main__":
    main()
