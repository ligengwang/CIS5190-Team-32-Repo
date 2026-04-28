from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

CSV_PATH = Path('scraped_headlines_clean_latest.csv')
SEED = 42
TEST_SIZE = 0.20
RESULTS_CSV = Path('experiment_results.csv')
PLOT_PATH = Path('experiment_metrics.png')
TOP_FEATURES_JSON = Path('top_features_word_logreg.json')


def evaluate(y_true, y_pred):
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=['FoxNews', 'NBC'])
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'fox_as_fox': int(cm[0, 0]),
        'fox_as_nbc': int(cm[0, 1]),
        'nbc_as_fox': int(cm[1, 0]),
        'nbc_as_nbc': int(cm[1, 1]),
    }


def main():
    news_df = pd.read_csv(CSV_PATH, encoding='utf-8-sig')
    X = news_df['headline_clean'].astype(str)
    y = news_df['source'].astype(str)

    mask = X.notna() & y.notna() & (X.str.strip() != '')
    X = X[mask]
    y = y[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )

    experiments = {
        'baseline_tfidf_logreg_100': Pipeline([
            ('vec', TfidfVectorizer(stop_words='english', max_features=100)),
            ('clf', LogisticRegression(max_iter=100, random_state=SEED)),
        ]),
        'tfidf_complement_nb_uni_bigram': Pipeline([
            ('vec', TfidfVectorizer(stop_words='english', max_features=20000, ngram_range=(1, 2), min_df=2)),
            ('clf', ComplementNB(alpha=0.5)),
        ]),
        'tfidf_linear_svc_uni_bigram': Pipeline([
            ('vec', TfidfVectorizer(stop_words='english', max_features=20000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ('clf', LinearSVC(C=1.0, random_state=SEED)),
        ]),
        'tfidf_logreg_unigram_20k': Pipeline([
            ('vec', TfidfVectorizer(stop_words='english', max_features=20000, ngram_range=(1, 1), min_df=2, sublinear_tf=True)),
            ('clf', LogisticRegression(max_iter=3000, random_state=SEED, C=2.0)),
        ]),
        'tfidf_logreg_uni_bigram_20k': Pipeline([
            ('vec', TfidfVectorizer(stop_words='english', max_features=20000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ('clf', LogisticRegression(max_iter=3000, random_state=SEED, C=4.0)),
        ]),
        'charwb_logreg_3_5': Pipeline([
            ('vec', TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2, sublinear_tf=True)),
            ('clf', LogisticRegression(max_iter=3000, random_state=SEED, C=2.0)),
        ]),
    }

    rows = []
    fitted = {}
    for name, model in experiments.items():
        model.fit(X_train, y_train)
        fitted[name] = model
        pred = model.predict(X_test)
        metrics = evaluate(y_test, pred)
        rows.append({'model': name, **metrics})

    results_df = pd.DataFrame(rows).sort_values('accuracy')
    results_df.to_csv(RESULTS_CSV, index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(results_df['model'], results_df['accuracy'], marker='o', label='Accuracy')
    plt.plot(results_df['model'], results_df['f1_macro'], marker='o', label='Macro F1')
    plt.ylim(0.6, 0.85)
    plt.ylabel('Score')
    plt.title('News Source Classification Experiments')
    plt.xticks(rotation=30, ha='right')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=180)

    # Interpretable feature analysis on the best word-level model.
    best_word_model = fitted['tfidf_logreg_uni_bigram_20k']
    vec = best_word_model.named_steps['vec']
    clf = best_word_model.named_steps['clf']
    feature_names = vec.get_feature_names_out()
    coefs = clf.coef_[0]

    # sklearn binary convention: positive coefficients favor classes_[1] == 'NBC'
    top_nbc_idx = coefs.argsort()[-20:][::-1]
    top_fox_idx = coefs.argsort()[:20]
    summary = {
        'classes': list(clf.classes_),
        'top_features_for_nbc': [
            {'feature': str(feature_names[i]), 'coef': float(coefs[i])} for i in top_nbc_idx
        ],
        'top_features_for_foxnews': [
            {'feature': str(feature_names[i]), 'coef': float(coefs[i])} for i in top_fox_idx
        ],
        'classification_report_best_word_model': classification_report(y_test, best_word_model.predict(X_test), output_dict=True),
    }
    TOP_FEATURES_JSON.write_text(json.dumps(summary, indent=2))

    print('Saved:')
    print(f' - {RESULTS_CSV}')
    print(f' - {PLOT_PATH}')
    print(f' - {TOP_FEATURES_JSON}')
    print('\nBest models by accuracy:')
    print(results_df.sort_values('accuracy', ascending=False)[['model', 'accuracy', 'f1_macro']].to_string(index=False))


if __name__ == '__main__':
    main()
