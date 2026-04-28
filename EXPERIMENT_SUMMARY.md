# Experiment Summary: News Source Classification

This note summarizes the baseline and early classical-model experiments on `scraped_headlines_clean_latest.csv`.

## What was added

- Reproducible experiment runner: `baseline_experiments.py`
- Metrics table saved to `experiment_results.csv`
- Line plot of accuracy and macro-F1 across model iterations saved to `experiment_metrics.png`
- Interpretable feature analysis for the best word-level logistic-regression model saved to `top_features_word_logreg.json`

## Dataset used

- File: `scraped_headlines_clean_latest.csv`
- Task: binary classification of headlines by source (`FoxNews` vs `NBC`)
- Split used in the experiments below: stratified 80/20 train-test split with `random_state=42`

## Models compared

1. `baseline_tfidf_logreg_100`
   - TF-IDF (`max_features=100`) + Logistic Regression
   - This is the spec-style baseline intended to reproduce the course starter setup.

2. `tfidf_complement_nb_uni_bigram`
   - TF-IDF uni+bigram + Complement Naive Bayes

3. `tfidf_linear_svc_uni_bigram`
   - TF-IDF uni+bigram + Linear SVC

4. `tfidf_logreg_unigram_20k`
   - TF-IDF unigrams (`max_features=20k`) + Logistic Regression

5. `tfidf_logreg_uni_bigram_20k`
   - TF-IDF uni+bigram (`max_features=20k`) + Logistic Regression
   - Best interpretable word-level model in this experiment set

6. `charwb_logreg_3_5`
   - Character n-gram TF-IDF (`char_wb`, 3-5 grams) + Logistic Regression
   - Best overall accuracy in this experiment set

## Results (local run)

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| baseline_tfidf_logreg_100 | 0.6763 | 0.6699 |
| tfidf_complement_nb_uni_bigram | 0.7658 | 0.7645 |
| tfidf_linear_svc_uni_bigram | 0.7763 | 0.7739 |
| tfidf_logreg_unigram_20k | 0.7842 | 0.7815 |
| tfidf_logreg_uni_bigram_20k | 0.7908 | 0.7881 |
| charwb_logreg_3_5 | 0.7947 | 0.7915 |

## Takeaways

- The spec-style baseline reproduced at **0.6763 accuracy**, which is above the course baseline target of roughly 0.6649.
- Moving from 100 TF-IDF features to richer feature spaces produces a large jump in performance.
- Among the classical, interpretable word-based models, **TF-IDF uni+bigram + Logistic Regression** performed best at **0.7908 accuracy / 0.7881 macro-F1**.
- The highest accuracy in this batch came from the **character n-gram logistic regression** model at **0.7947 accuracy / 0.7915 macro-F1**.
- For an exploratory / analysis component, the word-level logistic-regression coefficients are useful because they reveal which unigrams and bigrams are most predictive of each source.

## Recommended next step

Use `tfidf_logreg_uni_bigram_20k` as the main classical baseline for reporting and interpretation, and compare later transformer runs against it under the same data split and metrics.
