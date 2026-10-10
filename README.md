# Fake vs. Real News Classification

Classifying ~45k news articles as real or fake with classical NLP and gradient-boosted models,
then testing whether extra feature engineering, auxiliary data and ensembling actually help.

**Notebook:** [`fake_news_classification.ipynb`](fake_news_classification.ipynb)

## Approach

| Stage | What is done |
|---|---|
| EDA | subject distributions, word clouds, text-length analysis, data profiling |
| Cleaning | 196 duplicates dropped, subjects mapped to *News / Politics / Other* domains, lemmatisation and stop-word removal |
| Features | TF-IDF, 5,000 features, unigrams + bigrams |
| Baseline models | Logistic Regression, Random Forest, XGBoost, MLP, each evaluated overall **and per domain** |
| Improvement | train/val/test split, stylistic features (sentence count, word length, punctuation, casing), auxiliary fake headlines, zero-shot DistilBERT check, weighted XGBoost + RF ensemble |
| Validation | held-out test set and 5-fold cross-validation |

## Results

| Model | Test F1 |
|---|---|
| Logistic Regression | 0.99 |
| Random Forest | 1.00 |
| XGBoost | 1.00 |
| Ensemble + engineered features | 0.997 (CV F1 0.997) |

The improved pipeline made slightly *more* mistakes than the baseline (19 vs 12 false
positives), which is the most useful finding: on this dataset the extra complexity did not pay
off. Near-perfect scores across every model also point to strong source and style artefacts in
the data, so the numbers say more about the dataset than about fake-news detection in general.

## Run it

```bash
pip install -r requirements.txt
jupyter notebook fake_news_classification.ipynb
```

The dataset is downloaded automatically with `kagglehub`
([Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)).

## Next steps

- Ablate each improvement to find which one hurt performance.
- Strip source artefacts (agency bylines, datelines) and re-evaluate.
- Test cross-dataset generalisation and tune with Optuna.
