# Customer Review Sentiment Analysis

A simple machine learning project that classifies customer reviews as
**Positive**, **Negative**, or **Neutral** using TF-IDF features and a
Logistic Regression classifier (with a Naive Bayes comparison).

## How it works

1. Loads a labeled dataset of 30 customer reviews (`reviews.csv`)
2. Cleans the text (lowercasing, removing punctuation/extra whitespace)
3. Splits the data into training and testing sets
4. Converts text into TF-IDF feature vectors
5. Trains a Logistic Regression classifier
6. Evaluates accuracy, confusion matrix, and classification report
7. Compares against a Naive Bayes classifier
8. Predicts sentiment on new, unseen reviews
9. Saves the trained model and vectorizer to disk (`sentiment_model.pkl`, `tfidf_vectorizer.pkl`)

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python sentiment_analysis.py
```

## Files

| File | Description |
|---|---|
| `reviews.csv` | Labeled dataset (10 Positive / 10 Negative / 10 Neutral reviews) |
| `sentiment_analysis.py` | Full training and evaluation pipeline |
| `requirements.txt` | Python dependencies |

## Example output

```
Predictions on new reviews:
  'Amazing product and fast delivery!' -> Positive
  'The quality is terrible.' -> Negative
  'The package arrived today.' -> Neutral
  'Absolutely wonderful, I will buy again.' -> Positive
  'Not worth the price, very disappointed.' -> Negative
```

## Notes

This is a small demonstration dataset (30 reviews), so accuracy on unseen
text will be limited — it's meant to show the end-to-end pipeline rather
than serve as a production-grade sentiment model.
