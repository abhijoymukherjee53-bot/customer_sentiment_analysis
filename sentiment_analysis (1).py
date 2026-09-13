"""
Customer Review Sentiment Analysis
-----------------------------------
Classifies customer reviews as Positive, Negative, or Neutral
using TF-IDF features and a Logistic Regression classifier.

Run with:
    python sentiment_analysis.py
"""

import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib


def clean_text(text):
    """Lowercase, strip punctuation, and collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    # 1. Load dataset
    df = pd.read_csv("reviews.csv")
    print("Dataset shape:", df.shape)
    print(df["sentiment"].value_counts(), "\n")

    # 2. Clean text
    df["clean_review"] = df["review"].apply(clean_text)

    # 3. Split data
    X = df["clean_review"]
    y = df["sentiment"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"Training samples: {len(X_train)} | Testing samples: {len(X_test)}\n")

    # 4. TF-IDF features
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print("Vocabulary size:", len(vectorizer.get_feature_names_out()), "\n")

    # 5. Train Logistic Regression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    # 6. Evaluate
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print(f"Logistic Regression accuracy: {acc:.2f}\n")

    labels = ["Positive", "Negative", "Neutral"]
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(
        cm, index=[f"Actual {l}" for l in labels], columns=[f"Pred {l}" for l in labels]
    )
    print("Confusion Matrix:")
    print(cm_df, "\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # 7. Compare with Naive Bayes
    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train)
    nb_acc = accuracy_score(y_test, nb_model.predict(X_test_tfidf))
    print(f"Naive Bayes accuracy: {nb_acc:.2f}\n")

    # 8. Try it on new reviews
    new_reviews = [
        "Amazing product and fast delivery!",
        "The quality is terrible.",
        "The package arrived today.",
        "Absolutely wonderful, I will buy again.",
        "Not worth the price, very disappointed.",
    ]
    new_clean = [clean_text(r) for r in new_reviews]
    new_tfidf = vectorizer.transform(new_clean)
    new_preds = model.predict(new_tfidf)

    print("Predictions on new reviews:")
    for review, pred in zip(new_reviews, new_preds):
        print(f"  '{review}' -> {pred}")

    # 9. Save the trained model and vectorizer
    joblib.dump(model, "sentiment_model.pkl")
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
    print("\nModel and vectorizer saved as sentiment_model.pkl and tfidf_vectorizer.pkl")


if __name__ == "__main__":
    main()
