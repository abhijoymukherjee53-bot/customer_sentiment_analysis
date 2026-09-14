import re
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Review Sentiment Classifier", page_icon="💬", layout="centered")


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_resource
def train():
    df = pd.read_csv("reviews.csv")
    df["clean_review"] = df["review"].apply(clean_text)

    X = df["clean_review"]
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train_tfidf, y_train)
    lr_acc = accuracy_score(y_test, lr_model.predict(X_test_tfidf))

    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train)
    nb_acc = accuracy_score(y_test, nb_model.predict(X_test_tfidf))

    return df, vectorizer, lr_model, lr_acc, nb_model, nb_acc


df, vectorizer, lr_model, lr_acc, nb_model, nb_acc = train()

st.title("💬 Customer Review Sentiment Classifier")
st.caption(
    "TF-IDF features + Logistic Regression, trained on a 30-review dataset "
    "(10 Positive / 10 Negative / 10 Neutral)."
)

with st.expander("Model details"):
    st.write(f"**Vocabulary size:** {len(vectorizer.get_feature_names_out())}")
    st.write(f"**Logistic Regression test accuracy:** {lr_acc:.2f}")
    st.write(f"**Naive Bayes test accuracy:** {nb_acc:.2f}")
    st.dataframe(df[["review", "sentiment"]], use_container_width=True)

st.subheader("Try it")
review = st.text_area(
    "Enter a customer review",
    value="Amazing product and fast delivery!",
    height=100,
)

if st.button("Classify sentiment", type="primary"):
    if review.strip():
        cleaned = clean_text(review)
        vec = vectorizer.transform([cleaned])
        pred = lr_model.predict(vec)[0]
        proba = lr_model.predict_proba(vec)[0]
        classes = lr_model.classes_

        color = {"Positive": "green", "Negative": "red", "Neutral": "orange"}.get(pred, "blue")
        st.markdown(f"### Predicted sentiment: :{color}[{pred}]")

        proba_df = pd.DataFrame({"sentiment": classes, "confidence": proba}).sort_values(
            "confidence", ascending=False
        )
        st.bar_chart(proba_df.set_index("sentiment"))
    else:
        st.warning("Please enter a review first.")

st.divider()
st.caption(
    "Built from a scikit-learn TF-IDF + Logistic Regression pipeline. "
    "Small training set (30 examples) means edge cases like sarcasm may be misclassified."
)
