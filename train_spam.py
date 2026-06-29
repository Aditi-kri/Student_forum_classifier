import re
import joblib
import nltk
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os
from huggingface_hub import login

hf_token = os.getenv("HF_TOKEN")

if hf_token:
    login(token=hf_token)
from sentence_transformers import SentenceTransformer

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt


nltk.download("stopwords")


# Load Dataset

df = pd.read_csv("posts.csv", encoding="cp1252")

# Remove missing values
df = df.dropna(subset=["post", "spam_label"])

# Standardize labels
df["spam_label"] = df["spam_label"].str.lower().str.strip()


# NLP Preprocessing

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def preprocess(text):

    text = str(text).lower()

    text = re.sub(r"[^a-zA-Z ]", " ", text)

    words = text.split()

    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

df["clean_post"] = df["post"].apply(preprocess)


# Sentence Embedding Model

print("Loading Sentence Transformer...")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Generating sentence embeddings...")

X = embedding_model.encode(
    df["clean_post"].tolist(),
    show_progress_bar=True
)

y = df["spam_label"]


# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Train Logistic Regression

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)


pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)

print("\nSpam Classification Results")
print("=" * 45)

print(f"Accuracy : {accuracy:.4f}")


print("\nClassification Report\n")

print(classification_report(y_test, pred))


cm = confusion_matrix(y_test, pred)

print("\nConfusion Matrix\n")

print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

disp.plot(cmap="Blues")

plt.title("Spam Detection Confusion Matrix")

plt.show()


joblib.dump(model, "spam_model.pkl")

print("\nModel Saved Successfully")
print("spam_model.pkl")

print("\nSentence Transformer Used:")
print("all-MiniLM-L6-v2")