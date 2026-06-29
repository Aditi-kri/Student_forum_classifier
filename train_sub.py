import re
import joblib
import nltk
import pandas as pd
import matplotlib.pyplot as plt

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


nltk.download("stopwords")


df = pd.read_csv("posts.csv", encoding="cp1252")

# Keep only genuine posts
df = df[df["spam_label"].str.lower().str.strip() == "genuine"]

# Remove missing values
df = df.dropna(subset=["post", "subject"])

# Standardize subject names
df["subject"] = df["subject"].str.strip()


stemmer = PorterStemmer()

stop_words = set(stopwords.words("english"))

def preprocess(text):

    text = str(text).lower()

    # Remove numbers and punctuation
    text = re.sub(r"[^a-zA-Z ]", " ", text)

    words = text.split()

    # Remove stopwords + stemming
    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

df["clean_post"] = df["post"].apply(preprocess)


# Load Sentence Transformer

print("Loading Sentence Transformer...")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Generating sentence embeddings...")

X = embedding_model.encode(
    df["clean_post"].tolist(),
    show_progress_bar=True
)

y = df["subject"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Train Model
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

base_model = LinearSVC(
    C=1.0,
    random_state=42
)

subject_model = CalibratedClassifierCV(
    base_model,
    cv=5
)

subject_model.fit(X_train, y_train)

#subject_model = LogisticRegression(
 #   max_iter=1000,
  #  random_state=42,
   # class_weight="balanced"
 #)

#subject_model.fit(X_train, y_train)
# Prediction

y_pred = subject_model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)

print("\n")
print("="*50)
print("Subject Classification Results")
print("="*50)

print(f"Accuracy : {accuracy:.4f}")


print("\nClassification Report\n")

print(classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix\n")

print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=subject_model.classes_
)

disp.plot(cmap="Blues")

plt.title("Subject Classification Confusion Matrix")

plt.show()


# Save Model

joblib.dump(subject_model, "subject_model.pkl")

print("\nModel Saved Successfully")

print("subject_model.pkl")

print("\nSentence Transformer Used : all-MiniLM-L6-v2")

print("\nClasses Learned:")

print(subject_model.classes_)