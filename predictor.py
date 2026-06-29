import re
import joblib
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import os
from huggingface_hub import login

hf_token = os.getenv("HF_TOKEN")

if hf_token:
    login(token=hf_token)
nltk.download("stopwords", quiet=True)
from sentence_transformers import SentenceTransformer
# Load Models
print("Loading models...")

spam_model = joblib.load("spam_model.pkl")
subject_model = joblib.load("subject_model.pkl")

print("Loading Sentence Transformer...")
model = SentenceTransformer("all-MiniLM-L6-v2")
model.save("embedding_model")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# NLP Preprocessing
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def preprocess(text):
    text = str(text).lower()

    # Remove punctuation and numbers
    text = re.sub(r"[^a-zA-Z ]", " ", text)

    # Tokenization
    words = text.split()

    # Stop-word removal + stemming
    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# Prediction Function

def predict_post(post):

    # Clean text
    clean_post = preprocess(post)

    # Convert sentence to embedding
    embedding = embedding_model.encode([clean_post])

    
    # Spam Prediction
    
    spam_prediction = spam_model.predict(embedding)[0]

    spam_probabilities = spam_model.predict_proba(embedding)[0]

    spam_classes = spam_model.classes_

    spam_scores = {
        spam_classes[i]: float(spam_probabilities[i])
        for i in range(len(spam_classes))
    }

    # If spam
    if spam_prediction.lower() == "spam":

        return {
            "type": "Spam",
            "spam_confidence": round(spam_scores["spam"] * 100, 2),
            "genuine_confidence": round(spam_scores["genuine"] * 100, 2),
            "subject": None,
            "subject_confidence": None
        }

    
    # Subject Prediction
    
    subject_prediction = subject_model.predict(embedding)[0]

    subject_probabilities = subject_model.predict_proba(embedding)[0]

    subject_classes = subject_model.classes_

    subject_scores = {
        subject_classes[i]: float(subject_probabilities[i])
        for i in range(len(subject_classes))
    }

    return {
        "type": "Genuine",

        "spam_confidence": round(spam_scores["spam"] * 100, 2),

        "genuine_confidence": round(spam_scores["genuine"] * 100, 2),

        "subject": subject_prediction,

        "subject_confidence": round(
            subject_scores[subject_prediction] * 100,
            2
        ),

        # Optional: probabilities of all subjects
       # "all_subject_probabilities": {
          #  subject: round(prob * 100, 2)
          #  for subject, prob in subject_scores.items()
       # }
    }


if __name__ == "__main__":

    while True:

        post = input("\nEnter forum post (type 'exit' to quit): ")

        if post.lower() == "exit":
            break

        result = predict_post(post)

        print("\nPrediction Result")
        print("-" * 40)

        print("Type :", result["type"])

        print(
            f"Genuine Confidence : "
            f"{result['genuine_confidence']}%"
        )

        print(
            f"Spam Confidence : "
            f"{result['spam_confidence']}%"
        )

        if result["type"] == "Genuine":

            print(f"Subject : {result['subject']}")

            print(
                f"Subject Confidence : "
                f"{result['subject_confidence']}%"
            )

           

           # for subject, prob in result["all_subject_probabilities"].items():

             #   print(f"{subject:15} : {prob}%")