#  Student Forum Discussion System

## Overview

The **Student Forum Discussion System** is an NLP-based web application that automatically analyzes forum posts submitted by students. The system first determines whether a post is **Genuine** or **Spam**. If the post is identified as genuine, it is further classified into one of several academic subjects, including **Data Structures and Algorithms (DSA)**, **Operating Systems (OS)**, **Database Management Systems (DBMS)**, **Computer Networks (CN)**, **Object-Oriented Programming (OOP)**, or **Miscellaneous**.

The project uses modern Natural Language Processing techniques with **Sentence Transformers** to generate semantic sentence embeddings and **Machine Learning** models for accurate text classification. The application is deployed using **Streamlit**, providing a simple and interactive user interface for real-time predictions.

---

## Features

*  Detects whether a forum post is **Spam** or **Genuine**
*  Classifies genuine posts into:

  * Data Structures & Algorithms (DSA)
  * Operating Systems (OS)
  * Database Management Systems (DBMS)
  * Computer Networks (CN)
  * Object-Oriented Programming (OOP)
  * Miscellaneous
*  Uses **Sentence Embeddings (all-MiniLM-L6-v2)** for semantic text representation
*  Performs NLP preprocessing including:
  * Tokenization
  * Stop-word Removal
  * Porter Stemming
*  Displays confidence percentages for predictions
*  Evaluates models using Accuracy, Classification Report, and Confusion Matrix
*  Interactive Streamlit web interface
*  Maintains prediction history during the current session
*  Includes clickable example posts and live word count

---

## Technologies Used

### Programming Language

* Python

### Machine Learning & NLP

* Sentence Transformers
* Scikit-learn
* Logistic Regression
* LinearSVC 
* NLTK

### Libraries

* Pandas
* NumPy
* Joblib
* Matplotlib

### Web Framework

* Streamlit

---

## Project Workflow

1. The user enters a forum post through the Streamlit interface.
2. The text is preprocessed using tokenization, stop-word removal, and stemming.
3. A Sentence Transformer converts the cleaned text into semantic sentence embeddings.
4. The Spam Detection model predicts whether the post is Spam or Genuine.
5. If the post is Genuine, the Subject Classification model predicts the most relevant academic subject.
6. The application displays:

   * Prediction result
   * Confidence percentage
   * Subject category
   * Prediction history

---

## Machine Learning Pipeline

**Input Post**
→ **Text Preprocessing**
→ **Sentence Embedding Generation**
→ **Spam Classification**
→ **Subject Classification**
→ **Prediction with Confidence Score**

---

## Project Structure

```text
Student_Forum_Classifier/
│
├── app.py
├── predictor.py
├── train_spam.py
├── train_subject.py
├── spam_model.pkl
├── subject_model.pkl
├── posts.csv
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/student-forum-classifier.git
cd student-forum-classifier
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## Model Evaluation

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

These metrics help measure the effectiveness of both spam detection and subject classification.

---

## Future Enhancements

* User authentication and login system
* Forum database integration
* Automatic moderator alerts for spam posts
* Support for additional academic subjects
* Sentiment analysis for forum discussions
  
  

---

## Author

Developed as an NLP and Machine Learning project for automated forum post moderation and academic subject classification using Python, Sentence Transformers, Scikit-learn, and Streamlit.
