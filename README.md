# SMS Spam Detection using Machine Learning and LSTM

This project is an end-to-end SMS spam detection system built using classical machine learning models and a deep learning LSTM model. It classifies incoming SMS messages as either **Spam** or **Ham (Not Spam)**.

## Overview

The goal of this project is to detect and classify SMS messages that are spam using natural language processing techniques. The application is built using **Streamlit** for an interactive frontend, and is powered by four models:

- Random Forest
- Naive Bayes
- Support Vector Machine (SVM)
- LSTM (Long Short-Term Memory)

---

## Problem Statement

With the increasing volume of SMS communication, spam messages have become a serious concern. This project aims to create a robust model that accurately identifies spam messages and helps in filtering them out before reaching the user.

---

## Features

- Real-time SMS classification
- Four different models to choose from
- Performance metrics displayed for all models
- Streamlit web interface that mimics a native messaging app
- Collapsible hamburger menu with navigation to 'Spam messages' section
- Mobile and desktop compatible UI

---

## Motivation

The rise in phishing attempts and promotional spam through SMS inspired this project. Providing users with a tool that can predict spam messages accurately using both traditional and deep learning models is a step toward safer communication.

---

## Tech Stack

| Component     | Technology               |
|--------------|--------------------------|
| Frontend     | Streamlit                |
| Backend      | Python, Pandas, Numpy    |
| ML Models    | Scikit-learn (RF, NB, SVM), TensorFlow (LSTM) |
| Database     | Not applicable (CSV-based for this version) |
| Visualization| Matplotlib, Seaborn      |

---

## Dataset

- Source: [UCI SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- Format: CSV with 2 columns (`label`, `message`)
- Classes: `spam`, `ham`

---

## Data Preprocessing

- Lowercasing text
- Removing punctuation and stopwords
- Lemmatization
- Tokenization
- Padding (for LSTM)
- Label encoding

---

## Models Used

| Model          | Description                                     |
|----------------|-------------------------------------------------|
| Random Forest  | Ensemble model with multiple decision trees     |
| Naive Bayes    | Probabilistic classifier for text classification |
| SVM            | Max-margin classifier good for text data        |
| LSTM           | Deep learning model capturing sequential patterns |

---

## Evaluation Metrics

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **Confusion Matrix**

LSTM and Random Forest showed the highest accuracy and generalization.

---

## How to Run

```bash
git clone https://github.com/yourusername/sms-spam-detector.git
cd sms-spam-detector
pip install -r requirements.txt
streamlit run app.py
