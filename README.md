# Depression and Anxiety Detection from Social Media Text using NLP and Machine Learning

## Overview

This project focuses on detecting depression and anxiety-related content from Twitter data using Natural Language Processing (NLP) and Machine Learning techniques. The system analyzes tweet text, extracts meaningful features, and classifies tweets into mental health-related categories.

The project demonstrates a complete NLP pipeline including data preprocessing, feature extraction, machine learning model training, evaluation, and prediction.

---

## Features

* Tweet text preprocessing and cleaning
* TF-IDF feature extraction
* Logistic Regression classifier
* Linear Support Vector Machine (SVM)
* Random Forest classifier
* Soft Voting Ensemble model
* Model performance evaluation
* FastAPI-based prediction API
* Confidence score generation for predictions

---

## Project Workflow

1. Data Collection and Loading
2. Text Preprocessing
3. TF-IDF Feature Extraction
4. Model Training
5. Hyperparameter Tuning
6. Ensemble Learning
7. Model Evaluation
8. Prediction and API Deployment

---

## System Architecture

Tweet Input
↓
Preprocessing
↓
TF-IDF Vectorization
↓

* Logistic Regression
* Linear SVM
* Random Forest

↓

Soft Voting Ensemble
↓
Ensemble Prediction Module
↓
FastAPI Service
↓
Prediction Output (Label + Confidence Score)

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* NLTK
* FastAPI
* Uvicorn
* Matplotlib
* Seaborn
* Joblib

---

## Machine Learning Models

The following machine learning algorithms were used:

* Logistic Regression
* Linear Support Vector Machine (SVM)
* Random Forest
* Soft Voting Ensemble

---

## Dataset

The project uses a tweet-based mental health dataset containing approximately 19,000+ labeled English tweets.

The dataset was preprocessed to remove noise and convert textual data into machine-readable features using TF-IDF vectorization.

---

## Evaluation Metrics

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

---

## API Example

### Request

```json
{
  "text": "I feel depressed and hopeless"
}
```

### Response

```json
{
  "prediction": 1,
  "label": "Depressed",
  "confidence": 85.33
}
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/depression-anxiety-detection-nlp.git
cd depression-anxiety-detection-nlp
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the API:

```bash
uvicorn app:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

to access the Swagger API documentation.

---

## Future Improvements

* Integration of Transformer-based models (BERT, RoBERTa)
* Multi-class mental health classification
* Real-time tweet monitoring
* Web dashboard for visualization
* Larger and more diverse datasets

---

## Authors

* Javed Akhtar

---

## Disclaimer

This project is intended for academic and research purposes only. It should not be used as a clinical diagnostic tool for mental health assessment.
