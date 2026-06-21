# Depression and Anxiety Detection from Social Media Text using NLP and Machine Learning

![Python](https://img.shields.io/badge/Python-3.x-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![NLP](https://img.shields.io/badge/NLP-TF--IDF-red)

## Overview

This project focuses on detecting depression and anxiety-related content from Twitter data using Natural Language Processing (NLP) and Machine Learning techniques. The system analyzes tweet text, extracts meaningful features, and classifies tweets into mental health-related categories.

The project demonstrates a complete NLP pipeline including data preprocessing, feature extraction, machine learning model training, evaluation, and prediction.

This project demonstrates the application of Natural Language Processing (NLP), feature engineering, ensemble learning, and REST API deployment for social media text classification.

---

## Project Preview

### System Architecture

![Architecture](screenshots/architecture.png)

### API Documentation

![Swagger UI](screenshots/swagger-ui.png)

### Prediction Example

![Prediction Output](screenshots/api-output.png)

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

## Repository Structure

```text
Depression-Anxiety-Detection-NLP
│
├── data/
├── models/
├── notebooks/
├── screenshots/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```
---

## Installation

Clone the repository:

```bash
git clone https://github.com/JavedAkhtar0/Depression-Anxiety-Detection-NLP.git
cd Depression-Anxiety-Detection-NLP
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

## Results

The system was trained and evaluated using Logistic Regression, Linear SVM, Random Forest, and a Soft Voting Ensemble model.

The models were assessed using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix analysis. Experimental results demonstrated that TF-IDF combined with classical machine learning techniques can effectively classify mental health-related social media posts.

The ensemble approach improved prediction stability by combining predictions from multiple classifiers.

---

## Future Improvements

* Integration of Transformer-based models (BERT, RoBERTa)
* Multi-class mental health classification
* Real-time tweet monitoring
* Web dashboard for visualization
* Larger and more diverse datasets

---

## Author

Javed Akhtar

MCA Student | Machine Learning & Backend Development Enthusiast

GitHub: [JavedAkhtar0](https://github.com/JavedAkhtar0)
---

## Disclaimer

This project was developed for learning and research purposes to explore the application of Natural Language Processing and Machine Learning in text classification tasks.
