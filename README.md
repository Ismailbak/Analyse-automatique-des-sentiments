# Sentiment140 NLP Project

Project scaffold for sentiment analysis on the Sentiment140 dataset. This repository contains datasets, notebooks, modular `src/` code, model storage, visualizations, and reports to support experimentation from preprocessing through model fine-tuning and evaluation.

## Project Structure

### 📁 `data/`
- `raw/` — Original Sentiment140 dataset (sentiment140.csv)
- `processed/` — Cleaned tweets and preprocessed data (cleaned_tweets.csv, features_tfidf.pkl)
- `embeddings/` — Saved word embeddings (glove_vectors.pkl, bert_embeddings.pkl)

### 📁 `notebooks/`
Jupyter notebooks for each project phase:
- `01_exploration_preprocessing.ipynb` — Data exploration and cleaning
- `02_ml_models_baselines.ipynb` — Logistic Regression, SVM, Random Forest
- `03_dl_lstm_gru.ipynb` — LSTM/GRU deep learning models
- `04_dl_bert_transfer_learning.ipynb` — BERT fine-tuning
- `05_clustering_unsupervised.ipynb` — K-Means, LDA, t-SNE
- `06_results_visualization.ipynb` — Final comparisons and charts

### 📁 `src/`
Reusable Python modules:
- `data_loader.py` — Load and split datasets
- `text_cleaning.py` — Text preprocessing and tokenization
- `feature_engineering.py` — TF-IDF, embeddings, vectorizers
- `train_ml.py` — Train classical ML models
- `train_dl.py` — LSTM/GRU model builders
- `bert_finetuning.py` — BERT fine-tuning utilities
- `evaluation.py` — Metrics, confusion matrices, plots

### 📁 `models/`
Saved model artifacts:
- `ml/` — Pickled ML models (logreg.pkl, svm.pkl, random_forest.pkl)
- `dl/` — Deep learning models (lstm_model.h5, bert_model/)
- `vectorizers/` — Saved vectorizers and tokenizers (tfidf_vectorizer.pkl, tokenizer.pkl)

### 📁 `visuals/`
Generated visualizations:
- `wordclouds/` — Positive and negative word clouds
- `confusion_matrices/` — Confusion matrix plots for each model
- `charts/` — Accuracy, F1-score comparisons, t-SNE plots

### 📁 `reports/`
Documentation and presentations:
- `Rapport_Sentiment140_NLP.pdf` — Final report (10–15 pages)
- `rapport_intermediaire_S3.pdf` — Intermediate report (3–4 pages)
- `presentation_slides.pptx` — Oral presentation

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

See `requirements.txt` for core Python dependencies (scikit-learn, TensorFlow, PyTorch, Transformers, etc.).

