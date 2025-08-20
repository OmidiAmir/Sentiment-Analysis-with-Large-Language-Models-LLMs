# BERT Fine-Tuning on SMILE Twitter Emotion Dataset
This project fine-tunes BERT for sentiment analysis using the SMILE Twitter Emotion dataset. The goal is to classify emotional categories from tweets by leveraging transfer learning with a pretrained language model.

## Features
- Data Preparation: Load and preprocess SMILE dataset, train/validation split
- Tokenizer & Model: Load bert-base-uncased with Hugging Face Transformers
- Fine-Tuning: Train BERT using PyTorch with AdamW optimizer and linear scheduler
- Evaluation: Compute validation loss, weighted F1-score, and per-class accuracy

## Tech Stack
- Python, PyTorch, Hugging Face Transformers
- scikit-learn, pandas, NumPy
- tqdm for training progress

  
