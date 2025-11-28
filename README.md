# 📰 Fake News Detection using Bi-Directional LSTM (TensorFlow/Keras)

This project implements a deep learning-based Fake News Detector using Natural Language Processing techniques and a Bidirectional LSTM network. The model classifies news articles as **Real** or **Fake** based on their textual content.  
The project trains on two datasets: `Fake.csv` and `True.csv`, automatically labeling them for supervised learning.

---

## 📌 Project Overview

Fake news is one of the biggest challenges in today's digital world. This project demonstrates how Deep Learning techniques, particularly a **Bidirectional LSTM**, can extract context and patterns from language to classify articles with high accuracy.

The workflow includes:

- Data loading and labeling  
- Text cleaning and normalization  
- Tokenization and padding  
- Model training  
- Evaluation and saving the best model

---

## 🧠 Model Architecture

| Layer Type | Description |
|------------|------------|
| Embedding Layer | Converts words to numerical dense vectors |
| Bidirectional LSTM | Learns long-term context in both forward and backward directions |
| Dense Layer | Feature extraction |
| Dropout | Reduces overfitting |
| Output Layer | Sigmoid activation for binary classification |

- **Loss Function:** Binary Crossentropy  
- **Optimizer:** Adam  
- **Evaluation Metric:** Accuracy  

---

## 📂 Dataset Format

Link : https://www.kaggle.com/api/v1/datasets/download/saurabhshahane/fake-news-classification


You must include two CSV files in the project directory:

| File Name | Meaning |
|-----------|---------|
| `Fake.csv` | Contains fake news articles |
| `True.csv` | Contains real news articles |

Both files must contain at least:

```
text  → column with the full article content
```

Labels are added automatically:

```
Real News  = 0  
Fake News  = 1
```

---

## ⚙️ Requirements

Install the necessary dependencies using:

```sh
pip install tensorflow pandas numpy scikit-learn
```

> **Recommended Python Version:** 3.10 (TensorFlow compatible)

---

## ▶️ How to Run

1. Ensure `Fake.csv` and `True.csv` are in the same project folder.
2. Activate your Python virtual environment (recommended).
3. Run the script:

```sh
python Main.py
```

The script will:

✔ Preprocess the text  
✔ Train the BiLSTM model  
✔ Evaluate validation accuracy  
✔ Save the best performing model automatically as:

```
best_bilstm_model.keras
```
✔ Ask for input where wecan give unseen data 
---

## 📊 Example Output

You may see something like:

```
Epoch 6/10
Validation Accuracy: 99.91%
Best model saved to: best_bilstm_model.keras
```

Meaning training was successful and the model performed well.

---

## 📁 Project Files Summary

| File | Purpose |
|------|---------|
| `Main.py` | Main training script |
| `Fake.csv` | Fake news dataset |
| `True.csv` | Real news dataset |
| `best_bilstm_model.keras` | Saved trained model |
| `README.md` | Documentation |

---

## 👤 Authors

 | R.No | Name |
|------|------|
| SE23UARI059 | K Jaya Prakash Varma  |
| SE23UARI058 | K Phalguna Reddy |
| SE23UARI019 | B Lakshay |
| SE23UARI017 | B Pranadeep Reddy |
| SE23UMCS007 | B Yashwant Raj |
---

⭐ If this project helped you, consider giving the repository a star.
