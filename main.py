import os
import re
import string
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ==============================
# 1. CONFIG
# ==============================

FAKE_PATH = r"C:\Users\pjay4\OneDrive\Desktop - Copy\vscode\nlp\Fake.csv"
TRUE_PATH = r"C:\Users\pjay4\OneDrive\Desktop - Copy\vscode\nlp\True.csv"

LABEL_COL = "label"     # 0 = REAL, 1 = FAKE

MAX_SEQ_LEN    = 300
EMBED_DIM      = 128
LSTM_UNITS     = 128
BATCH_SIZE     = 64
EPOCHS         = 10

# ==============================
# 2. LOAD DATA
# ==============================

fake_df = pd.read_csv(FAKE_PATH)
true_df = pd.read_csv(TRUE_PATH)

# CSV columns: title, text, subject, date
fake_df[LABEL_COL] = 1  # FAKE
true_df[LABEL_COL] = 0  # REAL

# Combine title + text
fake_df["content"] = (fake_df["title"].astype(str) + " " + fake_df["text"].astype(str)).str.strip()
true_df["content"] = (true_df["title"].astype(str) + " " + true_df["text"].astype(str)).str.strip()

df = pd.concat([fake_df, true_df], ignore_index=True)
print("Columns:", df.columns.tolist())
print(df[["title", "text", "content", LABEL_COL]].head())

# ==============================
# 3. CLEAN TEXT
# ==============================

def clean_text(t):
    if pd.isna(t):
        return ""
    t = str(t).lower()
    t = re.sub(r"http\S+|www\S+|https\S+", " ", t)
    t = re.sub(r"<.*?>", " ", t)
    t = re.sub(r"\d+", " ", t)
    t = t.translate(str.maketrans("", "", string.punctuation))
    t = re.sub(r"\s+", " ", t).strip()
    return t

df["clean"] = df["content"].apply(clean_text)
df = df[df["clean"].str.len() > 0]

texts = df["clean"].tolist()
y = df[LABEL_COL].astype(int).values   # 0 = REAL, 1 = FAKE

print("\nLabel distribution:\n", pd.Series(y).value_counts())

# ==============================
# 4. TOKENIZE + PAD (FULL VOCAB)
# ==============================

tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)

vocab_size = len(tokenizer.word_index) + 1
print("\nVocab size:", vocab_size)

padded = pad_sequences(
    sequences,
    maxlen=MAX_SEQ_LEN,
    padding="post",
    truncating="post"
)

X = padded
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# ==============================
# 5. TRAIN / VALIDATION SPLIT
# ==============================

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weights = {int(cls): float(w) for cls, w in zip(np.unique(y_train), class_weights_array)}
print("\nClass weights:", class_weights)

# ==============================
# 6. BUILD BIDIRECTIONAL LSTM MODEL
# ==============================

model = Sequential()
model.add(
    Embedding(
        input_dim=vocab_size,
        output_dim=EMBED_DIM,
    )
)
model.add(Bidirectional(LSTM(LSTM_UNITS, return_sequences=False)))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\nModel Summary:\n")
model.summary()

# ==============================
# 7. TRAINING
# ==============================

checkpoint_path = "best_bilstm_model.keras"

early_stopping = EarlyStopping(
    monitor="val_accuracy",
    patience=3,
    mode="max",
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    checkpoint_path,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[early_stopping, checkpoint],
    verbose=1
)

# ==============================
# 8. EVALUATION
# ==============================

val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"\nValidation Accuracy: {val_acc * 100:.2f}%")

y_val_proba = model.predict(X_val)
y_val_pred = (y_val_proba >= 0.5).astype(int).ravel()

print("\nClassification Report:\n")
print(classification_report(y_val, y_val_pred, digits=4))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_val, y_val_pred))

unique, counts = np.unique(y_val_pred, return_counts=True)
print("\nUnique predictions on validation set:", dict(zip(unique, counts)))

# ==============================
# 9. SAVE TOKENIZER
# ==============================

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("\nTokenizer saved successfully!")

# ==============================
# 10. PREDICTION FUNCTION
# ==============================

def predict_news(text, threshold=0.5):
    """
    Takes raw text (headline or full article),
    cleans it exactly like training, and predicts REAL / FAKE.
    """
    text_clean = clean_text(text)
    if not text_clean:
        return "REAL", 0.0  # or handle as unknown

    seq = tokenizer.texts_to_sequences([text_clean])
    # IMPORTANT: match training padding + truncating
    pad = pad_sequences(
        seq,
        maxlen=MAX_SEQ_LEN,
        padding="post",
        truncating="post"
    )
    proba = model.predict(pad)[0][0]

    # proba ≈ P(label == 1 == FAKE)
    label = "FAKE" if proba >= threshold else "REAL"
    return label, float(proba)

# ==============================
# 11. QUICK CHECK ON REAL/FAKE ROWS
# ==============================

print("\nTesting with REAL row from dataset:")
real_example = df[df[LABEL_COL] == 0]["content"].iloc[0]
print("TEXT SAMPLE:", real_example[:200], "...")
print("PREDICTION:", predict_news(real_example))

print("\nTesting with FAKE row from dataset:")
fake_example = df[df[LABEL_COL] == 1]["content"].iloc[0]
print("TEXT SAMPLE:", fake_example[:200], "...")
print("PREDICTION:", predict_news(fake_example))

# ==============================
# 12. INTERACTIVE USER INPUT
# ==============================

print("\nModel is ready! Type a news article or headline and I'll tell you if it's REAL or FAKE.")
print("Type 'exit' to quit.\n")

while True:
    user_text = input("Enter news text: ")
    if user_text.strip().lower() in ["exit", "quit", "q"]:
        print("Exiting...")
        break

    label, proba = predict_news(user_text)
    print(f"Prediction: {label} (model score = {proba:.4f})\n")
