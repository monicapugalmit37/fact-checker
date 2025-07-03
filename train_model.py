import os
import tensorflow as tf

# === Setup GPU Memory Growth and Allocator BEFORE ANYTHING ELSE ===
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU memory growth enabled.")
    except RuntimeError as e:
        print("❌ GPU setup error:", e)
else:
    print("⚠️ No GPU found. Running on CPU.")

# Other imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, TFBertForSequenceClassification
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# === Load dataset ===
df = pd.read_csv("news_dataset.csv")  # columns: label, text
df.dropna(subset=['label', 'text'], inplace=True)

# Encode labels (real=0, fake=1)
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42)

# === Load tokenizer ===
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize inputs with smaller max_length
train_encodings = tokenizer(
    list(X_train), truncation=True, padding=True, max_length=64)
val_encodings = tokenizer(
    list(X_val), truncation=True, padding=True, max_length=64)

# Convert to tf.data.Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings), y_train)).shuffle(1000).batch(2)  # batch size reduced
val_dataset = tf.data.Dataset.from_tensor_slices((
    dict(val_encodings), y_val)).batch(2)

# === Load pre-trained BERT model ===
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')

# Compile
optimizer = Adam(learning_rate=2e-5)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

# === Train ===
model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=5,  # Increased epochs for more learning
    batch_size=8,
    callbacks=[EarlyStopping(monitor='val_loss', patience=2)]
)

# === Save Model and Tokenizer ===
model.save_pretrained("bert_fake_news_model")
tokenizer.save_pretrained("bert_fake_news_model")

print("✅ Model trained and saved successfully.")
