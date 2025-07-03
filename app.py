from flask import Flask, render_template, request
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load model
model = TFBertForSequenceClassification.from_pretrained("bert_fake_news_model")
tokenizer = BertTokenizer.from_pretrained("bert_fake_news_model")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_text = request.form.get('news')
        if not input_text:
            return render_template('index.html', prediction="❌ Please enter news text.")

        inputs = tokenizer(
            input_text, return_tensors='tf', truncation=True,
            padding=True, max_length=128)
        outputs = model(inputs)
        logits = outputs.logits
        prediction = tf.nn.softmax(logits, axis=1)
        pred_class = tf.argmax(prediction, axis=1).numpy()[0]

        label = "Fake" if pred_class == 0 else "REAL"
        return render_template('index.html', prediction=f"📰 This news is likely: {label}")
