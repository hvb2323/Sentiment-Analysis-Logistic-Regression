from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import re

app = Flask(__name__)



# Load your dataset
train_data = pd.read_csv('twitter_training.csv')
train_data.columns = ['id', 'information', 'type', 'text']
# Fill NaN values with empty strings
train_data['text'].fillna('', inplace=True)

# Assuming your dataset has columns 'text' and 'type'
X_train = train_data['text']
y_train = train_data['type']

# Convert set of stopwords to a list and preprocess them
stop_words_set = set(stopwords.words('english'))
stop_words = [re.sub('[^A-Za-z0-9 ]+', '', word.lower()) for word in stop_words_set]

# Train your model
bow_counts = CountVectorizer(tokenizer=word_tokenize, stop_words=stop_words, ngram_range=(1, 1), token_pattern=None)

X_train_bow = bow_counts.fit_transform(X_train)
model1 = LogisticRegression(C=1, solver="liblinear", max_iter=200)
model1.fit(X_train_bow, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        processed_text = preprocess_text(text)
        vectorized_text = bow_counts.transform([processed_text])
        prediction = model1.predict(vectorized_text)[0]
        return render_template('result.html',text=text, prediction=prediction[0])

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters
    text = re.sub('[^A-Za-z0-9 ]+', '', text)
    return text

if __name__ == '__main__':
    app.run(debug=True)
