# app.py

from flask import Flask, render_template, request
import pickle
import string
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords
nltk.download('stopwords')

# Initialize Flask App
app = Flask(__name__)

# Load Saved Model
model = pickle.load(open('fake_news_model.pkl', 'rb'))

# Load Saved TF-IDF Vectorizer
vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# Stopwords and Stemmer
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()


# Text Cleaning Function
def clean_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>+', '', text)

    # Remove punctuation
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)

    # Remove numbers
    text = re.sub(r'\w*\d\w*', '', text)

    # Tokenization
    words = text.split()

    # Remove stopwords and apply stemming
    cleaned_words = []

    for word in words:
        if word not in stop_words:
            cleaned_words.append(stemmer.stem(word))

    return " ".join(cleaned_words)


# Home Page
@app.route('/')
def home():
    return render_template('index.html')


# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():

    # Get user input
    news_text = request.form['news']

    # Clean text
    cleaned_news = clean_text(news_text)

    # Vectorize text
    vectorized_news = vectorizer.transform([cleaned_news])

    # Predict
    prediction = model.predict(vectorized_news)

    # Result
    if prediction[0] == 0:
        result = "Fake News"
    else:
        result = "Real News"

    return render_template('index.html', prediction=result)


# Run Flask App
if __name__ == '__main__':
    app.run(debug=True)