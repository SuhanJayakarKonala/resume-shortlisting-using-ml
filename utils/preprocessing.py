import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    # Remove special characters, numbers, and punctuation
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    # Remove stopwords and apply stemming
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

def preprocess_dataset(csv_path):
    df = pd.read_csv(csv_path)
    df.dropna(inplace=True)  # Drop rows with missing values
    df['Cleaned_Resume'] = df['Resume'].apply(clean_text)
    return df[['Cleaned_Resume', 'Category']]
