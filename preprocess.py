"""
Text Preprocessing Module for Fake News Detection
Handles cleaning, tokenization, and normalization of text data
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

# Initialize components
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def clean_text(text):
    """
    Perform basic text cleaning
    - Convert to lowercase
    - Remove HTML tags
    - Remove URLs
    - Remove special characters
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def tokenize(text):
    """
    Tokenize text into words
    """
    tokens = word_tokenize(text)
    return tokens


def remove_stopwords(tokens):
    """
    Remove common English stopwords
    """
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return filtered_tokens


def lemmatize(tokens):
    """
    Lemmatize tokens to their base form
    """
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized


def preprocess_text(text):
    """
    Complete preprocessing pipeline:
    1. Clean text
    2. Tokenize
    3. Remove stopwords
    4. Lemmatize
    5. Return as string
    """
    # Clean
    cleaned = clean_text(text)
    
    # Tokenize
    tokens = tokenize(cleaned)
    
    # Remove stopwords
    tokens = remove_stopwords(tokens)
    
    # Lemmatize
    tokens = lemmatize(tokens)
    
    # Join back to string
    processed_text = ' '.join(tokens)
    
    return processed_text


def preprocess_dataframe(df, text_columns=['title', 'text']):
    """
    Preprocess all text columns in a dataframe
    
    Args:
        df: Input dataframe
        text_columns: List of column names to preprocess
    
    Returns:
        Dataframe with preprocessed text
    """
    df_copy = df.copy()
    
    for column in text_columns:
        if column in df_copy.columns:
            print(f"Preprocessing '{column}' column...")
            df_copy[column] = df_copy[column].apply(preprocess_text)
    
    return df_copy


def combine_text(row, text_columns=['title', 'text']):
    """
    Combine multiple text columns into one
    """
    combined = ' '.join([str(row[col]) for col in text_columns if col in row])
    return combined


if __name__ == "__main__":
    # Test preprocessing
    sample_text = "Check out this AMAZING news!!! Visit http://example.com for more INFO!!! #FakeNews"
    print("Original text:", sample_text)
    print("Cleaned text:", clean_text(sample_text))
    print("Processed text:", preprocess_text(sample_text))
