import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import spacy
import re

def load_spacy_model():
    try:
        return spacy.load("es_core_news_sm")
    except:
        spacy.cli.download("es_core_news_sm")
        return spacy.load("es_core_news_sm")

nlp = load_spacy_model()

def clean_text_spacy(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    
    doc = nlp(text)
    
    tokens = []
    for token in doc:
        if (not token.is_stop and 
            not token.is_punct and 
            not token.is_space and 
            not token.like_num and
            not token.is_digit and
            not token.like_url and
            not token.like_email):
            tokens.append(token.lemma_)
    
    return " ".join(tokens)

def preprocess_data_spacy():
    df = pd.read_csv('data/train.csv')
    df['cleaned_text'] = df['text'].apply(clean_text_spacy)
    train_data, val_data = train_test_split(df, test_size=0.2, random_state=42)
    return train_data, val_data 