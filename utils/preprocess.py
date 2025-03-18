import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

# Descarga los datos necesarios de NLTK
nltk.download('punkt')
nltk.download('stopwords')

def clean_text(text):
    # Convertir a minúsculas
    text = str(text).lower()
    # Eliminar URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Eliminar menciones de usuario
    text = re.sub(r'\@\w+|\#', '', text)
    # Eliminar caracteres especiales y números
    text = re.sub(r'[^\w\s]', '', text)
    # Eliminar espacios en blanco adicionales
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_data():
    # Cargar el conjunto de datos
    df = pd.read_csv('data/train.csv')
    
    # Limpiar el texto
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Dividir los datos
    train_data, val_data = train_test_split(df, test_size=0.2, random_state=42)
    
    return train_data, val_data

def get_vocabulary_size(texts, max_words=10000):
    # Crear un conjunto de palabras únicas
    words = set()
    for text in texts:
        words.update(word_tokenize(text))
    
    # Devolver el mínimo entre el número de palabras únicas y max_words
    return min(len(words), max_words) 