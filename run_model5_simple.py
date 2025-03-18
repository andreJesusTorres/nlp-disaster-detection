#!/usr/bin/env python3
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import numpy as np
import re
import sys
import os
import matplotlib.pyplot as plt

print("Ejecutando Modelo 5 (Versión simplificada): Tokenización con Preprocesamiento Avanzado")

def advanced_clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    common_words = ['the', 'a', 'an', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
                    'when', 'where', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
                    'most', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
                    'so', 'than', 'too', 'very', 'can', 'will', 'just', 'should', 'now']
    
    words = text.split()
    filtered_words = [w for w in words if w not in common_words]
    text = ' '.join(filtered_words)
    
    return text

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/train.csv')
df['cleaned_text'] = df['text'].apply(advanced_clean_text)
train_data, val_data = train_test_split(df, test_size=0.2, random_state=42)

max_words = 10000
max_len = 100

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(train_data['cleaned_text'])

X_train = tokenizer.texts_to_sequences(train_data['cleaned_text'])
X_val = tokenizer.texts_to_sequences(val_data['cleaned_text'])

X_train = pad_sequences(X_train, maxlen=max_len)
X_val = pad_sequences(X_val, maxlen=max_len)

y_train = train_data['target'].values
y_val = val_data['target'].values

model = Sequential([
    Dense(64, activation='relu', input_shape=(max_len,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.001),
             loss='binary_crossentropy',
             metrics=['accuracy'])

print("Iniciando entrenamiento (con preprocesamiento avanzado)...")
history = model.fit(X_train, y_train,
                   epochs=10,
                   batch_size=32,
                   validation_data=(X_val, y_val))

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Pérdida de Entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida de Validación')
plt.title('Pérdida del Modelo')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Precisión de Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión de Validación')
plt.title('Precisión del Modelo')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()

plt.tight_layout()
plt.savefig('results/model5_spacy_simple.png')
plt.close()

print("¡Entrenamiento del Modelo 5 (simplificado) completado!") 