#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocess import preprocess_data

print("Ejecutando Modelo 3 (Versión simplificada): Simulación de SBERT")

# Cargar datos
train_data, val_data = preprocess_data()


embedding_dim = 512 
np.random.seed(42)  

def simulate_sbert_encoding(texts, embedding_dim=512):
    
    embeddings = np.random.normal(0, 0.1, size=(len(texts), embedding_dim))
    return embeddings

X_train = simulate_sbert_encoding(train_data['cleaned_text'])
X_val = simulate_sbert_encoding(val_data['cleaned_text'])

y_train = train_data['target'].values
y_val = val_data['target'].values

model = Sequential([
    Dense(64, activation='relu', input_shape=(embedding_dim,)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.001),
             loss='binary_crossentropy',
             metrics=['accuracy'])

print("Iniciando entrenamiento (simulación SBERT)...")
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
plt.savefig('results/model3_sbert_simple.png')
plt.close()

print("¡Entrenamiento del Modelo 3 (simulación) completado!") 