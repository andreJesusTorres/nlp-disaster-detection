import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocess import preprocess_data, get_vocabulary_size
import matplotlib.pyplot as plt

def create_tokenizer_model(max_words=10000, max_len=100):
    train_data, val_data = preprocess_data()
    
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
    plt.savefig('results/model1_training_history.png')
    plt.close()
    
    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = create_tokenizer_model()
    print("¡Entrenamiento del Modelo 1 completado!") 