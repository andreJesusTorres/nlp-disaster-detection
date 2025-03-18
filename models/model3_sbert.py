import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sentence_transformers import SentenceTransformer
import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocess import preprocess_data

def create_sbert_model(pretrained_model='distiluse-base-multilingual-cased-v1'):
    train_data, val_data = preprocess_data()
    
    sbert_model = SentenceTransformer(pretrained_model)
    
    X_train = sbert_model.encode(train_data['cleaned_text'].tolist())
    X_val = sbert_model.encode(val_data['cleaned_text'].tolist())
    
    y_train = train_data['target'].values
    y_val = val_data['target'].values
    
    input_shape = X_train.shape[1]
    
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dropout(0.2),
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
    plt.savefig(f'results/model3_sbert_{pretrained_model.replace("-", "_")}.png')
    plt.close()
    
    return model, sbert_model

def compare_sbert_models():
    pretrained_models = [
        'distiluse-base-multilingual-cased-v1',
        'paraphrase-multilingual-MiniLM-L12-v2',
        'paraphrase-multilingual-mpnet-base-v2'
    ]
    
    results = {}
    
    for model_name in pretrained_models:
        print(f"Entrenando con modelo preentrenado: {model_name}")
        model, sbert_model = create_sbert_model(model_name)
        
        train_data, val_data = preprocess_data()
        X_val = sbert_model.encode(val_data['cleaned_text'].tolist())
        y_val = val_data['target'].values
        
        val_loss, val_acc = model.evaluate(X_val, y_val)
        results[model_name] = {'loss': val_loss, 'accuracy': val_acc}
        
        print(f"Modelo {model_name}: Precisión de validación = {val_acc:.4f}")
    
    return results

if __name__ == "__main__":
    results = compare_sbert_models()
    print("\nComparativa de modelos SBERT:")
    for model_name, metrics in results.items():
        print(f"{model_name}: Precisión = {metrics['accuracy']:.4f}, Pérdida = {metrics['loss']:.4f}") 