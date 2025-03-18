#!/usr/bin/env python3
from models.model4_dense_layers import create_dense_model
print("Ejecutando Modelo 4: Red Neuronal con múltiples capas densas")
model, tokenizer = create_dense_model()
print("¡Entrenamiento del Modelo 4 completado!")
