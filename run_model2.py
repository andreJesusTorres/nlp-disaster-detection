#!/usr/bin/env python3
from models.model2_embeddings import create_embeddings_model
print("Ejecutando Modelo 2: Red Neuronal Feedforward con Embeddings")
model, tokenizer = create_embeddings_model()
print("¡Entrenamiento del Modelo 2 completado!")
