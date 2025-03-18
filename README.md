# Clasificación de Tweets sobre Desastres Naturales
## Trabajo Práctico - Procesamiento de Lenguaje Natural

**Alumno**: Ricardo
**Asignatura**: Procesamiento de Lenguaje Natural  
**Fecha de entrega**: Mayo 2024

## 1. Introducción

El presente trabajo aborda la tarea de clasificación automática de tweets relacionados con desastres. El objetivo principal es desarrollar modelos que puedan distinguir entre tweets que mencionan desastres reales y aquellos que utilizan lenguaje similar pero no se refieren a eventos catastróficos reales. Esta capacidad es relevante para sistemas de detección temprana y monitoreo de situaciones de emergencia en redes sociales.

## 2. Objetivos

- Implementar diferentes arquitecturas de redes neuronales para la clasificación de texto
- Comparar el rendimiento de distintas técnicas de representación textual
- Evaluar el impacto del preprocesamiento en la calidad de los modelos
- Identificar los enfoques más efectivos para esta tarea específica

## 3. Metodología

### 3.1 Dataset

Se utilizó el conjunto de datos de la competición "Natural Language Processing with Disaster Tweets" de Kaggle, que contiene tweets etiquetados como relacionados con desastres reales (1) o no (0).

Fuente: https://www.kaggle.com/competitions/nlp-getting-started/data

### 3.2 Preprocesamiento

Se implementaron dos niveles de preprocesamiento:
- **Básico**: Conversión a minúsculas, eliminación de URLs, menciones, caracteres especiales y espacios redundantes
- **Avanzado**: Utilizando la biblioteca Spacy para lematización y eliminación de stopwords

### 3.3 Modelos Implementados

Se desarrollaron cinco modelos con diferentes enfoques:

**Modelo 1: Red Neuronal con Tokenización Simple**
- Tokenización de palabras y transformación a secuencias numéricas
- Arquitectura: Capa de entrada → Capa densa (64) → Capa densa (32) → Salida

**Modelo 2: Red Neuronal con Embeddings**
- Representación vectorial de palabras mediante embeddings
- Arquitectura: Embedding → GlobalAveragePooling1D → Capa densa (64) → Capa densa (32) → Salida

**Modelo 3: Sentence-BERT Preentrenado**
- Utilización de modelos preentrenados para la codificación semántica de tweets
- Arquitectura: SBERT → Capa densa (64) → Dropout (0.2) → Capa densa (32) → Salida

**Modelo 4: Red Neuronal Profunda**
- Arquitectura más compleja con capas adicionales y regularización mediante dropout
- Estructura: Embedding → GlobalAveragePooling1D → Capa densa (128) → Dropout (0.3) → Capa densa (64) → Dropout (0.2) → Capa densa (32) → Capa densa (16) → Salida

**Modelo 5: Red Neuronal con Preprocesamiento Avanzado**
- Utilización de Spacy para preprocesamiento lingüístico avanzado
- Misma arquitectura que el Modelo 1 pero con mejor representación textual

## 4. Implementación

### 4.1 Estructura del Proyecto

```
├── data/               # Datos de entrenamiento
├── models/             # Implementaciones de las arquitecturas
├── results/            # Visualizaciones y métricas
├── utils/              # Funciones de preprocesamiento
├── run_model1-5.py     # Scripts de ejecución
└── requirements.txt    # Dependencias del proyecto
```

### 4.2 Instrucciones de Ejecución

El código se desarrolló para ser ejecutado en un entorno Docker, asegurando reproducibilidad:

```bash
# Iniciar contenedor Docker
docker run -it -v "$(pwd):/workspace" python:3.9-slim bash

# Instalar dependencias
cd /workspace
pip install -r requirements.txt

# Ejecutar modelos específicos
python run_model1.py  # Modelo con tokenización
python run_model2.py  # Modelo con embeddings
python run_model4.py  # Modelo con arquitectura profunda
```

Para los modelos 3 y 5 se requieren dependencias adicionales:
```bash
pip install sentence-transformers spacy
python -m spacy download es_core_news_sm
```

## 5. Resultados y Discusión

### 5.1 Comparativa de Rendimiento

| Modelo | Precisión (Validación) | Características Destacables |
|--------|----------------------|---------------------------|
| 1. Tokenización Simple | ~50% | Alta pérdida, rendimiento limitado |
| 2. Embeddings | ~70-80% | Mejor captura de relaciones semánticas |
| 4. Arquitectura Profunda | ~75-80% | Mayor resistencia al sobreajuste |

### 5.2 Observaciones Clave

1. **Representación vectorial vs. tokenización simple**: Los modelos basados en embeddings (Modelo 2) superaron consistentemente al enfoque de tokenización simple (Modelo 1), demostrando la importancia de las representaciones semánticas.

2. **Importancia de la arquitectura**: La incorporación de capas adicionales y mecanismos de regularización (Modelo 4) contribuyó a una mejor generalización, reduciendo el sobreajuste.

3. **Impacto del preprocesamiento**: La limpieza adecuada del texto resultó fundamental para todos los modelos. El preprocesamiento avanzado con Spacy (Modelo 5) optimizó la calidad de la representación textual.

Las visualizaciones detalladas del entrenamiento se encuentran disponibles en la carpeta `results/`, donde se puede observar la evolución de la precisión y pérdida a lo largo de las épocas.

## 6. Conclusiones

Este trabajo ha permitido explorar y comparar diferentes enfoques para la clasificación de tweets sobre desastres, destacando la superioridad de los modelos basados en embeddings y arquitecturas más complejas. Los resultados sugieren que:

1. La representación semántica mediante embeddings captura mejor las relaciones entre palabras que los métodos de tokenización simple.

2. Las arquitecturas con mayor profundidad y mecanismos de regularización permiten una mejor generalización.

3. El preprocesamiento lingüístico constituye un factor determinante en el rendimiento de los modelos.

## 7. Referencias

- Kaggle. (2023). Natural Language Processing with Disaster Tweets. https://www.kaggle.com/competitions/nlp-getting-started/
- Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. arXiv preprint arXiv:1908.10084. 