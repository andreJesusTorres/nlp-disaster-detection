# 🎯 NLP Disaster Detection System

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://tensorflow.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-green.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-Proprietary-red.svg)](LICENSE)

> Advanced Natural Language Processing system for automatic classification of disaster-related tweets using multiple neural network architectures. **This project is part of my professional portfolio to demonstrate my development skills and practices.**

## 📋 Table of Contents

- [✨ Features](#-features)
- [🛠️ Technologies](#️-technologies)
- [📦 Installation](#-installation)
- [🎮 Usage](#-usage)
- [🏗️ Project Structure](#️-project-structure)
- [🧪 Testing](#-testing)
- [📄 License](#-license)

## ✨ Features

### 🎯 Core Functionality
- **Multi-Model Architecture**: 5 different neural network approaches for disaster tweet classification
- **Advanced Text Preprocessing**: Both basic and advanced preprocessing using NLTK and SpaCy
- **Real-time Classification**: Distinguish between real disaster tweets and non-disaster content
- **Performance Visualization**: Automatic generation of training history plots and metrics
- **Reproducible Results**: Docker-based environment for consistent execution

### 🎨 User Experience
- **Modular Design**: Each model can be run independently
- **Comprehensive Documentation**: Detailed analysis of each approach
- **Performance Comparison**: Side-by-side evaluation of different techniques
- **Easy Setup**: Simple installation and execution process

## 🛠️ Technologies

### Core ML Framework
| Technology | Version | Purpose |
|------------|---------|---------|
| [TensorFlow](https://tensorflow.org/) | 2.15+ | Deep learning framework |
| [Scikit-learn](https://scikit-learn.org/) | 1.3+ | Machine learning utilities |
| [NumPy](https://numpy.org/) | 1.24+ | Numerical computing |

### NLP Libraries
| Technology | Version | Purpose |
|------------|---------|---------|
| [NLTK](https://www.nltk.org/) | 3.8+ | Natural language processing |
| [SpaCy](https://spacy.io/) | 3.7+ | Advanced NLP pipeline |
| [Sentence-Transformers](https://www.sbert.net/) | 2.2+ | Pre-trained sentence embeddings |

### Data & Visualization
| Technology | Version | Purpose |
|------------|---------|---------|
| [Pandas](https://pandas.pydata.org/) | 2.0+ | Data manipulation |
| [Matplotlib](https://matplotlib.org/) | 3.7+ | Plotting and visualization |
| [Seaborn](https://seaborn.pydata.org/) | 0.12+ | Statistical data visualization |

### Development Tools
- Docker for reproducible environments
- Jupyter-compatible code structure
- Comprehensive logging and metrics

## 📦 Installation

### Prerequisites
- Python 3.9 or higher
- Docker (optional, for containerized execution)
- 4GB+ RAM for model training

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/nlp-disaster-detection.git
   cd nlp-disaster-detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up SpaCy language model**
   ```bash
   python -m spacy download en_core_web_sm
   ```

4. **Download NLTK data**
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   ```

5. **Run a specific model**
   ```bash
   # Basic tokenization model
   python run_model1.py
   
   # Embeddings-based model
   python run_model2.py
   
   # Sentence-BERT model
   python run_model3_simple.py
   
   # Deep neural network
   python run_model4.py
   
   # Advanced preprocessing model
   python run_model5_simple.py
   ```

6. **View results**
   - Training plots: `results/` directory
   - Model performance metrics displayed in console

## 🎮 Usage

### Getting Started
1. Ensure your dataset is in `data/train.csv` format
2. Choose the model architecture based on your requirements
3. Run the corresponding script
4. Monitor training progress and view generated plots

### Key Features Usage

#### Basic Tokenization Model
```python
from models.model1_tokenizer import create_tokenizer_model

# Create and train model
model, tokenizer = create_tokenizer_model(max_words=10000, max_len=100)
```

#### Embeddings-Based Model
```python
from models.model2_embeddings import create_embeddings_model

# Create and train model with word embeddings
model, tokenizer = create_embeddings_model()
```

#### Sentence-BERT Model
```python
from models.model3_sbert import create_sbert_model

# Create and train model with pre-trained BERT embeddings
model = create_sbert_model()
```

### Model Performance Comparison

| Model | Architecture | Validation Accuracy | Key Features |
|-------|--------------|-------------------|--------------|
| Model 1 | Simple Tokenization | ~50% | Basic text processing |
| Model 2 | Word Embeddings | ~70-80% | Semantic relationships |
| Model 3 | Sentence-BERT | ~75-85% | Pre-trained embeddings |
| Model 4 | Deep Neural Network | ~75-80% | Complex architecture |
| Model 5 | Advanced Preprocessing | ~70-75% | SpaCy-based cleaning |

## 🏗️ Project Structure

```
📁 nlp-disaster-detection/
├── 📁 data/                    # Dataset storage
│   └── 📄 train.csv           # Training dataset
├── 📁 models/                  # Neural network implementations
│   ├── 🧠 model1_tokenizer.py      # Basic tokenization model
│   ├── 🧠 model2_embeddings.py     # Embeddings-based model
│   ├── 🧠 model3_sbert.py          # Sentence-BERT model
│   ├── 🧠 model4_dense_layers.py   # Deep neural network
│   └── 🧠 model5_tokenizer_spacy.py # Advanced preprocessing
├── 📁 results/                 # Generated outputs
│   ├── 📊 model1_training_history.png
│   ├── 📊 model2_training_history.png
│   ├── 📊 model3_sbert_simple.png
│   ├── 📊 model4_dense_layers.png
│   └── 📊 model5_spacy_simple.png
├── 📁 utils/                   # Utility functions
│   ├── 🔧 preprocess.py        # Basic text preprocessing
│   └── 🔧 preprocess_spacy.py  # Advanced preprocessing
├── 🚀 run_model1.py           # Model 1 execution script
├── 🚀 run_model2.py           # Model 2 execution script
├── 🚀 run_model3_simple.py    # Model 3 execution script
├── 🚀 run_model4.py           # Model 4 execution script
├── 🚀 run_model5_simple.py    # Model 5 execution script
├── 📋 requirements.txt        # Python dependencies
└── 📖 README.md              # Project documentation
```

## 🧪 Testing

### Running Models
```bash
# Test all models sequentially
for i in {1..5}; do
  if [ $i -eq 3 ] || [ $i -eq 5 ]; then
    python run_model${i}_simple.py
  else
    python run_model${i}.py
  fi
done
```

### Expected Outputs
- ✅ Training history plots in `results/` directory
- ✅ Console output with accuracy and loss metrics
- ✅ Model performance comparison data

### Performance Validation
- All models achieve >50% validation accuracy
- Training plots show convergence patterns
- No overfitting detected in properly configured models

## 📄 License

This project is proprietary software. All rights reserved. This code is made publicly available solely for portfolio demonstration purposes. See the [LICENSE](LICENSE) file for full terms and restrictions.

---

<div align="center">
  <p>
    <a href="#-nlp-disaster-detection-system">Back to top</a>
  </p>
</div> 