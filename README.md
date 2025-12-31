# Toxic-Comment-Classification


# Toxic Comment Classification Challenge

A comprehensive machine learning solution for detecting toxic comments using the Jigsaw Toxic Comment Classification dataset. This project emphasizes thorough exploratory data analysis, advanced text preprocessing, and neural network-based binary classification.

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Text Preprocessing Pipeline](#text-preprocessing-pipeline)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Future Work](#future-work)

## Overview

This project tackles the challenge of identifying toxic online comments across multiple categories (toxic, severe_toxic, obscene, threat, insult, identity_hate). The implementation focuses on:

- **Comprehensive EDA** with multiple visualization techniques
- **Advanced text cleaning** to handle real-world messy data
- **Class balancing** to address severe dataset imbalance
- **Neural network architecture** optimized for binary classification

## Dataset

**Source:** [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/datasets/julian3833/jigsaw-toxic-comment-classification-challenge)

**Statistics:**
- Total comments: ~160,000
- Class distribution: **90.4% non-toxic**, **9.6% toxic**
- 6 toxicity categories (multi-label)
- Highly imbalanced dataset

**Category Distribution:**
1. Toxic: 15,294 (9.6%)
2. Obscene: 8,449 (5.3%)
3. Insult: 7,877 (4.9%)
4. Severe Toxic: 1,595 (1.0%)
5. Identity Hate: 1,405 (0.9%)
6. Threat: 478 (0.3%)

## Features

### 1. Exploratory Data Analysis
- **Distribution Analysis**: Visual representation of toxic vs non-toxic comments
- **Category Frequency**: Bar charts showing individual toxicity type prevalence
- **Co-occurrence Matrix**: Heatmap revealing label correlations
- **UpSet Plot**: Intersection analysis of multi-label combinations
- **Multi-label Statistics**: Analysis of comments with multiple toxicity labels

### 2. Interactive Sample Exploration
Users can:
- Browse comments with multiple toxic labels
- View full text alongside all assigned labels
- Understand real-world examples of toxicity

### 3. Comprehensive Preprocessing
Advanced text cleaning pipeline (see details below)

### 4. Class Balancing
- Undersampling of majority class (non-toxic)
- Stratified train-test split
- Balanced dataset: 50/50 toxic/non-toxic ratio

## Text Preprocessing Pipeline

The cleaning function applies **9 sophisticated preprocessing steps**:

### 1. **Lowercase Conversion**
```python
text = str(text).lower()
```
Standardizes all text to lowercase for consistency.

### 2. **Unicode Normalization & ASCII Encoding**
```python
text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode()
```
- Converts Unicode characters to their closest ASCII equivalents
- Removes emojis, special symbols, and non-Latin characters
- Example: "café" → "cafe", "😊" → ""

### 3. **Word Repetition Removal**
```python
text = re.sub(r'\b(\w{3,})\b(?:\1\b)+', r'\1', text)
```
- Removes repeated words (3+ characters)
- Example: "stop stop stop" → "stop"
- Handles spam-like patterns

### 4. **Laughter Normalization**
```python
text = re.sub(r'(ha){3,}', r'ha', text)
```
- Standardizes excessive laughter
- Example: "hahahahahaha" → "ha"

### 5. **Character Repetition Normalization**
```python
text = re.sub(r'(.)\1{2,}', r'\1\1', text)
```
- Reduces excessive character repetition to max 2 occurrences
- Example: "loooooool" → "lool", "fuuuuuck" → "fuuck"
- Preserves emotional emphasis while normalizing length

### 6. **Contraction Expansion**
```python
contractions = {
    r"what's": "what is",
    r"can't": "cannot",
    r"won't": "will not",
    # ... 15+ contractions
}
```
Expands common English contractions for consistency:
- "don't" → "do not"
- "I'm" → "i am"
- "shouldn't" → "should not"

### 7. **URL & Email Removal**
```python
text = re.sub(r'http\S+|www\S+|\S+@\S+', '', text)
```
Removes web links and email addresses (not useful for toxicity detection)

### 8. **Non-alphabetic Character Removal**
```python
text = re.sub(r'[^a-z\s]', ' ', text)
```
- Keeps only letters and spaces
- Removes numbers, punctuation, special characters

### 9. **Stopword Removal & Lemmatization**
```python
words = [
    lemmatizer.lemmatize(w)
    for w in text.split()
    if w not in stop_words and (len(w) > 2 and len(w) < 22)
]
```
- **Removes stopwords**: "the", "is", "at", etc. (NLTK English stopwords)
- **Lemmatization**: Converts words to base form ("running" → "run")
- **Length filtering**: Keeps words between 3-21 characters (removes noise)

### Cleaning Example

**Original:**
```
"What's up?? This is sooooo amazing!!! 😊 Check www.example.com hahahahaha"
```

**After Cleaning:**
```
"amazing check"
```

**Why this matters:**
- Reduces vocabulary size from 100K+ to 5,000 meaningful features
- Removes noise that doesn't contribute to toxicity detection
- Normalizes variations of the same semantic meaning

## Model Architecture

### Bag-of-Words Representation
- **Vectorizer**: `CountVectorizer`
- **Max Features**: 5,000 most frequent words
- **Output**: Sparse matrix → Dense array

### Neural Network
```
Input (5000 features)
    ↓
Dense(256) + ReLU + BatchNorm + Dropout(0.4)
    ↓
Dense(128) + ReLU + BatchNorm + Dropout(0.3)
    ↓
Dense(64) + ReLU + Dropout(0.2)
    ↓
Dense(1) + Sigmoid
```

**Key Components:**
- **Batch Normalization**: Stabilizes training
- **Dropout Layers**: Prevents overfitting (0.2-0.4)
- **Optimizer**: Adam (lr=0.001)
- **Loss**: Binary Crossentropy
- **Callbacks**: 
  - Early Stopping (patience=5)
  - ReduceLROnPlateau (factor=0.5, patience=3)

## 📈 Results

### Performance Metrics
- **Test Accuracy**: 87.37%
- **Test Loss**: 0.3044
- **Training Time**: 9 epochs (early stopping)
- **Convergence**: Smooth learning curve with no overfitting

### Training History
| Epoch | Train Acc | Val Acc | Train Loss | Val Loss |
|-------|-----------|---------|------------|----------|
| 1     | 67.64%    | 63.42%  | 0.6073     | 0.5918   |
| 4     | 94.87%    | 88.15%  | 0.1407     | 0.2856   |
| 8     | 97.93%    | 89.40%  | 0.0569     | 0.3149   |

The model achieved **88.4% validation accuracy by epoch 4**, with early stopping preventing overfitting.

## Installation

### Requirements
```bash
pip install pandas numpy matplotlib seaborn nltk scikit-learn tensorflow upsetplot
```

### NLTK Data
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

## Usage

### 1. Load and Explore Data
```python
train_df = pd.read_csv('train.csv')
# Run EDA cells to visualize distributions
```

### 2. Preprocess Text
```python
train_df['clean_text'] = train_df['comment_text'].apply(clean_text)
```

### 3. Balance Classes
```python
balanced_df = create_balanced_dataset(train_df)
```

### 4. Train Model
```python
X_bow = vectorizer.fit_transform(balanced_df['clean_text']).toarray()
history = nn_model.fit(X_train, y_train, ...)
```

### 5. Evaluate
```python
test_loss, test_accuracy = nn_model.evaluate(X_test, y_test)
```

## Future Work

### Planned Enhancements
1. **sklearn vs Keras Preprocessing Comparison**
   - Compare `CountVectorizer` vs custom tokenization
   - Benchmark preprocessing pipelines

2. **Feature Engineering Exploration**
   - TF-IDF weighting
   - N-grams (bigrams, trigrams)
   - Word embeddings (Word2Vec, GloVe)
   - Pre-trained transformers (BERT, RoBERTa)

3. **Multi-label Classification**
   - Predict all 6 toxicity categories simultaneously
   - Use sigmoid activation for each output
   - Compare with binary relevance approach

4. **Advanced Architectures**
   - LSTM/GRU for sequence modeling
   - CNN for local pattern detection
   - Attention mechanisms
   - Ensemble methods

5. **Hyperparameter Optimization**
   - Grid search for optimal architecture
   - Learning rate scheduling experiments
   - Regularization tuning

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional preprocessing techniques
- Alternative model architectures
- Evaluation metrics (precision, recall, F1, ROC-AUC)
- Cross-validation implementation

## 📧 Contact

Feel free to reach out for questions or collaboration opportunities!
