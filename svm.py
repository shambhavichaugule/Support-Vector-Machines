import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
import os
from huggingface_hub import login
load_dotenv()
login(token=os.getenv("HF_TOKEN"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
dataset = load_dataset("zeroshot/twitter-financial-news-sentiment")
df = dataset['train'].to_pandas()

# Features and target
X = df['text']
y = df['label']

# Split first — before vectorizing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert text to numbers using TF-IDF
# Fit only on training data — never on test data
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)        # only transform, not fit

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Vocabulary size: {len(tfidf.vocabulary_)}")
print(f"Feature matrix shape: {X_train_tfidf.shape}")

# Train SVM
model = SVC(kernel='linear', class_weight='balanced', random_state=42)
model.fit(X_train_tfidf, y_train)

# Evaluate
y_pred = model.predict(X_test_tfidf)

print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, 
      target_names=['Negative', 'Positive', 'Neutral']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Save chart
plt.figure(figsize=(7, 5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('SVM - Sentiment Analysis Confusion Matrix')
plt.colorbar()
plt.xticks([0, 1, 2], ['Negative', 'Positive', 'Neutral'])
plt.yticks([0, 1, 2], ['Negative', 'Positive', 'Neutral'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig("svm_confusion_matrix.png")
print("\nChart saved!")