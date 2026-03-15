# SVM — Twitter Financial News Sentiment Analysis

## Project Summary

Built a Support Vector Machine (SVM) model to classify financial tweets into negative, positive, and neutral sentiment using a real-world Twitter dataset from Hugging Face. This project introduces NLP (Natural Language Processing) — converting raw text into numbers for machine learning.

**Dataset:** [zeroshot/twitter-financial-news-sentiment](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment)

**Business Problem:** Can we automatically classify the sentiment of financial news tweets — so traders, analysts, and product teams can monitor market sentiment at scale without reading every tweet manually?

---

## What I Built

A multi-class SVM classifier predicting:
- `0` → Negative sentiment
- `1` → Positive sentiment
- `2` → Neutral sentiment

---

## Dataset Overview

```
Total tweets:  9,543
Neutral:       6,178  (65%)
Positive:      1,923  (20%)
Negative:      1,442  (15%)
```

Real financial tweets like:
```
"$BYND - JPMorgan reels in expectations on Beyond Meat"  → Negative
"$AAPL - Apple upgraded at Goldman Sachs"                → Positive
"$TSLA - Tesla reports quarterly earnings"               → Neutral
```

---

## Key Learnings

### 1. What is SVM?

Previous algorithms (logistic regression, decision trees) draw boundaries between classes. SVM finds the **best possible boundary** — the one with maximum distance from both classes.

```
Logistic Regression → draws any line that separates classes
SVM                 → draws the line with MAXIMUM margin from both classes
```

That maximum distance is called the **margin**. The data points closest to the boundary are called **support vectors** — that's where the name comes from.

**Two kernels used in this project:**
- **Linear kernel** — draws a straight boundary (faster, more interpretable)
- **RBF kernel** — draws a curved boundary (handles complex patterns, slower)

### 2. Text Vectorization — TF-IDF

Machine learning models can't read text. Every word must be converted to a number first. Used **TF-IDF (Term Frequency — Inverse Document Frequency)**:

- **TF** — how often a word appears in this tweet
- **IDF** — how unique this word is across all tweets

Words like "cuts", "downgrade", "weak" score high in negative tweets.
Words like "upgrade", "growth", "strong" score high in positive tweets.
Common words like "the", "and", "is" get low scores — they appear everywhere.

```python
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)    # only transform, never fit
```

### 3. Bigrams — Capturing Phrases

Single words (unigrams): "price", "target", "cuts"

Two word phrases (bigrams): "price target", "revenue cuts", "earnings growth"

Bigrams capture more meaning than single words — "price" alone is neutral but "price cuts" is negative.

```python
# Unigrams only
TfidfVectorizer(ngram_range=(1, 1))

# Unigrams + Bigrams
TfidfVectorizer(ngram_range=(1, 2))
```

### 4. Data Leakage in NLP — Split Before Vectorizing

Critical rule: **always split data before fitting TF-IDF.**

```python
# ✅ Correct order
X_train, X_test = train_test_split(X)     # split first
tfidf.fit_transform(X_train)              # fit only on train
tfidf.transform(X_test)                   # only transform test

# ❌ Wrong order — data leakage
tfidf.fit_transform(X)                    # test data influences vocabulary
X_train, X_test = train_test_split(X)     # too late
```

If you fit TF-IDF on the full dataset, test words influence the vocabulary and scores — the model indirectly sees test data during training.

### 5. Full Experiment Results

Systematically tested three improvements:

| Experiment | Config | Accuracy | Neg Recall | Pos Recall | Neu Recall |
|---|---|---|---|---|---|
| Baseline | linear, 5k features, unigrams | 78% | 68% | 71% | 83% |
| Exp 1 | linear, 10k features, unigrams | 79% | 67% | 72% | 85% |
| Exp 2 | linear, 10k features, bigrams | 80% | 69% | 72% | 85% |
| Exp 3 | rbf, 10k features, bigrams | 82% | 56% | 66% | 93% |

**What each change did:**
- **Bigger vocabulary (5k→10k)** — tiny accuracy gain, minimal impact
- **Adding bigrams** — meaningful gain, improved negative recall. Phrases add real signal
- **RBF kernel** — biggest accuracy jump (+2%) but hurt minority class recall significantly

### 6. Higher Accuracy ≠ Better Product

Experiment 3 has the highest accuracy (82%) but the worst negative recall (56%).
Experiment 2 has lower accuracy (80%) but the best negative recall (69%).

**For a financial sentiment product — Experiment 2 is the right model to ship.**

Missing a negative signal about a stock could mean a missed sell signal — that's far more costly than slightly lower overall accuracy.

This is the most important PM insight: **optimize for the metric that matters to the business, not the metric that looks best on paper.**

### 7. Class Imbalance in NLP

```
Neutral:  65% of data → model learned it best (93% recall)
Negative: 15% of data → model struggles most (56-68% recall)
```

Same fix as previous projects:
```python
model = SVC(kernel='linear', class_weight='balanced', random_state=42)
```

### 8. SVM vs Previous Models

| Model | Dataset | Task | Accuracy |
|---|---|---|---|
| Logistic Regression | Shipment delays | Binary classification | 50% |
| Decision Tree | Shipment delays | Binary classification | 56% |
| Random Forest | Shipment delays | Binary classification | 52% |
| **SVM** | **Twitter sentiment** | **3-class NLP** | **80-82%** |

The jump is not because SVM is better — it's because this dataset has **real patterns**. Words genuinely predict sentiment. The shipment dataset was synthetically generated with no real signal.

---

## Final Model Configuration

```python
# TF-IDF Vectorizer
tfidf = TfidfVectorizer(
    max_features=10000,      # vocabulary size
    ngram_range=(1, 2)       # unigrams + bigrams
)

# SVM Model
model = SVC(
    kernel='linear',         # linear boundary
    class_weight='balanced', # handle class imbalance
    random_state=42
)
```

**Final Performance (Experiment 2):**
```
Accuracy:          80%
Negative Recall:   69%
Positive Recall:   72%
Neutral Recall:    85%
```

---

## Tools & Libraries

```python
datasets                              # Hugging Face dataset loading
pandas                                # Data manipulation
numpy                                 # Numerical operations
scikit-learn                          # Model training and evaluation
sklearn.svm.SVC                       # Support Vector Machine
sklearn.feature_extraction.TfidfVectorizer  # Text vectorization
matplotlib                            # Visualisation
python-dotenv                         # Environment variable management
huggingface_hub                       # Authentication
```

---

## How to Run

```bash
# Clone the repo
git clone https://github.com/shambhavichaugule/svm.git
cd svm

# Activate virtual environment
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Add your Hugging Face token to .env
echo "HF_TOKEN=your_token_here" > .env

# Run the model
python svm.py
```

---

## PM Perspective

This project simulates a real product: **a financial sentiment monitoring tool.**

**Who uses this:**
- Traders monitoring market sentiment in real time
- Analysts tracking sentiment around specific stocks
- Product teams building financial news aggregators

**Before building:**
- Define what matters most — catching negative sentiment (risk) vs overall accuracy
- Understand class distribution — 65% neutral means accuracy is misleading
- Validate that text actually contains signal — financial tweets do, synthetic data doesn't

**In production:**
- Monitor sentiment drift — financial language changes over time (new terms, new tickers)
- Retrain regularly on fresh tweets — a model trained on 2024 tweets may miss 2026 language
- Build human review for low confidence predictions — don't fully automate high stakes decisions
- Consider using a pre-trained financial language model (FinBERT) for even better results

**Key insight:** The choice between Experiment 2 (80%) and Experiment 3 (82%) is not a technical decision — it's a product decision about what kind of errors are acceptable. A PM must define this before the data scientist starts tuning.

---

## Next Steps

- Try FinBERT — a pre-trained transformer model specifically trained on financial text
- Add confidence scores to predictions — flag low confidence tweets for human review
- Build a simple dashboard showing sentiment trends over time
- Experiment with ensemble — combine SVM with other models for better minority class recall

---

## Learning Series

| Project | Algorithm | Dataset | Accuracy |
|---|---|---|---|
| [Linear Regression](https://github.com/shambhavichaugule/linear-regression-project) | Linear Regression | Car prices | R²=0.09 |
| [Logistic Regression](https://github.com/shambhavichaugule/logistic-regression-project) | Logistic Regression | Shipment delays | 50% |
| [Decision Tree & Random Forest](https://github.com/shambhavichaugule/decision-tree-random-forest) | DT + RF | Shipment delays | 56% |
| This project | SVM + TF-IDF | Twitter sentiment | 80% |

---