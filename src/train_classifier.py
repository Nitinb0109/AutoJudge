import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import os

# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv("data/problems.csv")

X_text = df["text"].fillna("")
y = df["problem_class"]

# ----------------------------
# TF-IDF Vectorization
# ----------------------------
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words="english"
)

X = tfidf.fit_transform(X_text)

# ----------------------------
# Train-test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------
# Train classifier
# ----------------------------
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# ----------------------------
# Evaluation
# ----------------------------
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("✅ Classification Model Trained")
print("Accuracy:", acc)
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ----------------------------
# Save model & vectorizer
# ----------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/classifier.pkl")
joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")

print("\n💾 Classifier and TF-IDF vectorizer saved to models/")
