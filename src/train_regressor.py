import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os

# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv("data/problems.csv")

X_text = df["text"].fillna("")
y = df["problem_score"]

# Convert to numeric safely
y = pd.to_numeric(y, errors="coerce")
df = df.loc[y.notna()]
X_text = X_text.loc[df.index]
y = y.loc[df.index]

# ----------------------------
# TF-IDF
# ----------------------------
tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X = tfidf.fit_transform(X_text)

# ----------------------------
# Train-test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# Train regressor
# ----------------------------
reg = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

reg.fit(X_train, y_train)

# ----------------------------
# Evaluation
# ----------------------------
y_pred = reg.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5


print("✅ Regression Model Trained")
print("MAE:", mae)
print("RMSE:", rmse)

# ----------------------------
# Save model & vectorizer
# ----------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(reg, "models/regressor.pkl")
joblib.dump(tfidf, "models/tfidf_vectorizer_reg.pkl")

print("\n💾 Regressor and TF-IDF vectorizer saved to models/")
