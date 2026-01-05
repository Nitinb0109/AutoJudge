# AutoJudge – Programming Problem Difficulty Predictor

AutoJudge is a machine learning project that predicts the difficulty of programming problems
using only their textual description.

The system predicts:
- Difficulty Class: Easy / Medium / Hard
- Difficulty Score: Numerical value

---

## Dataset

This project uses the TaskComplexityEval-24 dataset.
The dataset contains programming problems with difficulty labels and scores.

Each problem includes:
- title
- description
- input_description
- output_description
- problem_class
- problem_score

The dataset was originally in JSONL format and converted to CSV using a preprocessing script.

---

## Features

- Combined text (title + description + input + output)
- TF-IDF features
- Text length
- Word count
- Keyword frequency

---

## Models

Classification Model:
- Logistic Regression
- Predicts Easy / Medium / Hard

Regression Model:
- Random Forest Regressor
- Predicts numerical difficulty score

---

## Evaluation

Classification:
- Accuracy
- Confusion Matrix

Regression:
- MAE
- RMSE

---

## Web Interface

A Streamlit web app allows users to enter a problem description and get:
- Predicted difficulty class
- Predicted difficulty score

---

## How to Run

Install dependencies:
pip install -r requirements.txt

Run preprocessing:
python src/preprocess.py

Train models:
python src/train_classifier.py
python src/train_regressor.py

Run web app:
streamlit run app.py

---

## Project Structure

AutoJudge/
- data/
- models/
- src/
- app.py
- requirements.txt
- README.md

---

## Author

Nitin Bansiya
