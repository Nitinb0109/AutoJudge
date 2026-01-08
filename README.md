# AutoJudge – Programming Problem Difficulty Predictor
AutoJudge is a machine learning system that predicts the difficulty of competitive programming problems using only their textual description.

It performs two tasks:
- Difficulty Classification → Easy / Medium / Hard
- Difficulty Regression → Numerical difficulty score

---

## 🧾 Dataset
This project uses the TaskComplexityEval-24 dataset.

Each problem contains:
- title
- description
- input_description
- output_description
- problem_class (Easy / Medium / Hard)
- problem_score (Numeric)

The original dataset was in JSONL format and converted to CSV using a preprocessing script.

---

## 🧠 Approach & Models Used
Feature Engineering:
- Combined text (title + description + input + output)
- TF-IDF vectorization
- Text length features
- Keyword frequency

Models Used:
- Classification → Logistic Regression
- Regression → Random Forest Regressor

---

## 📊 Evaluation Metrics & Results
Classification Results:
- Accuracy: 0.47
- Confusion Matrix:
[[ 28  75  50]
 [ 12 294  83]
 [ 17 196  68]]

Regression Results:
- MAE: 1.714
- RMSE: 2.052

---

## 🌐 Web Interface
A Streamlit web app allows users to paste a new problem description and receive:
✔ Predicted difficulty class  
✔ Predicted difficulty score  

User Inputs:
- Problem Title
- Problem Description
- Input Format
- Output Format

Outputs:
- Difficulty Class
- Difficulty Score

---

## 🚀 How to Run Locally
1. Install dependencies:
pip install -r requirements.txt

2. Run preprocessing:
python src/preprocess.py

3. Train models:
python src/train_classifier.py
python src/train_regressor.py

4. Run the web app:
streamlit run app.py

---

## 📁 Project Structure
AutoJudge/
├── data/
│   ├── problems_data.jsonl
│   └── problems.csv
├── models/
│   ├── classifier.pkl
│   ├── regressor.pkl
│   ├── tfidf_vectorizer.pkl
│   └── tfidf_vectorizer_reg.pkl
├── src/
│   ├── preprocess.py
│   ├── train_classifier.py
│   ├── train_regressor.py
│   └── features.py
├── app.py
├── requirements.txt
└── README.md

---

## 🎥 Demo Video (Mandatory)
Google Drive Video Link:
https://drive.google.com/file/d/1F4T_IKW39_ml8LmQzJ6jYsDtTOtYuTJi/view?usp=drivesdkC

---
## Project report
Google Drive Pdf Link:
https://drive.google.com/file/d/1WLgH-NlSglhaWN2C4_GILWOGqfM_xJdq/view?usp=drivesdk

## 📝 Notes
- The project runs locally without hosting
- Models are saved automatically after training
- No manual dataset labeling required
