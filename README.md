
# 🕵️ Fraud Job Detection System

A machine learning project to detect fraudulent job postings. This system includes a training script to build the model and an interactive Streamlit dashboard for real-time predictions and visual explanations.

---

## 📌 Project Overview

This project aims to identify potentially **fraudulent job listings** using natural language processing (NLP) and logistic regression. The dashboard allows users to upload a CSV of job data and receive predictions, along with SHAP explanations for model interpretability.

---

## 🚀 Key Features & Technologies Used

- **Technologies**:  
  - Python, Streamlit, scikit-learn, pandas, matplotlib, SHAP, joblib

- **Key Features**:  
  - CSV upload and batch prediction  
  - Visualizations: Pie chart, histogram, and SHAP-based bar chart  
  - Model explainability using SHAP values  
  - Downloadable predictions CSV

---

## 🛠️ Setup Instructions

Follow these steps to run the project locally:

### 1. Clone the repository
```bash
git clone https://github.com/MudragadaVishal/job-fraud-detector.git
cd job-fraud-detector
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model (Optional)
If you don’t have the pre-trained model file:
```bash
python train_model.py
```
- Provide the training CSV filename (e.g., `Training.csv`) when prompted.
- The model will be saved as `fraud_detector.pkl`.

### 4. Run the Streamlit app
```bash
streamlit run test.py
```

### 5. Use the app
- Upload your **test CSV file** containing job listings.
- Click **"Run Prediction"** to analyze the data.
- View interactive visualizations and download results.

---

## 📁 Files Description

| File | Description |
|------|-------------|
| `train_model.py` | Script to train the fraud detection model using job title and description |
| `test.py` | Streamlit dashboard for predicting and visualizing fraudulent listings |
| `fraud_detector.pkl` | Saved machine learning model (auto-generated after training) |

---

## 📬 Contact

For queries or collaboration:
**M Vishal**  
🔗 [LinkedIn](https://linkedin.com/in/mudragada-vishal-43039a172) | [GitHub](https://github.com/MudragadaVishal)
