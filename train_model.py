# train_model.py

import pandas as pd
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

# -------- Get user input for training file --------
train_file = input("Enter training CSV filename (e.g., Training.csv): ").strip()

# -------- Load training data --------
if not os.path.exists(train_file):
    raise FileNotFoundError(f"Training file '{train_file}' not found.")

print("Loading training data...")
train_df = pd.read_csv(train_file)
train_df['text'] = train_df['title'].fillna('') + ' ' + train_df['description'].fillna('')
train_df = train_df.dropna(subset=['fraudulent'])

X = train_df['text']
y = train_df['fraudulent']

# -------- Split for evaluation --------
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# -------- Build and train model --------
print("Training model...")
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=10000)),
    ('clf', LogisticRegression(class_weight='balanced', max_iter=1000))
])

pipeline.fit(X_train, y_train)

# -------- Evaluate on validation set --------
print("\nEvaluating on validation data:")
y_pred = pipeline.predict(X_val)
print(classification_report(y_val, y_pred))
print(f"F1 Score: {f1_score(y_val, y_pred):.4f}")

# -------- Save model --------
joblib.dump(pipeline, "fraud_detector.pkl")
print("\nModel trained and saved as 'fraud_detector.pkl'")
