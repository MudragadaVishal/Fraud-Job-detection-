import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap

st.set_page_config(page_title="🕵️ Job Fraud Detector", layout="centered")
st.title("🕵️‍♂️ Job Fraud Detection Dashboard")
st.markdown(
    "Upload a job postings CSV to detect potential **fraudulent listings** using a trained machine learning model."
)

@st.cache_resource
def load_model():
    return joblib.load("fraud_detector.pkl")

@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

model = load_model()
uploaded_file = st.file_uploader("📁 Upload your test CSV file", type=["csv"])

if uploaded_file:
    df = load_data(uploaded_file)
    df['text'] = df['title'].fillna('') + ' ' + df['description'].fillna('')

    if st.button("🚀 Run Prediction"):
        with st.spinner("Analyzing job posts..."):
            df['fraud_probability'] = model.predict_proba(df['text'])[:, 1]
            df['fraud_label'] = df['fraud_probability'].apply(lambda x: 'Fraudulent' if x > 0.5 else 'Legitimate')

        st.success("✅ Prediction complete!")

        st.markdown("### 📊 Prediction Summary")
        col1, col2 = st.columns(2)
        fraud_count = df['fraud_label'].value_counts()
        fraud = fraud_count.get('Fraudulent', 0)
        legit = fraud_count.get('Legitimate', 0)

        col1.metric("Fraudulent Posts", fraud)
        col2.metric("Legitimate Posts", legit)

        fig, ax = plt.subplots()
        ax.pie([fraud, legit], labels=['Fraudulent', 'Legitimate'], autopct='%1.1f%%', startangle=140,
               colors=['#ff4b4b', '#4caf50'])
        ax.set_title("Fraud vs Legitimate Job Posts")
        st.pyplot(fig)

        st.markdown("### 📊 Histogram of Fraud Probabilities")
        fig_hist, ax_hist = plt.subplots()
        ax_hist.hist(df['fraud_probability'], bins=20, color='orange', edgecolor='black')
        ax_hist.set_title("Distribution of Fraud Probabilities")
        ax_hist.set_xlabel("Fraud Probability")
        ax_hist.set_ylabel("Number of Jobs")
        ax_hist.grid(True)
        st.pyplot(fig_hist)

        st.markdown("### 🔍 Top 10 Most Suspicious Job Posts")
        st.dataframe(df.sort_values(by='fraud_probability', ascending=False)[['title', 'fraud_probability', 'fraud_label']].head(10))

        with st.expander("🤖 SHAP Explanation for First Prediction"):
            st.info("This explains why the model predicted the first job as fraudulent or legitimate.")

            try:
                vectorizer = model.named_steps['tfidf']
                classifier = model.named_steps['clf']
                sample_texts = df['text'].tolist()[:50]
                X_sample = vectorizer.transform(sample_texts)

                explainer = shap.LinearExplainer(classifier, X_sample, feature_perturbation="interventional")
                shap_values = explainer(X_sample)

                feature_names = vectorizer.get_feature_names_out()
                top_features = pd.DataFrame({
                    'feature': feature_names,
                    'shap_value': shap_values[0].values.flatten()
                }).sort_values(by='shap_value', ascending=False).head(10)

                fig, ax = plt.subplots()
                ax.barh(top_features['feature'], top_features['shap_value'], color='deepskyblue')
                ax.invert_yaxis()
                ax.set_xlabel("SHAP Value")
                ax.set_title("Top Words Influencing First Prediction")
                st.pyplot(fig)

            except Exception as e:
                st.error(f"SHAP explanation failed: {e}")

        with st.expander("📄 View All Predictions"):
            st.dataframe(df[['title', 'fraud_probability', 'fraud_label']], use_container_width=True)

        csv_out = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Predictions CSV", csv_out, file_name="predictions.csv", mime="text/csv")
        
