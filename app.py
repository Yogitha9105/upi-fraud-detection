import streamlit as st
import pandas as pd
import pickle

# Load files
model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))
accuracy = pickle.load(open("accuracy.pkl", "rb"))

# UI Config
st.set_page_config(page_title="UPI Fraud Detection", layout="wide")

st.title("💳 UPI Fraud Detection System")

st.markdown("### 📊 Model Accuracy")
st.success(f"Accuracy: {accuracy:.2f}")

uploaded_file = st.file_uploader("Upload transaction CSV file")

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Data")
    st.dataframe(data.head())

    # Drop unwanted
    if 'nameOrig' in data.columns:
        data = data.drop(['nameOrig', 'nameDest'], axis=1)

    # Convert categorical
    data = pd.get_dummies(data)

    # Align columns
    data = data.reindex(columns=columns, fill_value=0)

    # Predictions
    predictions = model.predict(data)
    probabilities = model.predict_proba(data)[:, 1]

    data['Fraud Prediction'] = predictions
    data['Fraud Probability'] = probabilities

    st.subheader("🔍 Prediction Results")
    st.dataframe(data)

    # Summary metrics
    total = len(data)
    fraud = (predictions == 1).sum()
    normal = (predictions == 0).sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", total)
    col2.metric("Fraud Detected", fraud)
    col3.metric("Normal Transactions", normal)

    # Chart
    st.subheader("📊 Fraud vs Normal")
    st.bar_chart(data['Fraud Prediction'].value_counts())