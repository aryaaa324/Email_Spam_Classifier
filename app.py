import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
from datetime import datetime

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Log file setup
LOG_FILE = "spam_logs.csv"
if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=["timestamp", "input", "prediction", "confidence"]).to_csv(LOG_FILE, index=False)

# Streamlit setup
st.set_page_config(page_title="📧 Email/SMS Spam Classifier", layout="centered")
st.title("📩 Email/SMS Spam Classifier")
st.markdown("A simple ML app that classifies your messages as Spam or Not Spam.")

def classify_text(text):
    vect = vectorizer.transform([text])
    pred = model.predict(vect)[0]
    proba = model.predict_proba(vect)[0]

    log = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input": text,
        "prediction": "Spam" if pred else "Not Spam",
        "confidence": f"{max(proba)*100:.2f}%"
    }
    pd.DataFrame([log]).to_csv(LOG_FILE, mode='a', index=False, header=False)

    return pred, proba

# --- Text Input UI ---
st.subheader("🔍 Single Message Check")
user_input = st.text_area("Enter your email or SMS text:")

if st.button("Check for Spam"):
    if user_input.strip():
        pred, proba = classify_text(user_input)
        label = "🛑 Spam" if pred else "✅ Not Spam"
        st.markdown(f"### Prediction: **{label}**")
        st.write(f"Confidence: {max(proba) * 100:.2f}%")

        # Probability Bar Chart
        fig, ax = plt.subplots()
        ax.bar(["Not Spam", "Spam"], proba * 100, color=["green", "red"])
        ax.set_ylabel("Confidence (%)")
        ax.set_ylim(0, 100)
        st.pyplot(fig)
    else:
        st.warning("Please enter some text!")

# --- Upload File UI ---
st.subheader("📂 Batch Check (.txt File)")
uploaded_file = st.file_uploader("Upload a .txt file", type="txt")

if uploaded_file is not None:
    content = uploaded_file.read().decode("utf-8").split('\n')
    results = []

    for line in content:
        if line.strip():
            pred, proba = classify_text(line)
            results.append({
                "Message": line.strip(),
                "Prediction": "Spam" if pred else "Not Spam",
                "Confidence": f"{max(proba)*100:.2f}%"
            })

    st.success("Batch classification complete!")
    st.dataframe(pd.DataFrame(results))

# --- Show Logs ---
st.subheader("📝 Recent Logs")
if st.checkbox("Show last 10 predictions"):
    log_df = pd.read_csv(LOG_FILE)
    st.dataframe(log_df.tail(10))
