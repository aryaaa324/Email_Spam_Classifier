# 📧 Email Spam Classifier using Machine Learning + Streamlit

This project is a complete **end-to-end Email/SMS Spam Classifier** powered by Machine Learning and a user-friendly **Streamlit web interface**. It detects whether a message is **spam** or **not spam (ham)** using a pre-trained **Multinomial Naive Bayes** model with **TF-IDF** vectorization. The app allows users to test messages in real time and keeps a record of predictions for future analysis.

---

## 🚀 Project Overview

Spam messages are a persistent problem in emails, SMS, and messaging platforms. This project provides an effective solution by:

- Training a spam classifier using **TF-IDF** and **Naive Bayes**
- Building an interactive **Streamlit app** for real-time classification
- Logging every prediction with a **timestamp** for review and analysis

---

## 📂 Dataset

- **Source**: [Kaggle – SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Total Samples**: 5,574 labeled messages
- **Columns**:
  - `v1`: Label (spam or ham)
  - `v2`: Message content

---

## 📁 Project Structure

email-spam-classifier/
│
├── spam.csv # Dataset
├── train_model.py # Script to train & save model
├── model.pkl # Trained model (generated)
├── vectorizer.pkl # TF-IDF vectorizer (generated)
├── app.py # Streamlit app UI
├── spam_logs.csv # Prediction logs (auto-generated)
├── requirements.txt # Python dependencies
├── venv/ # Optional virtual environment

yaml
Copy
Edit

---

## 🧠 ML Pipeline (in `train_model.py`)

### 1. Preprocessing
- Load and clean dataset
- Rename columns
- Encode labels: `ham → 0`, `spam → 1`

### 2. Vectorization
- Apply **TF-IDF** to convert text to feature vectors

### 3. Model Training
- Train a **Multinomial Naive Bayes** classifier on 80% of the data

### 4. Evaluation
- Evaluate using **Accuracy**, **Precision**, **Recall**, and **F1-score**

### 5. Saving
- Save trained `model.pkl` and `vectorizer.pkl` for app usage

---

## 🌐 Streamlit Web App (in `app.py`)

- 📥 **Single Message Check**: Type a message to get instant prediction
- 📂 **Batch File Upload**: Upload a `.txt` file of messages (one per line)
- 📝 **Logs**: View last 10 predictions from `spam_logs.csv`

### 🔧 Features:
- Real-time classification
- Clean and responsive UI
- Confidence visualization via bar chart
- Logs predictions with timestamps

---

## 🛠 Tech Stack

- **Python 3**
- **Scikit-learn** – Model training & preprocessing
- **Pandas / NumPy** – Data manipulation
- **Streamlit** – Web interface
- **Pickle** – Model persistence
- **Matplotlib** – Confidence bar chart

---

## 📊 Sample Output

```text
Accuracy: 0.98

              precision    recall  f1-score   support

           0       0.98      0.99      0.99       966
           1       0.96      0.91      0.93       149

    accuracy                           0.98      1115
```
---
## ▶️ How to Run the Project
1. Clone the Repository
```text
bash
git clone https://github.com/your-username/email-spam-classifier.git
cd email-spam-classifier
```

2. Install Dependencies
```text
bash
pip install -r requirements.txt
```

2. Train the Model (Optional if model.pkl exists)
```text
bash
python train_model.py
```

3. Launch the Streamlit App
```text
bash
streamlit run app.py
```
---
## 📌 Key Features
✅ Full ML pipeline: from raw data to prediction
🌐 Easy-to-use Streamlit interface
📈 Real-time prediction logging
🚀 Lightweight and fast to deploy
💡 Great starter project for ML and Streamlit beginners
