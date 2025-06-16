import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Load the dataset with the correct encoding
df = pd.read_csv("C://Users//91971//email_data//spam.csv", encoding='latin-1')  # Specify encoding

# Step 2: Rename columns if needed
df = df.rename(columns={"v1": "label", "v2": "message"})

# Step 3: Encode labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Step 4: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Step 5: Convert text to numerical features
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# Step 6: Train a model
model = MultinomialNB()
model.fit(X_train_counts, y_train)

# Step 7: Predict and evaluate
y_pred = model.predict(X_test_counts)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Test with custom input
test_messages = ["Free entry in a contest!", "Hey, are we meeting tomorrow?"]
test_counts = vectorizer.transform(test_messages)
predictions = model.predict(test_counts)

# Print predictions
for message, label in zip(test_messages, predictions):
    print(f"Message: '{message}' -> {'Spam' if label == 1 else 'Ham'}")