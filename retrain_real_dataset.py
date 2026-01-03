import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")

# Keep only needed columns
df = df[['v1', 'v2']]
df.columns = ['label', 'text']

# Convert labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

# Vectorizer (IMPORTANT CHANGE)
vectorizer = TfidfVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)

# Model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Evaluate
X_test_vec = vectorizer.transform(X_test)
accuracy = accuracy_score(y_test, model.predict(X_test_vec))
print("Model accuracy:", accuracy)

# Save trained artifacts
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ New trained model and vectorizer saved")
