# Import libraries
import pandas as pd
import nltk
import string

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Download stopwords
nltk.download('stopwords')

# Sample dataset
data = {
    "review": [
        "This product is amazing",
        "I love this item",
        "Excellent quality",
        "Very satisfied",
        "Best purchase ever",
        "Terrible product",
        "Very bad quality",
        "Waste of money",
        "I hate this item",
        "Not worth buying"
    ],

    "sentiment": [
        "positive",
        "positive",
        "positive",
        "positive",
        "positive",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative"
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Text cleaning function
def clean_text(text):

    text = text.lower()

    text = ''.join(
        char for char in text
        if char not in string.punctuation
    )

    words = text.split()

    words = [
        word for word in words
        if word not in stopwords.words('english')
    ]

    return ' '.join(words)

# Apply cleaning
df["review"] = df["review"].apply(clean_text)

# Features and Labels
X = df["review"]
y = df["sentiment"]

# Convert text into numbers
vectorizer = TfidfVectorizer()

X_vectorized = vectorizer.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = MultinomialNB()

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)

# Test custom review
new_review = ["This product is fantastic"]

new_review_clean = [clean_text(new_review[0])]

new_vector = vectorizer.transform(new_review_clean)

prediction = model.predict(new_vector)

print("Review:", new_review[0])
print("Predicted Sentiment:", prediction[0])