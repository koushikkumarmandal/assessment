import re
import nltk
import joblib
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

df = pd.read_csv("tickets.csv")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


df["text"] = df["subject"].fillna("") + " " + df["body"].fillna("")
df["text"] = df["text"].apply(clean_text)

X = df["text"]
y = df["category"]


vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


model = MultinomialNB()
model.fit(X_train, y_train)


prediction = model.predict(X_test)

print("=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(f"Accuracy : {accuracy_score(y_test, prediction):.4f}")

print("\nClassification Report\n")
print(classification_report(y_test, prediction))

print("Confusion Matrix\n")
print(confusion_matrix(y_test, prediction))


joblib.dump(model, "classifier.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\nModel saved successfully!")


def get_priority(ticket):

    keywords = [
        "urgent",
        "critical",
        "down",
        "crash",
        "failed",
        "error",
        "not working",
        "immediately",
        "asap",
    ]

    ticket = ticket.lower()

    for word in keywords:
        if word in ticket:
            return "High"

    return "Normal"


def predict_ticket(ticket):

    cleaned = clean_text(ticket)

    vector = vectorizer.transform([cleaned])

    category = model.predict(vector)[0]

    confidence = model.predict_proba(vector).max()

    route = (
        "Needs Human Review"
        if confidence < 0.60
        else "Auto Assign"
    )

    print("\n" + "=" * 60)
    print("Incoming Ticket")
    print("=" * 60)
    print(ticket)

    print("\nPredicted Category :", category)
    print(f"Confidence          : {confidence*100:.2f}%")
    print("Priority             :", get_priority(ticket))
    print("Routing              :", route)


sample_tickets = [
    "I paid twice and need a refund.",
    "The application crashes every time I login.",
    "Please approve my leave request.",
    "Can you tell me your business hours?",
    "The server is down. This is urgent."
]

print("\n\nSAMPLE PREDICTIONS")

for ticket in sample_tickets:
    predict_ticket(ticket)


print("\n" + "=" * 60)
print("LIVE TICKET CATEGORIZER")
print("=" * 60)

while True:

    text = input("\nEnter Ticket (type 'exit' to quit): ")

    if text.lower() == "exit":
        break

    predict_ticket(text)

print("\nProgram Finished.")