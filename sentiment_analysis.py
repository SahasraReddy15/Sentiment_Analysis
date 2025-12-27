import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

#Dataset
data = {
    "text": [
        # Positive
        "i love this product",
        "this app is amazing",
        "very happy with this app",
        "i really like this application",
        "excellent experience",
        "this is nice",
        "i like this app",
        "good application",
        "awesome scenery",
        "super experience",

        # Negative
        "i hate this product",
        "this app is very bad",
        "worst experience ever",
        "i am unhappy with this service",
        "i do not like this app",
        "this is not good",
        "bad application",
        "there is dirt",
        "place is dirty",
        "bad smell here"
    ],
    "sentiment": [
        "positive","positive","positive","positive","positive",
        "positive","positive","positive","positive","positive",
        "negative","negative","negative","negative","negative",
        "negative","negative","negative","negative","negative"
    ]
}


df = pd.DataFrame(data)

# Vectorization (NO stop words removal)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["sentiment"]

# Train model
model = MultinomialNB()
model.fit(X, y)

print("Model trained successfully!")

# Prediction loopexit
while True:
    user_input = input("\nEnter a sentence (or type 'exit' to stop): ")
    if user_input.lower() == "exit":
        print("Program ended.")
        break

    input_vector = vectorizer.transform([user_input])
    prediction = model.predict(input_vector)
    print("Sentiment:", prediction[0])
