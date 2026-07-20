from flask import Flask, render_template, request
import joblib
import re
import os

app = Flask(__name__)

print("Current Folder:", os.getcwd())
print("Templates Exists:", os.path.exists("templates"))
print("Index Exists:", os.path.exists("templates/index.html"))

# Load saved model and vectorizer
model = joblib.load("model/fake_news_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")


# Text Cleaning Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    news = request.form["news"]

    cleaned_news = clean_text(news)

    vector = vectorizer.transform([cleaned_news])

    prediction = model.predict(vector)

    confidence = model.predict_proba(vector)[0]
    confidence = round(max(confidence) * 100, 2)

    if prediction[0] == 0:
        result = "🔴 Fake News"
        color = "red"
    else:
        result = "🟢 Real News"
        color = "green"

    return render_template(
        "index.html",
        prediction=result,
        color=color,
        confidence=confidence
    )


if __name__ == "__main__":
    app.run(debug=True)