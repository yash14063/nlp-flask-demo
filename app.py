from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load models
sentiment_model = joblib.load('sentiment_model.pkl')
topic_model = joblib.load('topic_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

TASKS = {
    "sentiment": "Sentiment Analysis",
    "topic": "Topic Classification"
}

@app.route('/')
def home():
    return render_template('index.html', tasks=TASKS)

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    task = request.form['task']

    vec = vectorizer.transform([text])

    if task == "sentiment":
        model = sentiment_model
        label = model.predict(vec)[0]
        proba = np.max(model.predict_proba(vec))
        label_name = "Positive" if label == 1 else "Negative"
        feature_weights = model.coef_[0]
        important_words = get_important_words(text, feature_weights, vectorizer)
    elif task == "topic":
        model = topic_model
        label = model.predict(vec)[0]
        proba = np.max(model.predict_proba(vec))
        label_name = model.classes_[label]
        important_words = get_important_words(text, model.coef_[label], vectorizer)
    else:
        label_name = "Unknown"
        proba = 0.0
        important_words = []

    return render_template(
        'index.html',
        prediction=label_name,
        probability=round(proba, 2),
        tasks=TASKS,
        selected_task=task,
        input_text=text,
        important_words=important_words
    )

def get_important_words(text, coef, vectorizer):
    words = text.split()
    vocab = vectorizer.vocabulary_
    word_weights = [(word, coef[vocab[word.lower()]] if word.lower() in vocab else 0) for word in words]
    # Get top 5 words by absolute weight
    important = sorted(word_weights, key=lambda x: abs(x[1]), reverse=True)[:5]
    return important

if __name__ == '__main__':
    app.run(debug=True)