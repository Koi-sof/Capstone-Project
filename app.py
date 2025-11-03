
from flask import Flask, request, jsonify
import joblib
import torch
import os

# Force CPU for deployment
os.environ['CUDA_VISIBLE_DEVICES'] = ''

app = Flask(__name__)

# Load model
model = joblib.load('emotion_classifier.joblib')
model.device = torch.device('cpu')
if hasattr(model, 'model'):
    model.model = model.model.to('cpu')

def predict_emotions(texts):
    predictions = model.predict(texts)
    return [[model.classes_[i] for i, pred in enumerate(row) if pred == 1] for row in predictions]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    emotions = predict_emotions([text])[0]
    return jsonify({"text": text, "emotions": emotions})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
