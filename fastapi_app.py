
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import uvicorn

app = FastAPI(title="Emotion Classifier API")

# Load model
model = joblib.load('emotion_classifier.joblib')

class TextRequest(BaseModel):
    text: str

class TextsRequest(BaseModel):
    texts: list[str]

@app.post("/predict")
async def predict(request: TextRequest):
    emotions = model.predict_emotions(request.text)[0]
    return {"text": request.text, "emotions": emotions}

@app.post("/predict_batch")
async def predict_batch(request: TextsRequest):
    emotions = model.predict_emotions(request.texts)
    return {"texts": request.texts, "emotions": emotions}

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
