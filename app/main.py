from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.inference import IdiomSpanDetector

app = FastAPI(title="IdiomaticBERT API", version="1.0.0")

detector = IdiomSpanDetector(model_dir="model")


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Input sentence/text")
    max_length: int = Field(256, ge=16, le=512)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    return detector.predict(req.text, max_length=req.max_length)
