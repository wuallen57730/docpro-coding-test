from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

# Initialize the FastAPI app
app = FastAPI(
    title="AI Text Classifier API",
    description="A simple API for text classification using Hugging Face Transformers.",
    version="1.0.0"
)

# Configure CORS to allow requests from the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the classification pipeline
# We use 'facebook/bart-large-mnli' for zero-shot classification
# This allows us to classify text into any categories we define without specific training
try:
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    classifier = None

class TextRequest(BaseModel):
    text: str
    candidate_labels: list[str] = [
        "sports", "politics", "technology", "entertainment", "business", "health",
        "law", "science", "education", "environment", "finance", "travel", "food"
    ]

class ClassificationResponse(BaseModel):
    labels: list[str]
    scores: list[float]
    top_label: str
    top_score: float

@app.get("/")
def read_root():
    return {"message": "Welcome to the AI Text Classifier API"}

@app.post("/classify", response_model=ClassificationResponse)
def classify_text(request: TextRequest):
    if not classifier:
        raise HTTPException(status_code=503, detail="Model not loaded available")
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        # The pipeline returns a dict: 
        # {'sequence': '...', 'labels': ['label1', 'label2'], 'scores': [0.9, 0.1]}
        result = classifier(request.text, request.candidate_labels)
        
        return ClassificationResponse(
            labels=result['labels'],
            scores=result['scores'],
            top_label=result['labels'][0],
            top_score=result['scores'][0]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
