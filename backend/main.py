from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import os
from setfit import SetFitModel
import numpy as np

# Initialize the classification model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "setfit-model")
setfit_model = None
label_list: list[str] = []

def load_model():
    global setfit_model, label_list
    if setfit_model is not None:
        return

    if not os.path.isdir(MODEL_PATH):
        print(f"SetFit model not found at {MODEL_PATH}. Please run `python train_setfit.py` first.")
        return

    try:
        print(f"Loading SetFit model from {MODEL_PATH}...")
        model = SetFitModel.from_pretrained(MODEL_PATH)
        # SetFitModel.labels contains the list of labels used during training
        setfit_model = model
        label_list = list(model.labels)
        print(f"Model loaded successfully with labels: {label_list}")
    except Exception as e:
        print(f"Error loading model: {e}")
        setfit_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    load_model()
    yield
    # Clean up the ML models and release the resources
    global setfit_model
    setfit_model = None

# Initialize the FastAPI app
app = FastAPI(
    title="AI Text Classifier API",
    description="A simple API for text classification using SetFit (Few-Shot Learning).",
    version="2.0.0",
    lifespan=lifespan
)

# Configure CORS to allow requests from the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    text: str
    # candidate_labels is no longer needed for SetFit as labels are fixed in the model,
    # but we keep it optional for backward compatibility if needed, though ignored.
    candidate_labels: list[str] = []

class ClassificationResponse(BaseModel):
    labels: list[str]
    scores: list[float]
    top_label: str
    top_score: float

@app.get("/")
def read_root():
    return {"message": "Welcome to the AI Text Classifier API (SetFit Version)"}

@app.post("/classify", response_model=ClassificationResponse)
def classify_text(request: TextRequest):
    if setfit_model is None:
        # Try loading again if it failed initially or wasn't ready
        load_model()
        if setfit_model is None:
             raise HTTPException(status_code=503, detail="Model not loaded. Please run backend/train_setfit.py to train the model first.")
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        # SetFit predict_proba returns probabilities for each class
        # Input must be a list of strings
        probs = setfit_model.predict_proba([request.text])[0]
        
        # Convert numpy array to list
        probs = probs.tolist() if not isinstance(probs, list) else probs
        scores = [float(p) for p in probs]
        labels = label_list

        # Sort by score descending
        sorted_indices = np.argsort(scores)[::-1]
        sorted_labels = [labels[i] for i in sorted_indices]
        sorted_scores = [scores[i] for i in sorted_indices]
        
        return ClassificationResponse(
            labels=sorted_labels,
            scores=sorted_scores,
            top_label=sorted_labels[0],
            top_score=sorted_scores[0]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
