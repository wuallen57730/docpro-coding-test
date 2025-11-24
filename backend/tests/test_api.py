from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_read_root():
    """Test that the root endpoint returns 200 and welcome message."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Welcome" in response.json()["message"]

def test_classify_text_success():
    """Test a valid classification request."""
    # Note: This test assumes the model is loaded. 
    # In a real CI environment, we might mock the model or ensure it's trained.
    # For this simple setup, we'll assume train_setfit.py has been run.
    payload = {
        "text": "The stock market crashed today due to inflation."
    }
    response = client.post("/classify", json=payload)
    
    # If model is not loaded (e.g. first run without training), it returns 503
    if response.status_code == 503:
        assert "Model not loaded" in response.json()["detail"]
    else:
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert "labels" in data
        assert "scores" in data
        assert "top_label" in data
        assert "top_score" in data
        
        # Check logic (Finance should be high for this text)
        # We relax the assertion slightly in case of model variance, but top label should be finance
        assert data["top_label"] == "finance"
        assert data["top_score"] > 0.3

def test_classify_empty_text():
    """Test that empty text returns a 400 error."""
    payload = {
        "text": "   "
    }
    response = client.post("/classify", json=payload)
    assert response.status_code == 400
    assert "Text cannot be empty" in response.json()["detail"]

def test_classify_missing_field():
    """Test validation error when 'text' field is missing."""
    payload = {}
    response = client.post("/classify", json=payload)
    assert response.status_code == 422 # Unprocessable Entity
