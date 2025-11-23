import { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [inputText, setInputText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleClassify = async () => {
    if (!inputText.trim()) {
      setError("Please enter some text to classify.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      // In a real app, the URL should be in an environment variable
      const response = await axios.post("http://localhost:8000/classify", {
        text: inputText,
        candidate_labels: [
          "sports",
          "politics",
          "technology",
          "entertainment",
          "business",
          "health",
          "law",
          "science",
          "education",
          "environment",
          "finance",
          "travel",
          "food",
        ],
      });

      setResult(response.data);
    } catch (err) {
      console.error(err);
      setError(
        "Failed to classify text. Please ensure the backend is running."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <header>
        <h1>AI Text Classifier</h1>
        <p>Powered by SetFit (Few-Shot Learning)</p>
      </header>

      <main>
        <div className="input-section">
          <textarea
            placeholder="Paste your text here (English)..."
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            rows={10}
            disabled={loading}
          />
          <div className="controls">
            <button
              onClick={handleClassify}
              disabled={loading || !inputText.trim()}
            >
              {loading ? "Classifying..." : "Classify Text"}
            </button>
            <button
              className="clear-btn"
              onClick={() => {
                setInputText("");
                setResult(null);
                setError("");
              }}
              disabled={loading}
            >
              Clear
            </button>
          </div>
        </div>

        {error && <div className="error-message">{error}</div>}

        {result && (
          <div className="result-section">
            <h2>Classification Result</h2>
            <div className="summary-box">
              <h3>
                Top Topic: <span className="highlight">{result.top_label}</span>
              </h3>
              <p>Confidence: {(result.top_score * 100).toFixed(2)}%</p>

              <div className="all-scores">
                <h4>All Scores:</h4>
                <ul>
                  {result.labels.map((label, index) => (
                    <li key={label} className="score-item">
                      <span className="label">{label}</span>
                      <div className="bar-container">
                        <div
                          className="bar"
                          style={{ width: `${result.scores[index] * 100}%` }}
                        ></div>
                      </div>
                      <span className="score">
                        {(result.scores[index] * 100).toFixed(1)}%
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
