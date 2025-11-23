# AI Text Classifier

This project is a simple AI-powered text analysis system that performs **Topic Classification**. It uses a React frontend and a Python (FastAPI) backend with Hugging Face Transformers.

## Features

- **Topic Classification**: Categorizes text into predefined topics (Sports, Politics, Technology, Entertainment, Business, Health, Law, Science, Education, Environment, Finance, Travel, Food) using Zero-Shot Classification.
- **Simple Interface**: Clean and responsive UI built with React.
- **API**: RESTful API endpoint for classification.

## Tech Stack

- **Frontend**: React, Vite, Axios, CSS
- **Backend**: Python, FastAPI, Uvicorn, Hugging Face Transformers, PyTorch

## Setup Instructions

### Prerequisites

- Node.js and npm installed
- Python 3.8+ installed

### Backend Setup

1. Navigate to the `backend` directory:
   ```bash
   cd backend
   ```
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Mac/Linux
   source venv/bin/activate
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the server:
   ```bash
   python main.py
   ```
   The API will be available at `http://localhost:8000`.
   _Note: On the first run, the model (facebook/bart-large-mnli) will be downloaded (approximately 1.6GB)._

### Frontend Setup

1. Navigate to the `frontend` directory:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Run the development server:
   ```bash
   npm run dev
   ```
   The application will be available at `http://localhost:5173`.

## Usage

1. Ensure both backend and frontend servers are running.
2. Open `http://localhost:5173` in your browser.
3. Paste an English text article into the text area.
4. Click "Classify Text".
5. View the predicted topic and confidence scores below.

## Approach & Implementation

### 1. Algorithm & Model Selection

I selected **Zero-Shot Classification** using the `facebook/bart-large-mnli` model.

- **Why Zero-Shot?**

  - **Flexibility**: Traditional classification models require training on a fixed set of labeled data. If we wanted to add a "Law" category later, we would need to retrain the entire model. Zero-Shot learning allows us to define labels dynamically at runtime (e.g., adding "Law", "Finance") without any model retraining.
  - **Efficiency**: It eliminates the need for a large, labeled dataset, which is often the bottleneck in ML projects.

- **Why `facebook/bart-large-mnli`?**

  - **Performance**: BART (Bidirectional and Auto-Regressive Transformers) is a state-of-the-art architecture. The model fine-tuned on the MNLI (Multi-Genre Natural Language Inference) dataset is particularly effective at understanding the relationship between a premise (the text) and a hypothesis (This text is about {label}).
  - **Robustness**: It generalizes well to unseen topics, making it ideal for a general-purpose classifier.

  **Model Comparison:**

  | Model                                   | Size    | Speed  | Accuracy | Best For                            |
  | :-------------------------------------- | :------ | :----- | :------- | :---------------------------------- |
  | **facebook/bart-large-mnli** (Selected) | ~1.6 GB | Medium | High     | High-quality English classification |
  | valhalla/distilbart-mnli-12-1           | ~700 MB | Fast   | Medium   | Real-time/Low-latency apps          |
  | joeddav/xlm-roberta-large-xnli          | ~2.2 GB | Slow   | High     | Multilingual support                |

  _Sources:_

  - [facebook/bart-large-mnli](https://huggingface.co/facebook/bart-large-mnli)
  - [valhalla/distilbart-mnli-12-1](https://huggingface.co/valhalla/distilbart-mnli-12-1)
  - [joeddav/xlm-roberta-large-xnli](https://huggingface.co/joeddav/xlm-roberta-large-xnli)

  I chose `facebook/bart-large-mnli` because it offers the best balance of accuracy and robustness for a demonstration where quality is prioritized over raw speed. While `distilbart` is faster, its accuracy drop can be noticeable in nuanced topics. `xlm-roberta` is excellent but overkill for an English-only task and consumes significantly more resources.

### 2. System Architecture

- **Backend (FastAPI)**: Chosen for its high performance (Starlette-based) and native support for asynchronous operations, which is crucial when serving ML models that might block the main thread.
- **Frontend (React)**: Provides a responsive, component-based UI. I implemented a dynamic bar chart visualization to make the confidence scores intuitive for the user.

## Potential Improvements

If given more time and resources, I would enhance the system in the following ways:

1.  **Performance Optimization**:

    - **Model Distillation/Quantization**: The current model is large (~1.6GB). I would use techniques like quantization (INT8) or switch to a distilled version (e.g., `distilbart-mnli`) to reduce memory usage and inference latency by 2-3x.
    - **ONNX Runtime**: Export the model to ONNX format to leverage hardware acceleration (AVX512 on CPU or TensorRT on GPU).
    - **Caching**: Implement Redis caching to store results for frequently analyzed text hashes.

2.  **Feature Extensions**:
    - **Custom Labels**: Allow users to type in their own categories (e.g., "Urgent", "Spam") on the frontend to fully utilize the Zero-Shot capability.
    - **Multilingual Support**: Swap the model for `joeddav/xlm-roberta-large-xnli` to support classification in 100+ languages.
    - **Batch Processing**: Add an endpoint to accept a CSV file and classify thousands of rows in parallel.

## Evaluation Strategy

To rigorously evaluate the system, I would implement the following:

1.  **Quantitative Metrics**:

    - **Accuracy & F1-Score**: I would benchmark the model against standard datasets (like AG News or 20 Newsgroups) using our specific labels.
    - **Top-3 Accuracy**: Since texts can be ambiguous, measuring whether the correct label appears in the top 3 predictions is often more meaningful than strict Top-1 accuracy.
    - **Confidence Calibration**: Analyze if the model's confidence scores correlate with actual accuracy (e.g., is it actually right 90% of the time when it says 90% confident?).

2.  **Performance Metrics**:

    - **Latency (P95/P99)**: Measure the time taken for requests. The goal would be to keep P95 latency under 200ms for short texts.
    - **Throughput**: Test how many requests per second (RPS) the single API instance can handle before degrading.

3.  **Qualitative Analysis**:
    - **Confusion Matrix**: Visualize which categories are most often confused (e.g., "Business" vs. "Finance") to refine the label definitions.
    - **Adversarial Testing**: Test with tricky inputs (e.g., sarcasm, negation) to understand model limitations.
