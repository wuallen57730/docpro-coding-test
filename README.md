# AI Text Classifier

This coding test is a simple AI-powered text analysis system that performs **Topic Classification**. It uses a React frontend and a Python (FastAPI) backend with Hugging Face Transformers.

## Features

- **Topic Classification**: Categorizes text into predefined topics (Sports, Politics, Technology, Entertainment, Business, Health, Law, Science, Education, Environment, Finance, Travel, Food) using **Few-Shot Classification (SetFit)**.
- **Simple Interface**: Clean and responsive UI built with React.
- **API**: RESTful API endpoint for classification.

## Tech Stack

- **Frontend**: React, Vite, Axios, CSS
- **Backend**: Python, FastAPI, Uvicorn, SetFit, Sentence Transformers, PyTorch

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

4. Train the model (Required for first run):

   ```bash
   python train_setfit.py
   ```

   This will fine-tune the `sentence-transformers/paraphrase-MiniLM-L6-v2` model on a small dataset and save it to the `setfit-model/` directory.

5. Run the server:
   ```bash
   python main.py
   ```
   The API will be available at `http://localhost:8000`.
   _Note: The base model is lightweight (~80MB), so download and startup are very fast._

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

I selected a **Few-Shot Learning** approach using **SetFit** (Sentence Transformer Fine-tuning) with the `sentence-transformers/paraphrase-MiniLM-L6-v2` model.

#### Why SetFit with `paraphrase-MiniLM-L6-v2`?

1.  **High Accuracy with Little Data**: SetFit is designed specifically for scenarios where labeled data is scarce. By using contrastive learning, it achieves high accuracy with as few as 5-10 examples per class, significantly outperforming standard fine-tuning on small datasets.
2.  **Context Awareness**: Unlike Zero-Shot models that often rely on keyword matching (e.g., seeing "lawyer" and guessing "Law"), SetFit learns from the _examples_ provided. This allows it to correctly classify a "Courtroom Drama" as **Entertainment** because it learns the semantic pattern of the plot description rather than just spotting the word "lawyer".
3.  **Efficiency & Speed**: The `paraphrase-MiniLM-L6-v2` base model is extremely lightweight (~80MB) compared to large Zero-Shot models like `bart-large-mnli` (~1.6GB). This results in inference speeds that are **10x-50x faster** on CPU, making it viable for real-time applications without expensive GPUs.

#### Comparison with Other Approaches

| Approach                        | Model Size        | Inference Speed       | Data Requirements    | Pros                                   | Cons                                   |
| :------------------------------ | :---------------- | :-------------------- | :------------------- | :------------------------------------- | :------------------------------------- |
| **SetFit (Selected)**           | **Small (~80MB)** | **Very Fast (<50ms)** | **Low (5-10/class)** | **High accuracy, runs locally, cheap** | Requires a small training step         |
| **Zero-Shot (BART-MNLI)**       | Large (~1.6GB)    | Slow (~2.6s)          | None                 | No training needed, flexible labels    | Lower accuracy, struggles with context |
| **LLM Few-Shot (GPT-4)**        | Huge (API)        | Variable              | None (Prompting)     | Extremely versatile, high reasoning    | **High cost, latency, data privacy**   |
| **Standard Fine-Tuning (BERT)** | Medium (~400MB)   | Fast                  | High (100+/class)    | Industry standard for large datasets   | **Performs poorly with few examples**  |

I chose SetFit because it offers the **best trade-off** for this project: it fixes the accuracy issues of Zero-Shot (reaching >90%) while being significantly faster and cheaper to run than LLMs, all without needing a massive dataset.

### 2. System Architecture

- **Backend (FastAPI)**: Serves the trained SetFit model.
- **Frontend (React)**: Displays classification results with confidence scores.

## Training the Model

The backend uses a SetFit model that needs to be trained first.

1. Navigate to the `backend` directory:
   ```bash
   cd backend
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the training script:
   ```bash
   python train_setfit.py
   ```
   This will train the model on a small dataset defined in the script and save it to the `setfit-model/` directory.

## Evaluation & Performance

I have implemented an evaluation script (`backend/evaluate.py`) to benchmark the model's performance.

### Running the Evaluation

To run the evaluation suite:

```bash
python backend/evaluate.py
```

### Current Results (SetFit)

- **Accuracy**: 87%
- **Weighted F1-Score**: 0.89
- **Average Latency**: ~51ms per request (CPU)
- **Key Improvements**:
  - Solved the confusion between **Entertainment** and **Law** (e.g., courtroom dramas).
  - Solved the confusion between **Politics** and **Technology** (e.g., AI regulation).

## Software Engineering Practices

### 1. Containerization (Docker)

The project is fully containerized for easy deployment.

**Prerequisites**: Docker and Docker Compose installed.

**Run with Docker Compose**:

```bash
docker-compose up --build
```

- The backend will be available at `http://localhost:8000`.
- The frontend will be available at `http://localhost:3000`.
- _Note: The backend build process includes training the model, so the first build may take a few minutes._

### 2. Automated Testing

Unit tests are implemented using `pytest` to ensure API reliability.

**Run Tests**:

```bash
cd backend
pytest
```

## Future Improvements

1.  **Data Augmentation**: Use LLMs to generate more diverse training examples for the SetFit model.
2.  **Active Learning**: Implement a feedback loop where users can correct misclassifications to retrain the model.
