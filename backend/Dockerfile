# Use an official Python runtime with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.9-bookworm-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies using uv (much faster than pip)
# --system installs into the system python environment, avoiding the need for a venv in the container
RUN uv pip install --system --no-cache -r requirements.txt

# Copy the rest of the application code
COPY . .

# Train the model during the build process so it's baked into the image
# This prevents the container from training every time it starts
# Note: This increases build time but ensures the container is ready-to-run
RUN python train_setfit.py

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run main.py when the container launches
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
