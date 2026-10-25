# 1. Use the official lightweight Python base image
FROM python:3.11-slim

# 2. creates /app and makes it the working directory
WORKDIR /app   

# 3. Copy only dependency file first (for Docker caching)
COPY requirements-serve.txt .

# 4. Install Python dependencies (add curl if you use MLflow local tracking URI)
RUN pip install --upgrade pip \
    && pip install -r requirements-serve.txt \
    && apt-get clean && rm -rf /var/lib/apt/lists/*  # cleans the apt cache

# 5. Copy the entire project into the image
COPY . .    # copies the whole project into /app, minus anything listed in 
# .dockerignore

# Explicitly copy model (in case .dockerignore excluded mlruns)
# NOTE: destination changed to /app/src/serving/model to match inference.py's path
COPY src/serving/model /app/src/serving/model

# Copy MLflow run (artifacts + metadata) to the flat /app/model convenience path
COPY src/serving/model/3b1a41221fc44548aed629fa42b762e0/artifacts/model /app/model
COPY src/serving/model/3b1a41221fc44548aed629fa42b762e0/artifacts/feature_columns.txt /app/model/feature_columns.txt
COPY src/serving/model/3b1a41221fc44548aed629fa42b762e0/artifacts/preprocessing.pkl /app/model/preprocessing.pkl

# make "serving" and "app" importable without the "src." prefix
# ensures logs are shown in real-time (no buffering).
# lets you import modules using from app... instead of from src.app....
# PYTHONUNBUFFERED: logs appear immediately instead of being buffered
# (helps with docker logs and CloudWatch)
# PYTHONPATH: lets code import app... and serving... without the src. prefix
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

# 6. Expose FastAPI port
# Documents that the app listens on port 8000. It does not publish the port;
# you still need docker run -p 8000:8000 to publish it.
EXPOSE 8000

# 7. Run the FastAPI app using uvicorn (change path if needed)
CMD ["python", "-m", "uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
