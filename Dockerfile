# Use a lightweight Python image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Prevent Python from writing pyc files and keep stdout unbuffered
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# --- NEW FIX: Install Linux dependencies required by LightGBM ---
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project (including your data/seed folder)
COPY . .

# Bake a trained model into the image at build time
RUN python main.py

# Railway/Render inject $PORT and expect the app to bind to it
ENV PORT=8000
EXPOSE 8000
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT}