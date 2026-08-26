FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Bake a trained model into the image at build time, so the container
# is self-contained and reproducible (fixed seed) with nothing to wait
# on at startup. Real lightgbm/shap are installed above via
# requirements.txt, so a normal build trains on those, not the
# no-network sandbox fallbacks this was developed under.
RUN python main.py

# Railway/Render inject $PORT and expect the app to bind to it; default
# to 8000 for `docker run` without one set (e.g. plain docker-compose).
ENV PORT=8000
EXPOSE 8000
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT}
