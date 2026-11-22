# Stage 1: train the model inside the build, so the image always ships a matching artifact
FROM python:3.12-slim AS train
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
COPY scripts/ scripts/
COPY data/ data/
# Uses data/train.csv (Kaggle) if present, otherwise generates synthetic sample data
RUN if [ ! -f data/train.csv ]; then python scripts/make_sample_data.py --rows 20000 --out data/train.csv; fi \
 && PYTHONPATH=src python -m insurance_api.train --data data/train.csv --out models/model.joblib

# Stage 2: small runtime image with only the code and the model
FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home appuser
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
COPY --from=train /app/models/ models/
ENV PYTHONPATH=/app/src MODEL_PATH=/app/models/model.joblib
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "insurance_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
