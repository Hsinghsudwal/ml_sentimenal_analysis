FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY app_requirements.txt .
RUN pip install --prefix=/install --no-cache-dir -r app_requirements.txt

FROM python:3.12-slim

WORKDIR /app

# COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /install /usr/local

COPY app.py .
COPY pipelines ./pipelines
COPY artifact ./artifact

EXPOSE 9696

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9696"]
