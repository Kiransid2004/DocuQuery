FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*

COPY requirements.docker.txt .
RUN pip install --no-cache-dir -r requirements.docker.txt

COPY . .
RUN mkdir -p data data/images

EXPOSE 7860
ENV PORT=7860


CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}
