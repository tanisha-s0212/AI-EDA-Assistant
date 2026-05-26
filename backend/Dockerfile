FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV ML_PORT=3004
ENV AGENTIC_LAYER_ROOT=/agentic-layer

RUN apt-get update \
  && apt-get install -y --no-install-recommends chromium \
  && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir lightgbm pmdarima
COPY backend/ ./
COPY agentic-layer/ /agentic-layer/
EXPOSE 3004
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3004"]
