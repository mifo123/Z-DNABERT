FROM python:3.12-slim

WORKDIR /app

# Inštalácia potrebných balíkov v jednej vrstve + čistenie APT cache
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl perl \
    && rm -rf /var/lib/apt/lists/*

# Inštalácia Python závislostí
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Skopíruj projekt
COPY . .

EXPOSE 9123

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "9123"]
