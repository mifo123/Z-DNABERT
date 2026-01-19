FROM python:3.12-slim

WORKDIR /app

# CPU/BLAS thread limit (rozumne defaulty, dá sa overridnuť pri deployi)
ENV OMP_NUM_THREADS=24 \
    MKL_NUM_THREADS=24 \
    OPENBLAS_NUM_THREADS=24 \
    NUMEXPR_NUM_THREADS=24 \
    ZDNABERT_NUM_THREADS=24

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

# jeden uvicorn worker – paralelizmus riešiš na úrovni threadov + replic
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "9123"]
