# Použi oficiálny Python image
FROM python:3.12-slim


# Nastav pracovný adresár
WORKDIR /app

# Skopíruj requirements a nainštaluj závislosti
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt


# Skopíruj celý projekt
COPY . .

# Otvor port pre FastAPI
EXPOSE 9123

# Štart servera
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "9123"]
