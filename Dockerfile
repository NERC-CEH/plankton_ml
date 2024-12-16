# Container for FastAPI model serving
FROM python:3.12

WORKDIR /app

COPY ./src /app/src
COPY pyproject.toml /app/pyproject.toml

RUN pip install --no-cache-dir --upgrade -e .

# Local copies of model weights
COPY ./data /app/data

CMD ["fastapi", "run", "src/cyto_ml/models/api.py", "--port", "8000"]