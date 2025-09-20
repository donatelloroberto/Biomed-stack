# This is a lightweight stub. Replace with full OpenFold/AlphaFold container for real workloads.
FROM python:3.11-slim
WORKDIR /app
COPY app.py .
RUN pip install fastapi uvicorn pydantic
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8102"]
