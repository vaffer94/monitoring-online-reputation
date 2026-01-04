FROM python:3.12-slim

# Avoid Python buffering logs
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src ./src
COPY data ./data

# Expose FastAPI port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "src.service.main:app", "--host", "0.0.0.0", "--port", "8000"]
