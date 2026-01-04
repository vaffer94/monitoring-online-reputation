from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response
import time

from src.model.sentiment_model import load_sentiment_model, predict_sentiment

app = FastAPI(
    title="Sentiment Analysis API",
    version="1.0.0",
    description="Minimal sentiment service for MLOps assignment"
)

REQUEST_COUNT = Counter(
    "request_count_total",
    "Total number of prediction requests"
)

REQUEST_LATENCY = Histogram(
    "request_latency_seconds",
    "Latency of prediction requests"
)

SENTIMENT_COUNT = Counter(
    "sentiment_predictions_total",
    "Sentiment prediction count",
    ["label"]
)

# Load model once at startup (critical for performance)
model = load_sentiment_model()

class PredictRequest(BaseModel):
    texts: list[str]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    start_time = time.time()
    REQUEST_COUNT.inc()

    labels = predict_sentiment(model, req.texts)

    for label in labels:
        SENTIMENT_COUNT.labels(label=label).inc()

    REQUEST_LATENCY.observe(time.time() - start_time)

    return {"predictions": labels}

"""" Prometheus metrics endpoint"""
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>Sentiment Analysis</title>
        </head>
        <body>
            <h2>Sentiment Analysis Demo</h2>
            <form action="/predict_ui" method="post">
                <textarea name="text" rows="4" cols="50"
                    placeholder="Insert a sentence here..."></textarea>
                <br><br>
                <button type="submit">Analyze</button>
            </form>
        </body>
    </html>
    """

@app.post("/predict_ui", response_class=HTMLResponse)
def predict_ui(text: str = Form(...)):
    start_time = time.time()
    REQUEST_COUNT.inc()

    label = predict_sentiment(model, [text])[0]
    SENTIMENT_COUNT.labels(label=label).inc()

    REQUEST_LATENCY.observe(time.time() - start_time)

    return f"""
    <html>
        <head>
            <title>Sentiment Result</title>
        </head>
        <body>
            <h2>Sentiment Analysis Result</h2>
            <p><strong>Text:</strong> {text}</p>
            <p><strong>Predicted sentiment:</strong> {label}</p>
            <br>
            <a href="/">Analyze another sentence</a>
        </body>
    </html>
    """

################
# To run the service, use:
# uvicorn src.service.main:app --host 0.0.0.0 --port 8000 --reload
#
# Example request:
# curl -X POST http://localhost:8000/predict \
#  -H "Content-Type: application/json" \
#  -d '{"texts":["Amazing stay!", "Terrible service."]}'
#
# OR
# to use the swaggerUI: http://localhost:8000/docs#/default/
################