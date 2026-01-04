import json
from pathlib import Path
import pandas as pd

from src.preprocessing.preprocess import run_preprocessing, PROCESSED_DATA_PATH
from src.model.sentiment_model import load_sentiment_model, predict_sentiment

ARTIFACTS_DIR = Path("artifacts")
METRICS_PATH = ARTIFACTS_DIR / "training_metrics.json"

def run_training_pipeline(sample_size: int = 50):
    # Ensure processed dataset exists
    run_preprocessing()

    # Load processed data
    df = pd.read_csv(PROCESSED_DATA_PATH).dropna()
    if df.empty:
        raise ValueError("Processed dataset is empty.")

    # Load model
    model = load_sentiment_model()

    # Predict on a sample and compute simple agreement
    sample_df = df.sample(min(sample_size, len(df)), random_state=42)
    preds = predict_sentiment(model, sample_df["text"].tolist())

    # agreement is not a “true model accuracy” here, but it’s a useful signal to understand if the behavior is consistent
    agreement = sum(p == t for p, t in zip(preds, sample_df["sentiment"])) / len(preds)

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    metrics = {"sample_size": len(preds), "label_agreement": round(agreement, 4)}
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print("Training pipeline finished.")
    print(metrics)
    print(f"Saved metrics to: {METRICS_PATH}")

if __name__ == "__main__":
    run_training_pipeline()
