from transformers import pipeline
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# Configuration
# --------------------------------------------------
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
PROCESSED_DATA_PATH = Path("data/processed/sentiment_dataset.csv")

def load_sentiment_model():
    return pipeline(
        task="sentiment-analysis",
        model=MODEL_NAME
    )

def predict_sentiment(model, texts: list[str]) -> list[str]:
    """
    Run inference on texts
    """
    predictions = model(texts)
    return [pred["label"] for pred in predictions]

def validate_model_on_sample(sample_size: int = 10):
    df = pd.read_csv(PROCESSED_DATA_PATH)

    # Sample rows (keeps original indices)
    sample_df = df.sample(sample_size, random_state=42)

    model = load_sentiment_model()

    predicted = predict_sentiment(model, sample_df["text"].tolist())

    sample_df["predicted_sentiment"] = predicted

    # Merge predictions back into the full dataframe
    df.loc[sample_df.index, "predicted_sentiment"] = sample_df["predicted_sentiment"]

    # Save updated dataset (overwrite processed CSV)
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    print("Model validation sample:")
    print(sample_df[["text", "sentiment", "predicted_sentiment"]])

# --------------------------------------------------
# Main validation routine
# --------------------------------------------------
if __name__ == "__main__":
    validate_model_on_sample(28)
