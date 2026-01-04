import pandas as pd
from pathlib import Path

# --------------------------------------------------
# Configuration
# --------------------------------------------------
RAW_DATA_PATH = Path("data/raw/hotel_reviews.csv")
PROCESSED_DATA_PATH = Path("data/processed/sentiment_dataset.csv")

def load_raw_dataset(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def select_relevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["reviews.text", "reviews.rating"]]
    df = df.dropna()
    df.columns = ["text", "rating"]
    return df

def rating_to_sentiment(rating: int) -> str:
    """
    Map rating index to sentiment label
    """
    if rating <= 2:
        return "negative"
    elif rating == 3:
        return "neutral"
    else:
        return "positive"

def add_sentiment_label(df: pd.DataFrame) -> pd.DataFrame:
    df["sentiment"] = df["rating"].apply(rating_to_sentiment)
    return df[["text", "sentiment"]]

def save_processed_dataset(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

"""
# Main preprocessing pipeline
"""
def run_preprocessing():
    raw_df = load_raw_dataset(RAW_DATA_PATH)
    filtered_df = select_relevant_columns(raw_df)
    labeled_df = add_sentiment_label(filtered_df)
    save_processed_dataset(labeled_df, PROCESSED_DATA_PATH)

    print("Preprocessing completed.")
    print(f"Processed dataset saved to: {PROCESSED_DATA_PATH}")
    print(labeled_df.head())

if __name__ == "__main__":
    run_preprocessing()
