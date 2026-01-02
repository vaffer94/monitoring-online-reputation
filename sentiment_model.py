from transformers import pipeline

# Step 1: Load a pre-trained sentiment analysis pipeline
# This model is optimized for social media (Twitter) sentiment
sentiment_classifier = pipeline(
    task="sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)

# Step 2: Example social media texts
texts = [
    "I love this company, their service is amazing!",
    "The product is okay, nothing special.",
    "Terrible experience, I will never use this service again."
]

# Step 3: Run inference
results = sentiment_classifier(texts)

# Step 4: Print results
for text, result in zip(texts, results):
    print("Text:", text)
    print("Predicted sentiment:", result["label"])
    print("Confidence score:", round(result["score"], 3))
    print("-" * 50)
