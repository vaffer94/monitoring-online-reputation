from transformers import pipeline

# Step 1: Load a pre-trained sentiment analysis pipeline
# This model is optimized for social media (Twitter) sentiment
sentiment_classifier = pipeline(
    task="sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)

# Step 2: Example social media texts
# Since I didn't find a suitable dataset, I used some sample of hotel reviews at: https://www.kaggle.com/datasets/datafiniti/hotel-reviews?select=Datafiniti_Hotel_Reviews.csv
texts = [
    "Our experience at Rancho Valencia was absolutely perfect from beginning to end!!!! We felt special and very happy during our stayed. I would come back in a heart beat!!!",
    "Bad: Room size. Good: The close proximity to TD garden",
    "We had no choice but to stay here when a tornado hit the area and most of Vineland was without power. They charged us 190 for one night, wouldn't accept my AAA card and after leaving, we found out, we had been charged for 4 cats (which we don't own) and for another person."
]

# Step 3: Run inference
results = sentiment_classifier(texts)

# Step 4: Print results
for text, result in zip(texts, results):
    print("Text:", text)
    print("Predicted sentiment:", result["label"])
    print("Confidence score:", round(result["score"], 3))
    print("-" * 50)
