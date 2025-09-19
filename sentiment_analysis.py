import sqlite3
from transformers import pipeline

# Load the FinancialBERT sentiment model
try:
    sentiment_pipeline = pipeline("text-classification", model="yiyanghkust/finbert-tone")
except Exception as e:
    print(f"Error loading sentiment model: {e}")
    exit()

# Connect to the database
conn = sqlite3.connect("stock_data.db")
cursor = conn.cursor()

# Fetch news articles without sentiment scores
cursor.execute("SELECT Date, Title FROM news_articles WHERE Sentiment IS NULL")
news_articles = cursor.fetchall()

if news_articles:
    batch_size = 50  # Adjust based on system capacity
    titles = [title for _, title in news_articles]  

    # Process articles in batches
    for i in range(0, len(titles), batch_size):
        batch_titles = titles[i:i + batch_size]
        batch_results = sentiment_pipeline(batch_titles)

        # Update the database with sentiment scores
        for (date, title), result in zip(news_articles[i:i + batch_size], batch_results):
            label = result['label']  # 'Positive', 'Neutral', 'Negative'
            sentiment_score = {"Positive": 1, "Neutral": 0, "Negative": -1}.get(label, 0)

            cursor.execute(
                "UPDATE news_articles SET Sentiment = ? WHERE Date = ? AND Title = ? AND Sentiment IS NULL",
                (sentiment_score, date, title),
            )
    
    conn.commit()
    print("Sentiment analysis completed and updated in database.")
else:
    print("No new news articles to analyze.")

conn.close()
