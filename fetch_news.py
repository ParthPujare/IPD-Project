import sqlite3
import feedparser
from datetime import datetime
import time

# Prioritize Adani-related news over general renewable energy news
ADANI_KEYWORDS = [
    "Adani Green Energy",
    "Adani Group",
    "Gautam Adani",
    "Adani power",
    "Adani energy"
]

RENEWABLE_KEYWORDS = [
    "renewable energy",
    "solar energy",
    "wind energy",
    "hydroelectric power",
    "green energy",
    "sustainable energy"
]

BASE_URL = "https://news.google.com/rss/search?q={}"

conn = sqlite3.connect("stock_data.db")
cursor = conn.cursor()

# Fetch existing news titles and links to prevent duplicates
cursor.execute("SELECT Title, Link FROM news_articles")
existing_articles = set(cursor.fetchall())  # Stores (Title, Link) as a set

news_entries = set()

# First, fetch Adani-related news
for keyword in ADANI_KEYWORDS + RENEWABLE_KEYWORDS:
    url = BASE_URL.format(keyword.replace(" ", "+"))
    news_feed = feedparser.parse(url)

    for entry in news_feed.entries:
        date = datetime(*entry.published_parsed[:6]).strftime('%Y-%m-%d')
        title = entry.title
        source = entry.source["title"] if "source" in entry else "Unknown"
        link = entry.link

        # Only add if not already in the database
        if (title, link) not in existing_articles:
            news_entries.add((date, title, source, None, link))

    time.sleep(2)  # Prevents request limits

# Insert only new articles
if news_entries:
    cursor.executemany("INSERT INTO news_articles (Date, Title, Source, Sentiment, Link) VALUES (?, ?, ?, ?, ?)", list(news_entries))
    conn.commit()
    print(f"{len(news_entries)} new articles added to the database!")
else:
    print("No new articles found.")

conn.close()
