import sqlite3
import requests
from bs4 import BeautifulSoup
import feedparser

# ✅ STEP 1: SET UP DATABASE (Ensuring Uniqueness)
def setup_database():
    conn = sqlite3.connect("historical_news.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS news (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            published_date TEXT,
            summary TEXT,
            url TEXT UNIQUE,  -- Prevents duplicates
            source TEXT
        )
    """)
    conn.commit()
    conn.close()

# ✅ STEP 2: SCRAPE REUTERS (Using RSS Feed to Avoid `401` Errors)
def scrape_reuters_rss():
    url = "https://feeds.reuters.com/reuters/worldNews"
    feed = feedparser.parse(url)

    news_data = []
    for entry in feed.entries:
        title = entry.title
        link = entry.link
        published = entry.published if hasattr(entry, "published") else "Unknown Date"
        
        news_data.append((title, published, "", link, "Reuters"))
    
    return news_data

# ✅ STEP 3: SCRAPE BBC (Updated Selector)
def scrape_bbc():
    url = "https://www.bbc.com/news"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f" BBC request failed with status code {response.status_code}")
        return []
    
    soup = BeautifulSoup(response.text, "html.parser")
    articles = soup.find_all("a", class_="gs-c-promo-heading")  # Check class name

    news_data = []
    for article in articles:
        title = article.get_text(strip=True)
        link = article["href"]
        if link.startswith("/"):
            link = "https://www.bbc.com" + link  # Convert relative URLs to absolute
        
        news_data.append((title, "Unknown Date", "", link, "BBC"))
    
    return news_data

# ✅ STEP 4: SCRAPE GUARDIAN (Updated Selector)
def scrape_guardian():
    url = "https://www.theguardian.com/international"
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f" The Guardian request failed with status code {response.status_code}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    articles = soup.find_all("a", class_="dcr-1ixbliq")  # Check class name

    news_data = []
    for article in articles:
        title = article.get_text(strip=True)
        link = article["href"]

        news_data.append((title, "Unknown Date", "", link, "The Guardian"))

    return news_data

# ✅ STEP 5: SAVE DATA TO DATABASE (Avoids Duplicates)
def save_to_database(news_data):
    conn = sqlite3.connect("historical_news.db")
    cursor = conn.cursor()
    
    cursor.executemany("""
        INSERT OR IGNORE INTO news (title, published_date, summary, url, source)
        VALUES (?, ?, ?, ?, ?)
    """, news_data)

    conn.commit()
    conn.close()

# ✅ STEP 6: MAIN FUNCTION TO RUN ALL SCRAPERS
def main():
    setup_database()

    sources = {
        "BBC": scrape_bbc(),
        "Reuters": scrape_reuters_rss(),
        "Guardian": scrape_guardian(),
    }

    for source, news_data in sources.items():
        print(f"Saving {len(news_data)} articles from {source}...")
        save_to_database(news_data)

    print(" News Scraping Completed!")

if __name__ == "__main__":
    main()
