import pandas as pd
import feedparser
from datetime import datetime
import time
import urllib.parse
import requests

tickers = ['LUCK', 'OGDC', 'PPL', 'ENGRO', 'HUBC', 'HBL', 'MCB', 'UBL', 'TRG', 'SYS']

print("[INFO] Gathering news headlines for PSX tickers via Google News RSS...")

news_records = []
# Headers dena zaroori hai taake Google block na kare
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

for ticker in tickers:
    print(f"Fetching news for {ticker}...")
    raw_query = f"{ticker} stock Pakistan OR PSX"
    encoded_query = urllib.parse.quote(raw_query)
    
    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-PK&gl=PK&ceid=PK:en"
    
    try:
        # Requests se data fetch karna with 10 seconds timeout
        response = requests.get(rss_url, headers=headers, timeout=10)
        if response.status_code == 200:
            feed = feedparser.parse(response.content)
            
            for entry in feed.entries:
                title = entry.get('title', '')
                published = entry.get('published', '')
                
                try:
                    pub_date = pd.to_datetime(published).strftime('%Y-%m-%d')
                except:
                    pub_date = datetime.now().strftime('%Y-%m-%d')
                    
                news_records.append({
                    'Date': pub_date,
                    'Ticker': ticker,
                    'Title': title
                })
        else:
            print(f"[WARNING] Could not fetch for {ticker}, Status code: {response.status_code}")
    except Exception as e:
        print(f"[WARNING] Timeout or error for {ticker}: {e}")
    
    time.sleep(1)

if news_records:
    df_news = pd.DataFrame(news_records)
    df_news = df_news.drop_duplicates(subset=['Title'])
    
    print(f"\n[SUCCESS] Collected {len(df_news)} news headlines!")
    print(df_news.head())
    
    df_news.to_csv("psx_news.csv", index=False)
    print("[INFO] News data saved to 'psx_news.csv'")
else:
    print("[ERROR] No news found.")