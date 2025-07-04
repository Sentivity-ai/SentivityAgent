import os
import sys
import pandas as pd
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import random

# Add app directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))
from app.models import Post
from app.database import Base

# Path to DB
DB_URL = 'sqlite:///./sentivity.db'

# Create engine and session
engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)

TICKERS = ['AAPL', 'TSLA', 'MSFT', 'GOOGL', 'AMZN']
CATEGORIES = ['TECH', 'AUTO', 'ECOM', 'CLOUD', 'SOCIAL']
SUBREDDITS = ['wallstreetbets', 'investing', 'stocks', 'techstocks', 'finance']
TEMPLATES = [
    "{ticker} is showing strong momentum this week.",
    "Is {ticker} a buy or a sell right now?",
    "Why is everyone talking about {ticker}?",
    "{ticker} earnings report analysis.",
    "Risks and opportunities for {ticker} in 2024.",
    "{ticker} vs competitors: who will win?",
    "Sentiment for {ticker} is at an all-time high.",
    "Bearish outlook for {ticker} after recent news.",
    "{ticker} to the moon or not?",
    "What are the best price targets for {ticker}?"
]

# Generate demo posts
print('Generating rich demo posts for dashboard demo...')
def generate_demo_posts(n=100):
    demo = []
    for i in range(n):
        ticker = random.choice(TICKERS)
        # Sentiment: AAPL mostly positive, TSLA mostly negative, others mixed
        if ticker == 'AAPL':
            sentiment = round(random.uniform(0.1, 1.0), 2)
        elif ticker == 'TSLA':
            sentiment = round(random.uniform(-1.0, 0.1), 2)
        else:
            sentiment = round(random.uniform(-1.0, 1.0), 2)
        subreddit = random.choice(SUBREDDITS)
        category = random.choice(CATEGORIES)
        days_ago = random.randint(0, 29)
        created_utc = datetime.utcnow() - timedelta(days=days_ago, hours=random.randint(0,23), minutes=random.randint(0,59))
        title = random.choice(TEMPLATES).format(ticker=ticker)
        body = f"This is a demo post about {ticker} in the {category} sector. {title} Posted in r/{subreddit}. Sentiment: {sentiment}."
        demo.append({
            'ticker': ticker,
            'title': title,
            'body': body,
            'subreddit': subreddit,
            'author': f'user{random.randint(1, 50)}',
            'score': random.randint(1, 2000),
            'num_comments': random.randint(0, 200),
            'created_utc': created_utc,
            'sentiment_score': sentiment,
            'url': f'https://reddit.com/demo_post_{ticker}_{i+1}',
            'category': category
        })
    return pd.DataFrame(demo)

df = generate_demo_posts(100)

# Map columns (adjust as needed)
def map_row_to_post(row):
    try:
        created = pd.to_datetime(row.get('created_utc'))
        if pd.isnull(created):
            created = datetime.utcnow()
    except Exception:
        created = datetime.utcnow()
    # If Post model supports 'category', include it
    kwargs = dict(
        ticker=row.get('ticker', ''),
        title=row.get('title', ''),
        body=row.get('body', ''),
        subreddit=row.get('subreddit', ''),
        author=row.get('author', ''),
        score=int(row.get('score', 0)),
        num_comments=int(row.get('num_comments', 0)),
        created_utc=created,
        sentiment_score=float(row.get('sentiment_score', 0)),
        url=row.get('url', ''),
    )
    if 'category' in Post.__table__.columns:
        kwargs['category'] = row.get('category', '')
    return Post(**kwargs)

# Insert posts
session = SessionLocal()
added = 0
for _, row in df.iterrows():
    exists = session.query(Post).filter_by(url=row['url'], ticker=row['ticker']).first()
    if not exists:
        post = map_row_to_post(row)
        session.add(post)
        added += 1
session.commit()

# Print summary
summary = df.groupby('ticker').agg(
    count=('ticker', 'count'),
    avg_sentiment=('sentiment_score', 'mean')
)
print(f'Seeded {added} new demo posts.')
print('Posts per ticker and average sentiment:')
print(summary)
session.close() 