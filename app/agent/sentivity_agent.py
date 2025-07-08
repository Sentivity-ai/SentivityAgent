import pandas as pd
from transformers import pipeline
from app.models import Post

class SentivityAgent:
    def __init__(self, db_session):
        self.db = db_session
        self.sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    def load_recent_posts(self, ticker=None, limit=100):
        query = self.db.query(Post)
        if ticker:
            query = query.filter(Post.ticker == ticker)
        return query.order_by(Post.timestamp.desc()).limit(limit).all()

    def score_posts(self, posts):
        for post in posts:
            result = self.sentiment_model(post.content[:512])[0]
            post.sentiment = result['score'] if result['label'] == 'POSITIVE' else -result['score']
        self.db.commit()

    def summarize_clusters(self, posts):
        clusters = {'buy': [], 'hold': [], 'sell': []}
        for post in posts:
            if post.sentiment > 0.3:
                clusters['buy'].append(post)
            elif post.sentiment < -0.3:
                clusters['sell'].append(post)
            else:
                clusters['hold'].append(post)
        return clusters

    def get_recommendation(self, clusters):
        counts = {k: len(v) for k, v in clusters.items()}
        return max(counts, key=counts.get)

    def generate_report(self, ticker):
        posts = self.load_recent_posts(ticker)
        self.score_posts(posts)
        clusters = self.summarize_clusters(posts)
        recommendation = self.get_recommendation(clusters)
        return {
            "ticker": ticker,
            "recommendation": recommendation,
            "clusters": {k: [p.content for p in v] for k, v in clusters.items()}
        } 