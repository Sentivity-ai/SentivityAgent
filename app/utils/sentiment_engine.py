import os
import sys
from typing import List, Dict, Any
from sqlalchemy.orm import Session
import logging

# Add parent directory to path to import existing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.models import Post

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def analyze_sentiment_batch(posts: List[Post]) -> Dict[str, Any]:
    """
    Analyze sentiment for a batch of posts using existing sentiment analysis modules.
    """
    try:
        logger.info(f"Analyzing sentiment for {len(posts)} posts")
        
        # Import existing sentiment analysis modules
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            from sentSearch import analyze_ticker
        except ImportError as e:
            logger.error(f"Failed to import sentiment modules: {e}")
            raise Exception(f"Sentiment analysis modules not available: {e}")
        
        # Initialize VADER sentiment analyzer
        analyzer = SentimentIntensityAnalyzer()
        
        results = {
            "total_posts": len(posts),
            "analyzed_posts": 0,
            "average_sentiment": 0.0,
            "sentiment_distribution": {"positive": 0, "neutral": 0, "negative": 0}
        }
        
        sentiment_scores = []
        
        for post in posts:
            try:
                # Combine title and body for analysis
                text = f"{post.title} {post.body}"
                
                # Analyze sentiment using VADER
                sentiment_scores_vader = analyzer.polarity_scores(text)
                compound_score = sentiment_scores_vader['compound']
                
                # Update post sentiment score
                post.sentiment_score = compound_score
                sentiment_scores.append(compound_score)
                
                # Categorize sentiment
                if compound_score > 0.1:
                    results["sentiment_distribution"]["positive"] += 1
                elif compound_score < -0.1:
                    results["sentiment_distribution"]["negative"] += 1
                else:
                    results["sentiment_distribution"]["neutral"] += 1
                
                results["analyzed_posts"] += 1
                
            except Exception as e:
                logger.error(f"Error analyzing sentiment for post {post.id}: {e}")
                continue
        
        # Calculate average sentiment
        if sentiment_scores:
            results["average_sentiment"] = sum(sentiment_scores) / len(sentiment_scores)
        
        logger.info(f"Sentiment analysis completed. Analyzed {results['analyzed_posts']} posts")
        return results
        
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        raise e

def analyze_ticker_sentiment(ticker: str, posts: List[Post]) -> Dict[str, Any]:
    """
    Analyze sentiment for a specific ticker using existing sentSearch module.
    """
    try:
        logger.info(f"Analyzing ticker sentiment for {ticker}")
        
        # Import existing sentSearch module
        try:
            from sentSearch import analyze_ticker
        except ImportError as e:
            logger.error(f"Failed to import sentSearch: {e}")
            raise Exception(f"SentSearch module not available: {e}")
        
        # Use existing analyze_ticker function
        markdown_output, table_df = analyze_ticker(ticker, fetch_content=False)
        
        # Process results
        results = {
            "ticker": ticker,
            "markdown_output": markdown_output,
            "table_data": table_df.to_dict('records') if table_df is not None else [],
            "total_posts": len(posts)
        }
        
        logger.info(f"Ticker sentiment analysis completed for {ticker}")
        return results
        
    except Exception as e:
        logger.error(f"Ticker sentiment analysis failed for {ticker}: {e}")
        raise e

def generate_due_diligence_report(ticker: str, posts: List[Post]) -> Dict[str, Any]:
    """
    Generate due diligence report using existing dueDiligence module.
    """
    try:
        logger.info(f"Generating due diligence report for {ticker}")
        
        # Import existing dueDiligence module
        try:
            from dueDiligence import get_llm_report
        except ImportError as e:
            logger.error(f"Failed to import dueDiligence: {e}")
            raise Exception(f"Due diligence module not available: {e}")
        
        # Prepare data for LLM report
        stock_data = {
            "markdown_output": f"Analysis of {len(posts)} Reddit posts for {ticker}",
            "percent_change": 0.0,  # Placeholder
            "risk_level": "Medium",  # Placeholder
            "predicted_close": 0.0,  # Placeholder
            "last_actual_close": 0.0  # Placeholder
        }
        
        sentiment_data = {
            "markdown_output": f"Sentiment analysis for {ticker}",
            "table_summary": "Sentiment data summary"
        }
        
        hive_news = "General financial market news"
        sector_data_str = "Sector sentiment data"
        
        # Generate LLM report
        report = get_llm_report(stock_data, sentiment_data, hive_news, sector_data_str)
        
        results = {
            "ticker": ticker,
            "report": report,
            "total_posts": len(posts),
            "generated_at": "now"
        }
        
        logger.info(f"Due diligence report generated for {ticker}")
        return results
        
    except Exception as e:
        logger.error(f"Due diligence report generation failed for {ticker}: {e}")
        raise e

def get_sentiment_trends(posts: List[Post], days: int = 30) -> Dict[str, Any]:
    """
    Calculate sentiment trends over time.
    """
    try:
        from datetime import datetime, timedelta
        
        # Group posts by date
        daily_sentiments = {}
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        for post in posts:
            if post.created_utc >= cutoff_date and post.sentiment_score is not None:
                date_key = post.created_utc.date()
                if date_key not in daily_sentiments:
                    daily_sentiments[date_key] = []
                daily_sentiments[date_key].append(post.sentiment_score)
        
        # Calculate daily averages
        trends = []
        for date, scores in sorted(daily_sentiments.items()):
            if scores:
                trends.append({
                    "date": date.isoformat(),
                    "average_sentiment": round(sum(scores) / len(scores), 3),
                    "post_count": len(scores)
                })
        
        return {
            "trends": trends,
            "analysis_period_days": days
        }
        
    except Exception as e:
        logger.error(f"Failed to calculate sentiment trends: {e}")
        raise e

def get_top_influential_posts(posts: List[Post], limit: int = 10) -> List[Dict[str, Any]]:
    """
    Get top influential posts based on score and sentiment.
    """
    try:
        # Calculate influence score (combination of Reddit score and sentiment magnitude)
        influential_posts = []
        
        for post in posts:
            if post.sentiment_score is not None:
                # Influence score = Reddit score * sentiment magnitude
                influence_score = post.score * abs(post.sentiment_score)
                
                influential_posts.append({
                    "id": post.id,
                    "title": post.title,
                    "subreddit": post.subreddit,
                    "author": post.author,
                    "score": post.score,
                    "sentiment_score": post.sentiment_score,
                    "influence_score": influence_score,
                    "created_utc": post.created_utc,
                    "url": post.url
                })
        
        # Sort by influence score and return top posts
        influential_posts.sort(key=lambda x: x["influence_score"], reverse=True)
        return influential_posts[:limit]
        
    except Exception as e:
        logger.error(f"Failed to get influential posts: {e}")
        raise e 