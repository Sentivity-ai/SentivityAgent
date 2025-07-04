import os
import sys
import praw
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from typing import Optional, List
import logging

# Add parent directory to path to import existing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.models import Post, Log
from app.database import SessionLocal

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_reddit_scrape(
    db: Session,
    subreddit: str,
    ticker: Optional[str] = None,
    time_period: str = "7d",
    limit: int = 100
):
    """
    Run Reddit scraping and store results in database.
    Integrates with existing redditScraper.py functionality.
    """
    try:
        logger.info(f"Starting Reddit scrape for r/{subreddit}, ticker: {ticker}")
        
        # Import existing reddit scraper functions
        try:
            from redditScraper import analyze, get_top_overlapping_subreddits
        except ImportError as e:
            logger.error(f"Failed to import redditScraper: {e}")
            raise Exception(f"Reddit scraper module not available: {e}")
        
        # Calculate time period
        time_period_map = {
            "1d": 1,
            "7d": 7,
            "30d": 30,
            "1y": 365
        }
        days_back = time_period_map.get(time_period, 7)
        
        # Get overlapping subreddits for better coverage
        overlapping_subreddits = get_top_overlapping_subreddits(subreddit, top_n=5)
        subreddits_to_scrape = [subreddit] + overlapping_subreddits
        
        total_posts_scraped = 0
        
        for sub in subreddits_to_scrape:
            try:
                logger.info(f"Scraping subreddit: r/{sub}")
                
                # Use existing analyze function to get posts
                posts_data = analyze(sub, time_period)
                
                if isinstance(posts_data, dict) and 'posts' in posts_data:
                    posts = posts_data['posts']
                elif isinstance(posts_data, list):
                    posts = posts_data
                else:
                    logger.warning(f"Unexpected data format from analyze function for r/{sub}")
                    continue
                
                # Process and store posts
                for post_data in posts[:limit]:
                    try:
                        # Extract ticker from post if not provided
                        post_ticker = ticker
                        if not post_ticker:
                            # Try to extract ticker from title or body
                            post_ticker = extract_ticker_from_text(
                                post_data.get('title', '') + ' ' + post_data.get('content', '')
                            )
                        
                        if post_ticker:
                            # Create Post object
                            post = Post(
                                ticker=post_ticker.upper(),
                                title=post_data.get('title', '')[:1000],  # Limit length
                                body=post_data.get('content', '')[:5000],  # Limit length
                                subreddit=sub,
                                author=post_data.get('author', 'unknown'),
                                score=post_data.get('score', 0),
                                num_comments=post_data.get('num_comments', 0),
                                created_utc=datetime.fromtimestamp(post_data.get('created_utc', 0)),
                                sentiment_score=post_data.get('sentiment_score'),
                                url=post_data.get('url', ''),
                                is_self=post_data.get('is_self', True)
                            )
                            
                            # Check if post already exists (avoid duplicates)
                            existing_post = db.query(Post).filter(
                                Post.url == post.url,
                                Post.ticker == post.ticker
                            ).first()
                            
                            if not existing_post:
                                db.add(post)
                                total_posts_scraped += 1
                            
                    except Exception as e:
                        logger.error(f"Error processing post: {e}")
                        continue
                
                # Commit batch
                db.commit()
                logger.info(f"Scraped {len(posts)} posts from r/{sub}")
                
            except Exception as e:
                logger.error(f"Error scraping r/{sub}: {e}")
                continue
        
        # Log completion
        log_entry = Log(
            event="scrape_completed",
            level="INFO",
            message=f"Scraping completed for r/{subreddit}",
            log_metadata=f"total_posts_scraped: {total_posts_scraped}, subreddits: {subreddits_to_scrape}"
        )
        db.add(log_entry)
        db.commit()
        
        logger.info(f"Scraping completed. Total posts scraped: {total_posts_scraped}")
        return total_posts_scraped
        
    except Exception as e:
        logger.error(f"Scraping failed: {e}")
        
        # Log error
        log_entry = Log(
            event="scrape_error",
            level="ERROR",
            message=f"Scraping failed: {str(e)}",
            log_metadata=f"subreddit: {subreddit}, ticker: {ticker}"
        )
        db.add(log_entry)
        db.commit()
        
        raise e

def extract_ticker_from_text(text: str) -> Optional[str]:
    """
    Extract stock ticker from text using common patterns.
    """
    import re
    
    # Common ticker patterns
    ticker_patterns = [
        r'\$([A-Z]{1,5})\b',  # $AAPL, $TSLA
        r'\b([A-Z]{1,5})\s*stock\b',  # AAPL stock
        r'\b([A-Z]{1,5})\s*shares\b',  # AAPL shares
        r'\b([A-Z]{1,5})\s*price\b',  # AAPL price
    ]
    
    for pattern in ticker_patterns:
        matches = re.findall(pattern, text.upper())
        if matches:
            return matches[0]
    
    return None

def schedule_scraping():
    """
    Schedule regular scraping (for future implementation with cron jobs).
    """
    import schedule
    import time
    
    def job():
        """Scheduled scraping job"""
        db = SessionLocal()
        try:
            # Scrape popular financial subreddits
            subreddits = [
                "wallstreetbets",
                "investing", 
                "stocks",
                "StockMarket",
                "finance"
            ]
            
            for subreddit in subreddits:
                try:
                    run_reddit_scrape(db, subreddit, time_period="1d", limit=50)
                except Exception as e:
                    logger.error(f"Failed to scrape r/{subreddit}: {e}")
                    
        finally:
            db.close()
    
    # Schedule scraping every 12 hours
    schedule.every(12).hours.do(job)
    
    logger.info("Scheduled scraping every 12 hours")
    
    # Run the scheduler
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    # Test scraping
    db = SessionLocal()
    try:
        result = run_reddit_scrape(db, "wallstreetbets", limit=10)
        print(f"Scraped {result} posts")
    finally:
        db.close() 