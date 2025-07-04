from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc, asc
from typing import Optional, List
from datetime import datetime, timedelta

from app.database import get_db
from app.models import Post
from app.schemas import PostResponse

router = APIRouter()

@router.get("/", response_model=List[PostResponse])
async def get_posts(
    ticker: Optional[str] = Query(None, description="Filter by stock ticker"),
    subreddit: Optional[str] = Query(None, description="Filter by subreddit"),
    limit: int = Query(100, ge=1, le=1000, description="Number of posts to return"),
    offset: int = Query(0, ge=0, description="Number of posts to skip"),
    sort_by: str = Query("created_utc", description="Sort by field: created_utc, score, sentiment_score"),
    sort_order: str = Query("desc", description="Sort order: asc or desc"),
    days_back: Optional[int] = Query(None, ge=1, le=365, description="Filter posts from last N days"),
    db: Session = Depends(get_db)
):
    """
    Get posts with optional filtering and sorting.
    Supports filtering by ticker, subreddit, and date range.
    """
    try:
        query = db.query(Post)
        
        # Apply filters
        if ticker:
            query = query.filter(Post.ticker.ilike(f"%{ticker.upper()}%"))
        
        if subreddit:
            query = query.filter(Post.subreddit.ilike(f"%{subreddit}%"))
        
        if days_back:
            cutoff_date = datetime.utcnow() - timedelta(days=days_back)
            query = query.filter(Post.created_utc >= cutoff_date)
        
        # Apply sorting
        if sort_order.lower() == "asc":
            query = query.order_by(asc(getattr(Post, sort_by)))
        else:
            query = query.order_by(desc(getattr(Post, sort_by)))
        
        # Apply pagination
        posts = query.offset(offset).limit(limit).all()
        
        return posts
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch posts: {str(e)}")

@router.get("/{ticker}", response_model=List[PostResponse])
async def get_posts_by_ticker(
    ticker: str,
    limit: int = Query(100, ge=1, le=1000, description="Number of posts to return"),
    days_back: Optional[int] = Query(30, ge=1, le=365, description="Filter posts from last N days"),
    db: Session = Depends(get_db)
):
    """
    Get posts for a specific stock ticker.
    """
    try:
        query = db.query(Post).filter(Post.ticker.ilike(f"%{ticker.upper()}%"))
        
        if days_back:
            cutoff_date = datetime.utcnow() - timedelta(days=days_back)
            query = query.filter(Post.created_utc >= cutoff_date)
        
        posts = query.order_by(desc(Post.created_utc)).limit(limit).all()
        
        return posts
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch posts for {ticker}: {str(e)}")

@router.get("/stats/{ticker}")
async def get_ticker_stats(
    ticker: str,
    days_back: int = Query(30, ge=1, le=365, description="Analysis period in days"),
    db: Session = Depends(get_db)
):
    """
    Get statistics for a specific ticker including post count, sentiment distribution, etc.
    """
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        
        # Get posts for the ticker
        posts = db.query(Post).filter(
            Post.ticker.ilike(f"%{ticker.upper()}%"),
            Post.created_utc >= cutoff_date
        ).all()
        
        if not posts:
            return {
                "ticker": ticker.upper(),
                "total_posts": 0,
                "average_sentiment": 0,
                "sentiment_distribution": {"positive": 0, "neutral": 0, "negative": 0},
                "top_subreddits": [],
                "analysis_period_days": days_back
            }
        
        # Calculate statistics
        total_posts = len(posts)
        sentiment_scores = [p.sentiment_score for p in posts if p.sentiment_score is not None]
        average_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        
        # Sentiment distribution
        positive = len([s for s in sentiment_scores if s > 0.1])
        negative = len([s for s in sentiment_scores if s < -0.1])
        neutral = len(sentiment_scores) - positive - negative
        
        # Top subreddits
        subreddit_counts = {}
        for post in posts:
            subreddit_counts[post.subreddit] = subreddit_counts.get(post.subreddit, 0) + 1
        
        top_subreddits = sorted(subreddit_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "ticker": ticker.upper(),
            "total_posts": total_posts,
            "average_sentiment": round(average_sentiment, 3),
            "sentiment_distribution": {
                "positive": positive,
                "neutral": neutral,
                "negative": negative
            },
            "top_subreddits": [{"subreddit": sub, "count": count} for sub, count in top_subreddits],
            "analysis_period_days": days_back
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats for {ticker}: {str(e)}")

@router.get("/search/")
async def search_posts(
    q: str = Query(..., description="Search query"),
    ticker: Optional[str] = Query(None, description="Filter by ticker"),
    limit: int = Query(50, ge=1, le=200, description="Number of results to return"),
    db: Session = Depends(get_db)
):
    """
    Search posts by title and body content.
    """
    try:
        query = db.query(Post).filter(
            (Post.title.ilike(f"%{q}%")) | (Post.body.ilike(f"%{q}%"))
        )
        
        if ticker:
            query = query.filter(Post.ticker.ilike(f"%{ticker.upper()}%"))
        
        posts = query.order_by(desc(Post.created_utc)).limit(limit).all()
        
        return {
            "query": q,
            "results_count": len(posts),
            "posts": posts
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.get("/count/")
async def get_post_count(
    ticker: Optional[str] = Query(None, description="Filter by ticker"),
    subreddit: Optional[str] = Query(None, description="Filter by subreddit"),
    db: Session = Depends(get_db)
):
    """
    Get total count of posts with optional filtering.
    """
    try:
        query = db.query(Post)
        
        if ticker:
            query = query.filter(Post.ticker.ilike(f"%{ticker.upper()}%"))
        
        if subreddit:
            query = query.filter(Post.subreddit.ilike(f"%{subreddit}%"))
        
        count = query.count()
        
        return {
            "total_posts": count,
            "filters": {
                "ticker": ticker,
                "subreddit": subreddit
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get count: {str(e)}") 