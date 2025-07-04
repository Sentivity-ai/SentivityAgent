from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import Optional
import os
import sys
from datetime import datetime

# Add parent directory to path to import existing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.database import get_db
from app.models import Post, Log
from app.schemas import ScrapeRequest, ScrapeResponse
from app.utils.reddit_scraper import run_reddit_scrape

router = APIRouter()

@router.post("/run", response_model=ScrapeResponse)
async def trigger_scrape(
    request: ScrapeRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Trigger Reddit scraping for a specific subreddit and optionally filter by ticker.
    This endpoint integrates with the existing redditScraper.py functionality.
    """
    try:
        # Log the scraping request
        log_entry = Log(
            event="scrape_triggered",
            level="INFO",
            message=f"Scraping triggered for subreddit: {request.subreddit}, ticker: {request.ticker}",
            log_metadata=f"time_period: {request.time_period}, limit: {request.limit}"
        )
        db.add(log_entry)
        db.commit()
        
        # Run scraping in background to avoid blocking the API
        background_tasks.add_task(
            run_reddit_scrape,
            db=db,
            subreddit=request.subreddit,
            ticker=request.ticker,
            time_period=request.time_period,
            limit=request.limit
        )
        
        return ScrapeResponse(
            success=True,
            message=f"Scraping started for r/{request.subreddit}",
            posts_scraped=0,  # Will be updated by background task
            ticker=request.ticker,
            subreddit=request.subreddit,
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        # Log the error
        log_entry = Log(
            event="scrape_error",
            level="ERROR",
            message=f"Scraping failed: {str(e)}",
            log_metadata=f"subreddit: {request.subreddit}, ticker: {request.ticker}"
        )
        db.add(log_entry)
        db.commit()
        
        raise HTTPException(status_code=500, detail=f"Scraping failed: {str(e)}")

@router.get("/status")
async def get_scrape_status(db: Session = Depends(get_db)):
    """
    Get the status of recent scraping operations
    """
    try:
        # Get recent scrape logs
        recent_logs = db.query(Log).filter(
            Log.event.in_(["scrape_triggered", "scrape_completed", "scrape_error"])
        ).order_by(Log.timestamp.desc()).limit(10).all()
        
        # Get post count
        total_posts = db.query(Post).count()
        
        return {
            "total_posts": total_posts,
            "recent_operations": [
                {
                    "event": log.event,
                    "message": log.message,
                    "timestamp": log.timestamp,
                    "level": log.level
                }
                for log in recent_logs
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")

@router.post("/manual/{subreddit}")
async def manual_scrape(
    subreddit: str,
    ticker: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Manual scraping endpoint that runs immediately (not in background)
    """
    try:
        # Import the existing reddit scraper function
        from redditScraper import analyze
        
        # Run the existing analysis function
        result = analyze(subreddit, "7d")  # Default to 7 days
        
        # Log the manual scrape
        log_entry = Log(
            event="manual_scrape",
            level="INFO",
            message=f"Manual scrape completed for r/{subreddit}",
            log_metadata=f"ticker: {ticker}, result: {result}"
        )
        db.add(log_entry)
        db.commit()
        
        return {
            "success": True,
            "message": f"Manual scrape completed for r/{subreddit}",
            "result": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Manual scrape failed: {str(e)}") 