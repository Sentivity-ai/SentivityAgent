from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import os
import sys
import re
import requests
try:
    import openai
except ImportError:
    openai = None

# Add parent directory to path to import existing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.database import get_db
from app.models import Post, Log
from app.schemas import SentimentRequest, SentimentResponse, DueDiligenceResponse
from app.utils.sentiment_engine import analyze_sentiment_batch
from pydantic import BaseModel

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    context: Optional[str] = None

@router.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(
    request: SentimentRequest,
    db: Session = Depends(get_db)
):
    """
    Analyze sentiment for a specific ticker using scraped Reddit posts.
    Integrates with existing sentiment analysis modules.
    """
    try:
        # Log the sentiment analysis request
        log_entry = Log(
            event="sentiment_analysis",
            level="INFO",
            message=f"Sentiment analysis requested for {request.ticker}",
            log_metadata=f"start_date: {request.start_date}, end_date: {request.end_date}"
        )
        db.add(log_entry)
        db.commit()
        
        # Get posts for the ticker
        query = db.query(Post).filter(Post.ticker.ilike(f"%{request.ticker.upper()}%"))
        
        if request.start_date:
            query = query.filter(Post.created_utc >= request.start_date)
        
        if request.end_date:
            query = query.filter(Post.created_utc <= request.end_date)
        
        posts = query.order_by(Post.created_utc.desc()).all()
        
        if not posts:
            return SentimentResponse(
                ticker=request.ticker.upper(),
                total_posts=0,
                average_sentiment=0.0,
                sentiment_distribution={"positive": 0, "neutral": 0, "negative": 0},
                top_posts=[],
                analysis_timestamp=datetime.utcnow()
            )
        
        # Analyze sentiment using existing modules
        sentiment_results = await analyze_sentiment_batch(posts)
        
        # Calculate statistics
        sentiment_scores = [p.sentiment_score for p in posts if p.sentiment_score is not None]
        average_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        
        # Sentiment distribution
        positive = len([s for s in sentiment_scores if s > 0.1])
        negative = len([s for s in sentiment_scores if s < -0.1])
        neutral = len(sentiment_scores) - positive - negative
        
        # Top posts by sentiment score
        top_posts = []
        for post in sorted(posts, key=lambda x: abs(x.sentiment_score or 0), reverse=True)[:10]:
            top_posts.append({
                "id": post.id,
                "title": post.title,
                "subreddit": post.subreddit,
                "sentiment_score": post.sentiment_score,
                "score": post.score,
                "created_utc": post.created_utc,
                "url": post.url
            })
        
        return SentimentResponse(
            ticker=request.ticker.upper(),
            total_posts=len(posts),
            average_sentiment=round(average_sentiment, 3),
            sentiment_distribution={
                "positive": positive,
                "neutral": neutral,
                "negative": negative
            },
            top_posts=top_posts,
            analysis_timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        # Log the error
        log_entry = Log(
            event="sentiment_error",
            level="ERROR",
            message=f"Sentiment analysis failed for {request.ticker}: {str(e)}",
            log_metadata=f"start_date: {request.start_date}, end_date: {request.end_date}"
        )
        db.add(log_entry)
        db.commit()
        
        raise HTTPException(status_code=500, detail=f"Sentiment analysis failed: {str(e)}")

@router.post("/due-diligence", response_model=DueDiligenceResponse)
async def generate_due_diligence(
    request: SentimentRequest,
    db: Session = Depends(get_db)
):
    """
    Generate comprehensive due diligence report using LLM.
    Integrates with existing dueDiligence.py functionality.
    """
    try:
        # Log the due diligence request
        log_entry = Log(
            event="due_diligence_request",
            level="INFO",
            message=f"Due diligence requested for {request.ticker}",
            log_metadata=f"include_due_diligence: {request.include_due_diligence}"
        )
        db.add(log_entry)
        db.commit()
        
        # Import the existing due diligence function
        try:
            from dueDiligence import perform_due_diligence
        except ImportError:
            raise HTTPException(status_code=500, detail="Due diligence module not available")
        
        # Get posts for analysis
        query = db.query(Post).filter(Post.ticker.ilike(f"%{request.ticker.upper()}%"))
        
        if request.start_date:
            query = query.filter(Post.created_utc >= request.start_date)
        
        if request.end_date:
            query = query.filter(Post.created_utc <= request.end_date)
        
        posts = query.order_by(Post.created_utc.desc()).limit(100).all()
        
        if not posts:
            return DueDiligenceResponse(
                ticker=request.ticker.upper(),
                report="No posts found for analysis.",
                summary="Insufficient data for due diligence analysis.",
                risk_level="Unknown",
                recommendation="No recommendation available due to lack of data.",
                generated_at=datetime.utcnow()
            )
        
        # Generate due diligence report using existing functionality
        # Note: This is a simplified integration - you may need to adapt the existing function
        try:
            # For now, we'll create a basic report structure
            # In a full implementation, you'd call the actual due diligence function
            
            # Calculate basic metrics
            sentiment_scores = [p.sentiment_score for p in posts if p.sentiment_score is not None]
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
            
            # Determine risk level based on sentiment
            if avg_sentiment > 0.2:
                risk_level = "Low"
                recommendation = "Positive sentiment suggests favorable outlook"
            elif avg_sentiment < -0.2:
                risk_level = "High"
                recommendation = "Negative sentiment indicates potential risks"
            else:
                risk_level = "Medium"
                recommendation = "Mixed sentiment suggests cautious approach"
            
            # Create report
            report = f"""
# Due Diligence Report for {request.ticker.upper()}

## Executive Summary
Analysis of {len(posts)} Reddit posts from {request.start_date or 'earliest'} to {request.end_date or 'latest'}.

## Sentiment Analysis
- Average Sentiment Score: {avg_sentiment:.3f}
- Total Posts Analyzed: {len(posts)}
- Positive Posts: {len([s for s in sentiment_scores if s > 0.1])}
- Negative Posts: {len([s for s in sentiment_scores if s < -0.1])}

## Risk Assessment
Risk Level: {risk_level}

## Recommendation
{recommendation}

## Methodology
This report is based on sentiment analysis of Reddit posts mentioning {request.ticker.upper()}.
Sentiment scores range from -1 (very negative) to +1 (very positive).
            """.strip()
            
            return DueDiligenceResponse(
                ticker=request.ticker.upper(),
                report=report,
                summary=f"Analysis of {len(posts)} posts shows {risk_level.lower()} risk with {avg_sentiment:.3f} average sentiment.",
                risk_level=risk_level,
                recommendation=recommendation,
                generated_at=datetime.utcnow()
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to generate due diligence report: {str(e)}")
        
    except Exception as e:
        # Log the error
        log_entry = Log(
            event="due_diligence_error",
            level="ERROR",
            message=f"Due diligence failed for {request.ticker}: {str(e)}",
            log_metadata=f"include_due_diligence: {request.include_due_diligence}"
        )
        db.add(log_entry)
        db.commit()
        
        raise HTTPException(status_code=500, detail=f"Due diligence failed: {str(e)}")

@router.get("/trends/{ticker}")
async def get_sentiment_trends(
    ticker: str,
    days: int = 30,
    db: Session = Depends(get_db)
):
    """
    Get sentiment trends over time for a specific ticker.
    """
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Get posts grouped by day
        posts = db.query(Post).filter(
            Post.ticker.ilike(f"%{ticker.upper()}%"),
            Post.created_utc >= cutoff_date
        ).order_by(Post.created_utc).all()
        
        # Group by date and calculate daily averages
        daily_sentiments = {}
        for post in posts:
            date_key = post.created_utc.date()
            if date_key not in daily_sentiments:
                daily_sentiments[date_key] = []
            if post.sentiment_score is not None:
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
            "ticker": ticker.upper(),
            "analysis_period_days": days,
            "trends": trends
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get trends for {ticker}: {str(e)}")

@router.post("/chat/ask")
async def chat_ask(request: ChatRequest):
    """
    Enhanced chat endpoint: pulls analytics, sentiment, and due diligence for tickers in the user message.
    Now with timeouts and error logging for robustness.
    """
    if not openai or not os.getenv('open_api_key'):
        return {"response": "LLM functionality is not available. Please set the 'open_api_key' environment variable."}
    try:
        client = openai.OpenAI(api_key=os.getenv('open_api_key'))
        user_msg = request.message
        # Detect tickers (simple regex: 1-5 uppercase letters)
        tickers = re.findall(r'\b[A-Z]{1,5}\b', user_msg)
        context_blocks = []
        for ticker in tickers:
            print(f"[Agent] Fetching data for ticker: {ticker}")
            # Fetch sentiment stats
            try:
                stats = requests.get(f"http://localhost:8000/posts/stats/{ticker}", timeout=2).json()
                print(f"[Agent] Stats for {ticker}: {stats}")
            except Exception as e:
                print(f"[Agent] Error fetching stats for {ticker}: {e}")
                stats = None
            # Fetch due diligence (reuse logic)
            try:
                dd = requests.post(f"http://localhost:8000/sentiment/due-diligence", json={"ticker": ticker, "include_due_diligence": True}, timeout=4).json()
                print(f"[Agent] Due diligence for {ticker}: {dd}")
            except Exception as e:
                print(f"[Agent] Error fetching due diligence for {ticker}: {e}")
                dd = None
            # Fetch trending words (aggregate from posts)
            try:
                posts = requests.get(f"http://localhost:8000/posts/?ticker={ticker}&limit=200", timeout=2).json()
                word_freq = {}
                for p in posts:
                    words = (p.get('title','') + ' ' + p.get('body','')).lower().split()
                    for w in words:
                        if len(w) > 3:
                            word_freq[w] = word_freq.get(w, 0) + 1
                trending = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
                print(f"[Agent] Trending words for {ticker}: {trending}")
            except Exception as e:
                print(f"[Agent] Error fetching trending words for {ticker}: {e}")
                trending = []
            # Format context
            context_blocks.append(f"""
---
Ticker: {ticker}
Sentiment Stats: {stats if stats else 'N/A'}
Due Diligence: {dd.get('report') if dd else 'N/A'}
Trending Words: {', '.join([w for w,_ in trending]) if trending else 'N/A'}
---
""")
        # If no ticker, provide general stats
        if not context_blocks:
            try:
                stats = requests.get(f"http://localhost:8000/posts/stats/AAPL", timeout=2).json()
                print(f"[Agent] General stats: {stats}")
            except Exception as e:
                print(f"[Agent] Error fetching general stats: {e}")
                stats = None
            context_blocks.append(f"General Dashboard Stats: {stats if stats else 'N/A'}")
        disclaimer = "(The following context is generated from Sentivity's analytics and models. Always verify before making financial decisions.)"
        context = disclaimer + '\n' + '\n'.join(context_blocks)
        prompt = f"Context: {context}\nUser: {user_msg}"
        print(f"[Agent] Final prompt to LLM: {prompt[:500]}...")
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful financial assistant. Always use the provided context and cite stats when possible."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=700
        )
        print(f"[Agent] LLM response: {completion.choices[0].message.content[:200]}...")
        return {"response": completion.choices[0].message.content}
    except Exception as e:
        print(f"[Agent] Error in chat_ask: {e}")
        return {"response": f"Error generating LLM response: {e}"}

@router.get("/market/overview")
async def market_overview():
    """
    Mock market overview endpoint for demo purposes.
    """
    import random, time
    tickers = ['AAPL', 'TSLA', 'MSFT', 'GOOGL', 'AMZN']
    sectors = ['TECH', 'AUTO', 'ECOM', 'CLOUD', 'SOCIAL']
    market_summary = {
        t: {
            'price': round(random.uniform(100, 400), 2),
            'change': round(random.uniform(-5, 5), 2),
            'volume': random.randint(1_000_000, 10_000_000)
        } for t in tickers
    }
    sector_trends = [
        {'sector': s, 'avg_sentiment': round(random.uniform(-0.5, 0.8), 2)} for s in sectors
    ]
    top_movers = sorted([
        {'ticker': t, 'change': market_summary[t]['change']} for t in tickers
    ], key=lambda x: abs(x['change']), reverse=True)
    return {
        'market_summary': market_summary,
        'sector_trends': sector_trends,
        'top_movers': top_movers,
        'timestamp': int(time.time())
    } 