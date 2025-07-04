from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

# Health Check Response
class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    service: str
    version: str

# Post Schemas
class PostBase(BaseModel):
    ticker: str = Field(..., max_length=10)
    title: str
    body: str
    subreddit: str = Field(..., max_length=100)
    author: str = Field(..., max_length=100)
    score: int = 0
    num_comments: int = 0
    created_utc: datetime
    sentiment_score: Optional[float] = None
    url: str = Field(..., max_length=500)
    is_self: bool = True

class PostCreate(PostBase):
    pass

class PostResponse(PostBase):
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

# Scraping Schemas
class ScrapeRequest(BaseModel):
    subreddit: str = Field(..., description="Subreddit to scrape")
    ticker: Optional[str] = Field(None, max_length=10, description="Stock ticker to filter by")
    time_period: str = Field(default="7d", description="Time period: 1d, 7d, 30d, 1y")
    limit: int = Field(default=100, ge=1, le=1000, description="Number of posts to scrape")

class ScrapeResponse(BaseModel):
    success: bool
    message: str
    posts_scraped: int
    ticker: Optional[str] = None
    subreddit: str
    timestamp: datetime

# Sentiment Analysis Schemas
class SentimentRequest(BaseModel):
    ticker: str = Field(..., max_length=10, description="Stock ticker to analyze")
    start_date: Optional[datetime] = Field(None, description="Start date for analysis")
    end_date: Optional[datetime] = Field(None, description="End date for analysis")
    include_due_diligence: bool = Field(default=False, description="Generate due diligence report")

class SentimentResponse(BaseModel):
    ticker: str
    total_posts: int
    average_sentiment: float
    sentiment_distribution: Dict[str, int]
    top_posts: List[Dict[str, Any]]
    analysis_timestamp: datetime

class DueDiligenceResponse(BaseModel):
    ticker: str
    report: str
    summary: str
    risk_level: str
    recommendation: str
    generated_at: datetime

# User Schemas (for future authentication)
class UserBase(BaseModel):
    email: str = Field(..., max_length=255)
    role: str = Field(default="client", max_length=50)

class UserCreate(UserBase):
    password: str = Field(..., min_length=8)

class UserResponse(UserBase):
    id: int
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

# Log Schemas
class LogBase(BaseModel):
    event: str = Field(..., max_length=255)
    level: str = Field(default="INFO", max_length=20)
    message: str
    log_metadata: Optional[str] = None

class LogCreate(LogBase):
    pass

class LogResponse(LogBase):
    id: int
    timestamp: datetime
    
    class Config:
        from_attributes = True 