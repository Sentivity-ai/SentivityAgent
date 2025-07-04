from sqlalchemy import Column, Integer, String, Text, DateTime, Float, Boolean
from sqlalchemy.sql import func
from app.database import Base

class Post(Base):
    """Reddit post model for storing scraped data"""
    __tablename__ = "posts"
    
    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String(10), index=True)  # Stock ticker symbol
    title = Column(Text)
    body = Column(Text)
    subreddit = Column(String(100))
    author = Column(String(100))
    score = Column(Integer, default=0)
    num_comments = Column(Integer, default=0)
    created_utc = Column(DateTime)
    sentiment_score = Column(Float, nullable=True)
    url = Column(String(500))
    is_self = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class User(Base):
    """User model for authentication (future implementation)"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True)
    hashed_password = Column(String(255))
    role = Column(String(50), default="client")  # 'client' or 'admin'
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class Log(Base):
    """System log model for tracking events"""
    __tablename__ = "logs"
    
    id = Column(Integer, primary_key=True, index=True)
    event = Column(String(255))
    level = Column(String(20), default="INFO")  # INFO, WARNING, ERROR
    message = Column(Text)
    log_metadata = Column(Text, nullable=True)  # JSON string for additional data
    timestamp = Column(DateTime(timezone=True), server_default=func.now()) 