from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from datetime import datetime
import os
import sys

# Import database and models
from app.database import engine, get_db
from app.models import Base
from app.schemas import HealthResponse

# Import routes
from app.routes import scrape, posts, sentiment, agent

# Create database tables
Base.metadata.create_all(bind=engine)

# Initialize FastAPI app
app = FastAPI(
    title="Sentivity B2B Secure Platform",
    description="Backend API for Sentivity B2B client portal with sentiment analysis and due diligence capabilities",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(scrape.router, prefix="/scrape", tags=["scraping"])
app.include_router(posts.router, prefix="/posts", tags=["posts"])
app.include_router(sentiment.router, prefix="/sentiment", tags=["sentiment"])
app.include_router(agent.router, prefix="/api")

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        service="Sentivity B2B Backend",
        version="1.0.0"
    )

@app.get("/docs")
async def get_docs():
    """API documentation endpoint"""
    return {"message": "API documentation available at /docs"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True) 