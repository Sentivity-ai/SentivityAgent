#!/usr/bin/env python3
"""
FastAPI Backend Entry Point for Sentivity B2B Platform
This runs the FastAPI backend separately from the Gradio frontend.
"""

import uvicorn
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

if __name__ == "__main__":
    print("🚀 Starting Sentivity B2B Backend...")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🔍 Alternative Docs: http://localhost:8000/redoc")
    print("💚 Health Check: http://localhost:8000/")
    print("=" * 50)
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 