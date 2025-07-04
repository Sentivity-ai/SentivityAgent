#!/usr/bin/env python3
"""
Startup script for Sentivity B2B FastAPI Backend
This script starts the backend without importing the existing Gradio modules.
"""

import uvicorn
import os
import sys
from pathlib import Path

# Add the app directory to Python path
current_dir = Path(__file__).parent
app_dir = current_dir / "app"
sys.path.insert(0, str(app_dir))

def main():
    """Start the FastAPI backend server"""
    
    print("🚀 Starting Sentivity B2B Backend...")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🔍 Alternative Docs: http://localhost:8000/redoc")
    print("💚 Health Check: http://localhost:8000/")
    print("=" * 50)
    
    # Start the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        reload_dirs=[str(app_dir)]
    )

if __name__ == "__main__":
    main() 