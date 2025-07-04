#!/usr/bin/env python3
"""
Full Stack Runner for Sentivity B2B Platform
Runs both the FastAPI backend and Gradio frontend
"""

import subprocess
import time
import sys
import os
from pathlib import Path

def main():
    """Run both backend and frontend"""
    
    print("🚀 Starting Sentivity B2B Full Stack Platform...")
    print("=" * 60)
    print("📊 FastAPI Backend: http://localhost:8000")
    print("📖 API Docs: http://localhost:8000/docs")
    print("🎨 Gradio Frontend: http://localhost:7860")
    print("=" * 60)
    
    # Start backend in background
    print("🔧 Starting FastAPI Backend...")
    backend_process = subprocess.Popen([
        sys.executable, "start_backend.py"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait a moment for backend to start
    time.sleep(3)
    
    # Start frontend in background
    print("🎨 Starting Gradio Frontend...")
    frontend_process = subprocess.Popen([
        sys.executable, "gradio_app.py"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    print("\n✅ Both services are starting...")
    print("⏳ Please wait a moment for services to fully load...")
    print("\n🌐 Access your applications:")
    print("   • Frontend (Gradio): http://localhost:7860")
    print("   • Backend (FastAPI): http://localhost:8000")
    print("   • API Documentation: http://localhost:8000/docs")
    
    try:
        # Keep the script running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down services...")
        backend_process.terminate()
        frontend_process.terminate()
        print("✅ Services stopped")

if __name__ == "__main__":
    main() 