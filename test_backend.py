#!/usr/bin/env python3
"""
Test script for FastAPI backend
"""

import requests
import time
import sys
import os

def test_backend():
    """Test the FastAPI backend endpoints"""
    
    base_url = "http://localhost:8000"
    
    print("🧪 Testing Sentivity B2B Backend...")
    print("=" * 50)
    
    # Test health check
    try:
        print("1. Testing health check...")
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")
    
    # Test posts endpoint
    try:
        print("\n2. Testing posts endpoint...")
        response = requests.get(f"{base_url}/posts/")
        if response.status_code == 200:
            print("✅ Posts endpoint working")
            posts = response.json()
            print(f"   Found {len(posts)} posts")
        else:
            print(f"❌ Posts endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Posts endpoint error: {e}")
    
    # Test scrape status
    try:
        print("\n3. Testing scrape status...")
        response = requests.get(f"{base_url}/scrape/status")
        if response.status_code == 200:
            print("✅ Scrape status endpoint working")
            status = response.json()
            print(f"   Total posts: {status.get('total_posts', 0)}")
        else:
            print(f"❌ Scrape status failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Scrape status error: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Backend test completed!")

if __name__ == "__main__":
    test_backend() 