---
title: DemoAgentStockPredictor
emoji: 😻
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 5.34.2
app_file: app.py
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# 🛠️ Sentivity B2B Secure Platform — Backend

A secure, modular backend using FastAPI + SQLite that powers the Sentivity B2B client portal. The system supports authenticated access for clients, sentiment analysis & predictions using Reddit data, LLM-based due diligence generation, and persistent storage via SQLite (upgradeable to PostgreSQL).

## 🎯 Features

### ✅ Core Functionality
- **🔐 Authenticated access for clients** (Next Phase)
- **📊 Sentiment analysis & predictions** using Reddit data
- **🧠 LLM-based due diligence generation**
- **🗃️ Persistent storage** via SQLite (upgradeable to PostgreSQL)
- **⚙️ Modular microservice architecture** for scraping, storing, and serving insights

### ✅ Reddit Scraping Service
- Runs every 12h via GitHub Action or Render cron
- Uses PRAW or Pushshift to scrape top posts by subreddit or ticker
- Stores in SQLite posts table with sentiment analysis

### ✅ Sentiment + Due Diligence Engine
- Accepts ticker and date range
- Pulls relevant posts from database
- Sends batch to LSTM/HuggingFace model
- Returns sentiment breakdown and optional due diligence report

### ✅ Modular API Routes
- `/scrape/run` — triggers manual scrape
- `/posts/` — fetch or search scraped posts
- `/sentiment/predict` — runs sentiment on ticker
- `/sentiment/due-diligence` — returns PDF-like summary

## 📁 Project Structure

```
sentivity_backend/
│
├── app/
│   ├── main.py               # FastAPI entrypoint
│   ├── database.py           # SQLite setup and session management
│   ├── models.py             # SQLAlchemy models (Post, User, Log, etc.)
│   ├── schemas.py            # Pydantic request/response models
│   ├── routes/
│   │   ├── scrape.py         # Reddit scraper trigger route
│   │   ├── posts.py          # CRUD access to posts table
│   │   └── sentiment.py      # LLM-based sentiment/due diligence
│   └── utils/
│       ├── sentiment_engine.py  # Connects to HF or LSTM model
│       └── reddit_scraper.py    # Scheduled post scraping logic
│
├── requirements.txt          # Dependency list
├── Dockerfile                # Containerized deployment
├── README.md
└── .env                      # Environment secrets
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip
- Reddit API credentials (for scraping)

### Installation

1. **Clone and setup environment**
```bash
# Create environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Set up environment variables**
```bash
# Create .env file
cp .env.example .env

# Edit .env with your credentials
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
REDDIT_USER_AGENT=your_user_agent
OPENAI_API_KEY=your_openai_api_key
```

3. **Run the backend**
```bash
# Start FastAPI server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

4. **Access the API**
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/
- Alternative Docs: http://localhost:8000/redoc

## 🌐 API Endpoints

### Health & Status
| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | Health check |
| `/docs` | GET | API documentation |

### Scraping
| Route | Method | Description |
|-------|--------|-------------|
| `/scrape/run` | POST | Trigger Reddit scraping |
| `/scrape/status` | GET | Get scraping status |
| `/scrape/manual/{subreddit}` | POST | Manual scrape for subreddit |

### Posts
| Route | Method | Description |
|-------|--------|-------------|
| `/posts/` | GET | Fetch all posts with filtering |
| `/posts/{ticker}` | GET | Fetch posts by ticker |
| `/posts/stats/{ticker}` | GET | Get ticker statistics |
| `/posts/search/` | GET | Search posts by content |
| `/posts/count/` | GET | Get post count with filters |

### Sentiment Analysis
| Route | Method | Description |
|-------|--------|-------------|
| `/sentiment/predict` | POST | Analyze sentiment for ticker |
| `/sentiment/due-diligence` | POST | Generate due diligence report |
| `/sentiment/trends/{ticker}` | GET | Get sentiment trends over time |

## 📊 Database Schema

### Posts Table
```sql
CREATE TABLE posts (
    id INTEGER PRIMARY KEY,
    ticker TEXT,
    title TEXT,
    body TEXT,
    subreddit TEXT,
    author TEXT,
    score INTEGER,
    num_comments INTEGER,
    created_utc TIMESTAMP,
    sentiment_score REAL,
    url TEXT,
    is_self BOOLEAN,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
```

### Users Table (Future)
```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    email TEXT UNIQUE,
    hashed_password TEXT,
    role TEXT,
    is_active BOOLEAN,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
```

### Logs Table
```sql
CREATE TABLE logs (
    id INTEGER PRIMARY KEY,
    event TEXT,
    level TEXT,
    message TEXT,
    metadata TEXT,
    timestamp TIMESTAMP
);
```

## 🔧 Configuration

### Environment Variables
```bash
# Database
DATABASE_URL=sqlite:///./sentivity.db

# Reddit API
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=your_user_agent

# OpenAI API
OPENAI_API_KEY=your_openai_api_key

# YouTube API (optional)
YOUTUBE_API_KEY=your_youtube_api_key
```

### Scraping Configuration
- **Default time periods**: 1d, 7d, 30d, 1y
- **Default post limit**: 100 posts per subreddit
- **Scheduled scraping**: Every 12 hours
- **Subreddits**: wallstreetbets, investing, stocks, StockMarket, finance

## 🚀 Deployment

### Local Development
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Production (Render/Railway)
```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

### Docker Deployment
```bash
# Build image
docker build -t sentivity-backend .

# Run container
docker run -p 8000:8000 sentivity-backend
```

## 🔄 Integration with Existing Modules

The FastAPI backend seamlessly integrates with your existing Python modules:

- **`redditScraper.py`** → `/scrape/` endpoints
- **`sentSearch.py`** → `/sentiment/predict` endpoint
- **`dueDiligence.py`** → `/sentiment/due-diligence` endpoint
- **`stockPred.py`** → Future integration for stock predictions
- **`financialHive.py`** → Future integration for market analysis
- **`sectorSent.py`** → Future integration for sector sentiment

## 📈 Usage Examples

### Trigger Reddit Scraping
```bash
curl -X POST "http://localhost:8000/scrape/run" \
  -H "Content-Type: application/json" \
  -d '{
    "subreddit": "wallstreetbets",
    "ticker": "TSLA",
    "time_period": "7d",
    "limit": 100
  }'
```

### Get Posts for a Ticker
```bash
curl "http://localhost:8000/posts/TSLA?limit=50&days_back=7"
```

### Analyze Sentiment
```bash
curl -X POST "http://localhost:8000/sentiment/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "TSLA",
    "start_date": "2024-01-01T00:00:00Z",
    "end_date": "2024-01-31T23:59:59Z"
  }'
```

### Generate Due Diligence Report
```bash
curl -X POST "http://localhost:8000/sentiment/due-diligence" \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "TSLA",
    "include_due_diligence": true
  }'
```

## 🔮 Future Enhancements

### Phase 2: Authentication
- [ ] Supabase Auth integration
- [ ] JWT token management
- [ ] Role-based access control
- [ ] Protected routes

### Phase 3: Advanced Features
- [ ] PostgreSQL migration
- [ ] Email alerts on sentiment spikes
- [ ] Exportable PDF reports
- [ ] Custom models per client vertical
- [ ] Real-time WebSocket updates

### Phase 4: Scaling
- [ ] Redis caching
- [ ] Background job queues
- [ ] Microservice architecture
- [ ] Kubernetes deployment

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the API documentation at `/docs`
- Review the logs in the database

---

**Built with ❤️ for the Sentivity B2B Platform**
