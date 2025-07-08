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

# Sentivity B2B Secure Platform

## Overview
Sentivity is a modular, production-ready B2B platform for financial sentiment analysis, due diligence, and analytics. It features:
- Secure login (Supabase Auth)
- Automated Reddit scraping every 12h
- LLM-based RL agent for insights
- Live dashboard with analytics, market, and due diligence
- Downloadable PDF reports
- Deployable as a secure Electron desktop app or cloud web app

## Folder Structure
```
.
├── app/                # FastAPI backend
│   ├── agent/          # RL agent logic
│   ├── routes/         # API routes
│   ├── utils/          # Scraper, scheduler, PDF
│   ├── models.py       # SQLAlchemy models
│   ├── schemas.py      # Pydantic schemas
│   ├── main.py         # FastAPI entrypoint
│   └── database.py     # DB setup
├── sentivity-frontend/ # Vite + React frontend
│   ├── src/components/ # Pages & UI
│   ├── src/api/        # API utils
│   └── ...
├── electron/           # Electron desktop wrapper
│   ├── main.js
│   └── preload.js
├── requirements.txt    # Backend Python deps
├── README.md           # This file
└── ...
```

## Prerequisites
- Node.js >= 18
- Python >= 3.9
- Supabase account (for Auth & optional Postgres)
- (For Electron) OS: Windows, Mac, or Linux

## Setup Instructions

### 1. Clone & Install
```sh
git clone <repo-url>
cd sentivity-agent-b2b
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cd sentivity-frontend && npm install
cd ../electron && npm install
```

### 2. Configure Environment
- Copy `.env.example` to `.env` in both backend and frontend.
- Set your Supabase URL and Key in frontend `.env`:
  ```
  VITE_SUPABASE_URL=your_supabase_url
  VITE_SUPABASE_KEY=your_supabase_anon_key
  ```
- Set Reddit API keys and DB URL in backend `.env`.

### 3. Supabase Auth Setup
- Create a Supabase project at https://supabase.com
- Enable Email Auth (or SSO)
- Copy project URL and anon key to frontend `.env`

### 4. Database
- By default, uses SQLite (`sentivity.db`).
- For cloud, set up Supabase Postgres and update DB URL in backend.
- To migrate CSV data: use `scripts/seed_from_csv.py`.

### 5. Run Backend (FastAPI)
```sh
uvicorn app.main:app --reload
```

### 6. Run Frontend (Vite + React)
```sh
cd sentivity-frontend
npm run dev
```

### 7. Run Electron Desktop App
```sh
cd electron
npm start
```
- For production build: use `electron-builder` to package `.exe`/installer.

### 8. Reddit Scraper & Scheduler
- The backend scheduler runs automatically (APScheduler).
- To run manually: `python app/utils/reddit_scraper.py`

### 9. PDF Reports
- Download due diligence PDF from the Dashboard or `/api/agent-report/pdf?ticker=AAPL`.

## Developer Onboarding
- All code is modular and type-annotated.
- Add new routes in `app/routes/`, new pages in `sentivity-frontend/src/components/`.
- Use `.env` for all secrets.

## Troubleshooting & FAQ
- **Login fails:** Check Supabase keys and Auth settings.
- **No data:** Ensure scraper is running and DB is seeded.
- **PDF download fails:** Check backend logs for errors.
- **Electron issues:** Ensure backend and frontend are running before starting Electron in dev mode.

## Contact & Support
- [Your Name/Team]
- [Your Email or Support Link]

## License
MIT
