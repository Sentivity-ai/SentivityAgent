from apscheduler.schedulers.background import BackgroundScheduler
from app.utils.reddit_scraper import scrape_and_save

def start_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.add_job(scrape_and_save, 'interval', hours=12)
    scheduler.start() 