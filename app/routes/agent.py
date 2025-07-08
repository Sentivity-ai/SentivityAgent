from fastapi import APIRouter, Depends, Query, Response
from app.agent.sentivity_agent import SentivityAgent
from app.database import get_db
from app.utils.pdf_generator import generate_pdf
import tempfile

router = APIRouter()

@router.get('/agent-report')
def agent_report(ticker: str, db=Depends(get_db)):
    agent = SentivityAgent(db)
    return agent.generate_report(ticker)

@router.get('/agent-report/pdf')
def agent_report_pdf(ticker: str, db=Depends(get_db)):
    agent = SentivityAgent(db)
    report = agent.generate_report(ticker)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        generate_pdf(report, tmp.name)
        tmp.seek(0)
        pdf_bytes = tmp.read()
    return Response(content=pdf_bytes, media_type='application/pdf', headers={
        'Content-Disposition': f'attachment; filename="sentivity_report_{ticker}.pdf"'
    }) 