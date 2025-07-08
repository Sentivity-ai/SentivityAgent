from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def generate_pdf(report_data, filename):
    c = canvas.Canvas(filename, pagesize=letter)
    c.drawString(100, 750, f"Sentivity Agent Report for {report_data['ticker']}")
    c.drawString(100, 730, f"Recommendation: {report_data['recommendation']}")
    c.drawString(100, 710, f"Summary: {report_data.get('summary', 'N/A')}")
    c.save() 