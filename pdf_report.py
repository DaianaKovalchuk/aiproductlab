import io
from typing import List, Optional
from datetime import datetime
from dataclasses import dataclass

from fpdf import FPDF
from semantic_analyzer import AnalysisResult, SentenceScore

@dataclass
class FactCheckResult:
    sentence: str
    status: str
    similarity: Optional[float]
    source_title: Optional[str]
    source_snippet: Optional[str]
    source_url: Optional[str]
    sentence_numbers: List[str]
    source_numbers: List[str]
    numbers_status: str
    explanation: str

def build_pdf_bytes(
    question: str, 
    answer: str, 
    result: AnalysisResult, 
    fact_results: List[FactCheckResult],
    risk_threshold: float = 60.0
) -> bytes:
    """
    Максимально простая версия PDF.
    """
    pdf = FPDF()
    pdf.add_page()
    
    # Только базовый шрифт
    pdf.set_font('helvetica', '', 10)
    
    # Заголовок
    pdf.set_font('helvetica', 'B', 16)
    pdf.cell(0, 10, 'Hallucination Report', 0, 1, 'C')
    pdf.ln(5)
    
    # Дата
    pdf.set_font('helvetica', '', 8)
    pdf.cell(0, 5, f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 1)
    pdf.ln(5)
    
    # Риск
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 7, f'Risk: {result.overall_risk:.1f}%', 0, 1)
    pdf.ln(3)
    
    # Статистика
    pdf.set_font('helvetica', 'B', 10)
    pdf.cell(0, 6, 'Statistics:', 0, 1)
    pdf.set_font('helvetica', '', 9)
    pdf.cell(0, 5, f'Sentences: {result.metadata["num_sentences"]}', 0, 1)
    pdf.cell(0, 5, f'Avg similarity: {result.metadata["mean_sentence_similarity"]:.2f}', 0, 1)
    pdf.ln(5)
    
    # Факты
    if fact_results:
        pdf.set_font('helvetica', 'B', 10)
        pdf.cell(0, 6, 'Fact Check:', 0, 1)
        pdf.ln(2)
        
        confirmed = sum(1 for fr in fact_results if fr.status == "confirmed")
        partial = sum(1 for fr in fact_results if fr.status == "partial")
        contradicted = sum(1 for fr in fact_results if fr.status == "contradicted")
        no_source = sum(1 for fr in fact_results if fr.status == "no_source")
        
        pdf.set_font('helvetica', '', 9)
        pdf.cell(0, 5, f'Confirmed: {confirmed}', 0, 1)
        pdf.cell(0, 5, f'Partial: {partial}', 0, 1)
        pdf.cell(0, 5, f'Contradicted: {contradicted}', 0, 1)
        pdf.cell(0, 5, f'No source: {no_source}', 0, 1)
        pdf.ln(5)
        
        # Детали
        for i, fr in enumerate(fact_results[:5]):  # Только первые 5
            pdf.set_font('helvetica', 'B', 9)
            pdf.cell(0, 5, f'{i+1}. {fr.status}', 0, 1)
            pdf.set_font('helvetica', '', 8)
            
            # Безопасный вывод текста - только первые 100 символов
            text = fr.sentence[:100] + ('...' if len(fr.sentence) > 100 else '')
            pdf.multi_cell(0, 4, text)
            pdf.ln(2)
    
    # Получаем PDF
    pdf_bytes = pdf.output(dest='S')
    
    # Конвертируем в bytes
    if isinstance(pdf_bytes, bytes):
        return pdf_bytes
    elif isinstance(pdf_bytes, bytearray):
        return bytes(pdf_bytes)
    elif isinstance(pdf_bytes, str):
        return pdf_bytes.encode('latin-1')
    else:
        return bytes(pdf_bytes)
