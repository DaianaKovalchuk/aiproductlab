import io
from typing import List, Optional
from datetime import datetime
import os
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
    Упрощенная версия PDF-отчета.
    """
    pdf = FPDF()
    pdf.add_page()
    
    # Используем встроенный шрифт
    pdf.set_font('helvetica', '', 12)
    
    # Заголовок
    pdf.set_font('helvetica', 'B', 20)
    pdf.cell(0, 15, 'LLM Hallucination Report', 0, 1, 'C')
    pdf.set_font('helvetica', '', 10)
    pdf.cell(0, 8, f'Generated: {datetime.now().strftime("%d.%m.%Y %H:%M")}', 0, 1, 'C')
    pdf.ln(10)
    
    # Question
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 8, 'Question:', 0, 1)
    pdf.set_font('helvetica', '', 12)
    pdf.multi_cell(0, 8, question)
    pdf.ln(5)
    
    # Answer
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 8, 'Answer:', 0, 1)
    pdf.set_font('helvetica', '', 12)
    pdf.multi_cell(0, 8, answer)
    pdf.ln(5)
    
    # Risk
    pdf.set_font('helvetica', 'B', 14)
    pdf.cell(0, 10, f'Hallucination Risk: {result.overall_risk:.1f}%', 0, 1)
    pdf.ln(5)
    
    # Statistics
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 8, 'Statistics:', 0, 1)
    pdf.set_font('helvetica', '', 12)
    pdf.cell(0, 7, f'Sentences: {result.metadata["num_sentences"]}', 0, 1)
    pdf.cell(0, 7, f'Avg similarity: {result.metadata["mean_sentence_similarity"]:.2f}', 0, 1)
    pdf.ln(5)
    
    # Fact Check Results
    if fact_results:
        pdf.set_font('helvetica', 'B', 12)
        pdf.cell(0, 8, 'Fact Check Results:', 0, 1)
        pdf.ln(2)
        
        for i, fr in enumerate(fact_results, 1):
            # Статус
            status_symbol = {
                "confirmed": "✓",
                "partial": "~",
                "contradicted": "✗",
                "no_source": "?"
            }.get(fr.status, "•")
            
            pdf.set_font('helvetica', 'B', 11)
            pdf.cell(0, 7, f'{status_symbol} Statement {i}:', 0, 1)
            pdf.set_font('helvetica', '', 10)
            
            # Разбиваем длинные предложения
            sentence = fr.sentence
            while len(sentence) > 80:
                pdf.cell(0, 6, sentence[:80], 0, 1)
                sentence = sentence[80:]
            pdf.cell(0, 6, sentence, 0, 1)
            
            pdf.set_font('helvetica', '', 10)
            pdf.cell(0, 6, f'Status: {fr.status}', 0, 1)
            
            if fr.source_title:
                pdf.set_text_color(0, 0, 255)
                pdf.cell(0, 6, f'Source: {fr.source_title}', 0, 1)
                pdf.set_text_color(0, 0, 0)
            
            pdf.ln(3)
    
    # Get PDF bytes
    pdf_output = pdf.output(dest='S')
    
    # Convert to bytes
    if isinstance(pdf_output, bytes):
        return pdf_output
    elif isinstance(pdf_output, bytearray):
        return bytes(pdf_output)
    elif isinstance(pdf_output, str):
        return pdf_output.encode('latin-1')
    else:
        return bytes(pdf_output)
