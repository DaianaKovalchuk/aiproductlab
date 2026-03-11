import io
from typing import List
from datetime import datetime
import os

from fpdf import FPDF
from semantic_analyzer import AnalysisResult, SentenceScore

def build_pdf_bytes(question: str, answer: str, result: AnalysisResult, risk_threshold: float = 60.0) -> bytes:
    """
    Формирует PDF-отчет о проверке.
    """
    pdf = FPDF()
    pdf.add_page()
    
    # Используем встроенные шрифты
    pdf.set_font('helvetica', '', 12)
    
    # Дата и время на английском (чтобы избежать проблем с кириллицей)
    pdf.set_font('helvetica', '', 10)
    pdf.cell(0, 10, f'Date: {datetime.now().strftime("%d.%m.%Y %H:%M")}', 0, 1)
    pdf.ln(5)
    
    # Вопрос
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 10, 'Question:', 0, 1)
    pdf.set_font('helvetica', '', 12)
    # Очищаем от кириллицы для совместимости
    clean_question = question.encode('ascii', 'ignore').decode('ascii')
    pdf.multi_cell(0, 10, clean_question)
    pdf.ln(5)
    
    # Ответ
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 10, 'Answer:', 0, 1)
    pdf.set_font('helvetica', '', 12)
    clean_answer = answer.encode('ascii', 'ignore').decode('ascii')
    pdf.multi_cell(0, 10, clean_answer)
    pdf.ln(5)
    
    # Общий риск
    pdf.set_font('helvetica', 'B', 14)
    pdf.cell(0, 10, f'Hallucination Risk: {result.overall_risk:.1f}%', 0, 1)
    pdf.ln(5)
    
    # Статистика
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 10, 'Statistics:', 0, 1)
    pdf.set_font('helvetica', '', 12)
    pdf.cell(0, 10, f'Sentences: {result.metadata["num_sentences"]}', 0, 1)
    pdf.cell(0, 10, f'Avg similarity: {result.metadata["mean_sentence_similarity"]:.2f}', 0, 1)
    pdf.ln(5)
    
    # Получаем PDF как строку и конвертируем в bytes
    pdf_string = pdf.output(dest='S')
    
    # ИСПРАВЛЕНО: правильное преобразование в bytes
    if isinstance(pdf_string, str):
        return pdf_string.encode('latin-1')
    elif isinstance(pdf_string, bytes):
        return pdf_string
    elif isinstance(pdf_string, bytearray):
        return bytes(pdf_string)
    else:
        return bytes(pdf_string)
