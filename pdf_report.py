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
    PDF-отчет с поддержкой Unicode через встроенный шрифт fpdf2.
    """
    pdf = FPDF()
    pdf.add_page()
    
    # В fpdf2 есть встроенная поддержка Unicode через дефолтный шрифт
    # Но для кириллицы нужно использовать правильную кодировку
    pdf.set_font('helvetica', '', 12)
    
    # ===== ЗАГОЛОВОК =====
    pdf.set_font('helvetica', 'B', 20)
    pdf.cell(0, 15, 'LLM Hallucination Report', 0, 1, 'C')
    pdf.set_font('helvetica', '', 10)
    pdf.cell(0, 8, f'Generated: {datetime.now().strftime("%d.%m.%Y %H:%M")}', 0, 1, 'C')
    pdf.ln(10)
    
    # ===== ВОПРОС =====
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 8, 'Question:', 0, 1)
    pdf.set_font('helvetica', '', 12)
    # Для кириллицы используем multi_cell с правильной кодировкой
    pdf.multi_cell(0, 8, question.encode('latin-1', 'ignore').decode('latin-1'))
    pdf.ln(5)
    
    # ===== ОТВЕТ =====
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 8, 'Answer:', 0, 1)
    pdf.set_font('helvetica', '', 12)
    pdf.multi_cell(0, 8, answer.encode('latin-1', 'ignore').decode('latin-1'))
    pdf.ln(5)
    
    # ===== РИСК =====
    pdf.set_font('helvetica', 'B', 14)
    pdf.cell(0, 10, f'Hallucination Risk: {result.overall_risk:.1f}%', 0, 1)
    pdf.ln(5)
    
    # ===== СТАТИСТИКА =====
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 8, 'Statistics:', 0, 1)
    pdf.set_font('helvetica', '', 12)
    pdf.cell(0, 7, f'Sentences: {result.metadata["num_sentences"]}', 0, 1)
    pdf.cell(0, 7, f'Avg similarity: {result.metadata["mean_sentence_similarity"]:.2f}', 0, 1)
    pdf.ln(5)
    
    # ===== ФАКТОЛОГИЧЕСКАЯ ПРОВЕРКА =====
    if fact_results:
        pdf.set_font('helvetica', 'B', 12)
        pdf.cell(0, 8, 'Fact Check Results:', 0, 1)
        pdf.ln(2)
        
        confirmed = sum(1 for fr in fact_results if fr.status == "confirmed")
        partial = sum(1 for fr in fact_results if fr.status == "partial")
        contradicted = sum(1 for fr in fact_results if fr.status == "contradicted")
        no_source = sum(1 for fr in fact_results if fr.status == "no_source")
        total = len(fact_results)
        
        pdf.set_font('helvetica', '', 11)
        pdf.cell(0, 7, f'Total checked: {total}', 0, 1)
        pdf.cell(0, 7, f'✅ Confirmed: {confirmed}', 0, 1)
        pdf.cell(0, 7, f'🟡 Partial: {partial}', 0, 1)
        pdf.cell(0, 7, f'❌ Contradicted: {contradicted}', 0, 1)
        pdf.cell(0, 7, f'❓ No source: {no_source}', 0, 1)
        pdf.ln(5)
        
        # Детальный разбор
        pdf.set_font('helvetica', 'B', 12)
        pdf.cell(0, 8, 'Detailed Analysis:', 0, 1)
        pdf.ln(2)
        
        for i, fr in enumerate(fact_results, 1):
            # Статус
            status_symbol = {
                "confirmed": "✅",
                "partial": "🟡",
                "contradicted": "❌",
                "no_source": "❓"
            }.get(fr.status, "•")
            
            pdf.set_font('helvetica', 'B', 11)
            pdf.cell(0, 7, f'{status_symbol} Statement {i}:', 0, 1)
            pdf.set_font('helvetica', '', 10)
            
            # Текст предложения (очищаем от кириллицы)
            clean_sentence = fr.sentence.encode('latin-1', 'ignore').decode('latin-1')
            pdf.multi_cell(0, 6, clean_sentence)
            
            # Статус
            pdf.set_font('helvetica', '', 10)
            pdf.cell(0, 6, f'Status: {fr.status}', 0, 1)
            
            # Источник
            if fr.source_title:
                clean_source = fr.source_title.encode('latin-1', 'ignore').decode('latin-1')
                pdf.set_text_color(0, 0, 255)
                pdf.cell(0, 6, f'Source: {clean_source}', 0, 1)
                pdf.set_text_color(0, 0, 0)
            
            # Объяснение
            if fr.explanation:
                clean_expl = fr.explanation.encode('latin-1', 'ignore').decode('latin-1')
                pdf.set_font('helvetica', '', 9)
                pdf.multi_cell(0, 5, f'Note: {clean_expl}')
                pdf.set_font('helvetica', '', 10)
            
            pdf.ln(3)
    
    # ===== ИТОГОВЫЕ ВЫВОДЫ =====
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 8, 'Recommendations:', 0, 1)
    pdf.set_font('helvetica', '', 11)
    
    recommendations = []
    if result.overall_risk > 60:
        recommendations.append("• High risk - verify all facts")
    if fact_results:
        contradicted = sum(1 for fr in fact_results if fr.status == "contradicted")
        no_source = sum(1 for fr in fact_results if fr.status == "no_source")
        if contradicted > 0:
            recommendations.append(f"• {contradicted} contradictions found - check these statements")
        if no_source > 0:
            recommendations.append(f"• {no_source} statements with no sources - manual verification needed")
    
    if not recommendations:
        recommendations.append("• Answer appears consistent, but verify key facts")
    
    for rec in recommendations:
        pdf.multi_cell(0, 7, rec)
    
    # Получаем PDF как байты
    pdf_output = pdf.output(dest='S')
    
    # Конвертируем в bytes
    if isinstance(pdf_output, bytes):
        return pdf_output
    elif isinstance(pdf_output, bytearray):
        return bytes(pdf_output)
    elif isinstance(pdf_output, str):
        return pdf_output.encode('latin-1')
    else:
        return bytes(pdf_output)
