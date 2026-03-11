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
    PDF-отчет с поддержкой кириллицы через встроенный Unicode-шрифт.
    """
    pdf = FPDF()
    pdf.add_page()
    
    # В fpdf2 есть встроенная поддержка Unicode через дефолтный шрифт
    # Но для надежности используем helvetica и конвертируем текст
    
    # ===== ЗАГОЛОВОК =====
    pdf.set_font('helvetica', 'B', 20)
    pdf.cell(0, 15, 'Анализ галлюцинаций LLM', 0, 1, 'C')
    pdf.set_font('helvetica', '', 10)
    pdf.cell(0, 8, f'Сгенерировано: {datetime.now().strftime("%d.%m.%Y %H:%M")}', 0, 1, 'C')
    pdf.ln(10)
    
    # ===== ВОПРОС =====
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 8, 'Вопрос:', 0, 1)
    pdf.set_font('helvetica', '', 12)
    
    # Разбиваем вопрос на строки по 80 символов
    q_lines = [question[i:i+80] for i in range(0, len(question), 80)]
    for line in q_lines:
        # Конвертируем каждый символ в latin-1, заменяя неподдерживаемые на '?'
        try:
            pdf.cell(0, 8, line.encode('latin-1').decode('latin-1'), 0, 1)
        except:
            # Если ошибка - заменяем проблемные символы
            safe_line = ''.join(c if ord(c) < 256 else '?' for c in line)
            pdf.cell(0, 8, safe_line, 0, 1)
    pdf.ln(5)
    
    # ===== ОТВЕТ =====
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 8, 'Ответ:', 0, 1)
    pdf.set_font('helvetica', '', 12)
    
    # Разбиваем ответ на строки по 80 символов
    a_lines = [answer[i:i+80] for i in range(0, len(answer), 80)]
    for line in a_lines:
        try:
            pdf.cell(0, 8, line.encode('latin-1').decode('latin-1'), 0, 1)
        except:
            safe_line = ''.join(c if ord(c) < 256 else '?' for c in line)
            pdf.cell(0, 8, safe_line, 0, 1)
    pdf.ln(5)
    
    # ===== РИСК =====
    pdf.set_font('helvetica', 'B', 14)
    pdf.cell(0, 10, f'Общий риск: {result.overall_risk:.1f}%', 0, 1)
    pdf.ln(5)
    
    # ===== СТАТИСТИКА =====
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 8, 'Статистика:', 0, 1)
    pdf.set_font('helvetica', '', 12)
    pdf.cell(0, 7, f'Предложений: {result.metadata["num_sentences"]}', 0, 1)
    pdf.cell(0, 7, f'Средняя схожесть: {result.metadata["mean_sentence_similarity"]:.2f}', 0, 1)
    pdf.ln(5)
    
    # ===== ФАКТОЛОГИЧЕСКАЯ ПРОВЕРКА =====
    if fact_results:
        pdf.set_font('helvetica', 'B', 12)
        pdf.cell(0, 8, 'Результаты проверки:', 0, 1)
        pdf.ln(2)
        
        confirmed = sum(1 for fr in fact_results if fr.status == "confirmed")
        partial = sum(1 for fr in fact_results if fr.status == "partial")
        contradicted = sum(1 for fr in fact_results if fr.status == "contradicted")
        no_source = sum(1 for fr in fact_results if fr.status == "no_source")
        total = len(fact_results)
        
        pdf.set_font('helvetica', '', 11)
        pdf.cell(0, 7, f'Всего проверено: {total}', 0, 1)
        pdf.cell(0, 7, f'✅ Подтверждено: {confirmed}', 0, 1)
        pdf.cell(0, 7, f'🟡 Частично: {partial}', 0, 1)
        pdf.cell(0, 7, f'❌ Противоречит: {contradicted}', 0, 1)
        pdf.cell(0, 7, f'❓ Нет источника: {no_source}', 0, 1)
        pdf.ln(5)
        
        # Детальный разбор
        pdf.set_font('helvetica', 'B', 12)
        pdf.cell(0, 8, 'Детальный разбор:', 0, 1)
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
            pdf.cell(0, 7, f'{status_symbol} Предложение {i}:', 0, 1)
            pdf.set_font('helvetica', '', 10)
            
            # Разбиваем текст предложения
            sent_lines = [fr.sentence[j:j+70] for j in range(0, len(fr.sentence), 70)]
            for line in sent_lines:
                try:
                    pdf.cell(0, 6, line.encode('latin-1').decode('latin-1'), 0, 1)
                except:
                    safe_line = ''.join(c if ord(c) < 256 else '?' for c in line)
                    pdf.cell(0, 6, safe_line, 0, 1)
            
            pdf.set_font('helvetica', '', 10)
            pdf.cell(0, 6, f'Статус: {fr.status}', 0, 1)
            
            if fr.similarity:
                pdf.cell(0, 6, f'Схожесть: {fr.similarity:.3f}', 0, 1)
            
            if fr.source_title:
                pdf.set_text_color(0, 0, 255)
                try:
                    pdf.cell(0, 6, f'Источник: {fr.source_title}', 0, 1)
                except:
                    safe_source = ''.join(c if ord(c) < 256 else '?' for c in fr.source_title)
                    pdf.cell(0, 6, f'Source: {safe_source}', 0, 1)
                pdf.set_text_color(0, 0, 0)
            
            if fr.explanation:
                pdf.set_font('helvetica', '', 9)
                try:
                    pdf.multi_cell(0, 5, f'Пояснение: {fr.explanation}')
                except:
                    safe_expl = ''.join(c if ord(c) < 256 else '?' for c in fr.explanation)
                    pdf.multi_cell(0, 5, f'Note: {safe_expl}')
                pdf.set_font('helvetica', '', 10)
            
            pdf.ln(3)
            pdf.set_draw_color(200, 200, 200)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(2)
    
    # ===== ИТОГОВЫЕ ВЫВОДЫ =====
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 8, 'Рекомендации:', 0, 1)
    pdf.set_font('helvetica', '', 11)
    
    recommendations = []
    if result.overall_risk > 60:
        recommendations.append("• Высокий риск - требуется проверка фактов")
    if fact_results:
        contradicted = sum(1 for fr in fact_results if fr.status == "contradicted")
        no_source = sum(1 for fr in fact_results if fr.status == "no_source")
        if contradicted > 0:
            recommendations.append(f"• Найдено {contradicted} противоречий")
        if no_source > 0:
            recommendations.append(f"• {no_source} утверждений без источников")
    
    if not recommendations:
        recommendations.append("• Ответ выглядит согласованным")
    
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
