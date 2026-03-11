import io
from typing import List, Optional
from datetime import datetime
import os
import re
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

class PDF(FPDF):
    def __init__(self):
        super().__init__()
        self.unicode_fonts_loaded = False
        self.load_unicode_fonts()
    
    def load_unicode_fonts(self):
        """Загружает Unicode-шрифты DejaVu из папки проекта"""
        font_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Пути к файлам шрифтов
        regular_path = os.path.join(font_dir, 'DejaVuSansCondensed.ttf')
        bold_path = os.path.join(font_dir, 'DejaVuSansCondensed-Bold.ttf')
        italic_path = os.path.join(font_dir, 'DejaVuSansCondensed-Oblique.ttf')
        
        # Проверяем наличие файлов
        if os.path.exists(regular_path):
            self.add_font('DejaVu', '', regular_path, uni=True)
            if os.path.exists(bold_path):
                self.add_font('DejaVu', 'B', bold_path, uni=True)
            if os.path.exists(italic_path):
                self.add_font('DejaVu', 'I', italic_path, uni=True)
            self.unicode_fonts_loaded = True
            print("✅ Unicode fonts loaded successfully")
        else:
            print("⚠️ Unicode fonts not found, using Helvetica")
    
    def select_font(self, text: str = "", style: str = '', size: int = 10):
        """Выбирает шрифт в зависимости от текста"""
        if self.unicode_fonts_loaded:
            self.set_font('DejaVu', style, size)
        else:
            self.set_font('helvetica', style, size)

def build_pdf_bytes(
    question: str, 
    answer: str, 
    result: AnalysisResult, 
    fact_results: List[FactCheckResult],
    risk_threshold: float = 60.0
) -> bytes:
    """
    PDF-отчет с поддержкой кириллицы через DejaVu шрифты.
    """
    pdf = PDF()
    pdf.add_page()
    
    # Проверяем, загрузились ли шрифты
    if not pdf.unicode_fonts_loaded:
        print("⚠️ Using Helvetica - Cyrillic text may not display correctly")
    
    # ===== ЗАГОЛОВОК =====
    pdf.select_font(style='B', size=20)
    pdf.cell(0, 15, 'Анализ галлюцинаций LLM', 0, 1, 'C')
    pdf.select_font(size=10)
    pdf.cell(0, 8, f'Сгенерировано: {datetime.now().strftime("%d.%m.%Y %H:%M")}', 0, 1, 'C')
    pdf.ln(10)
    
    # ===== ВОПРОС =====
    pdf.select_font(style='B', size=12)
    pdf.cell(0, 8, 'Вопрос:', 0, 1)
    pdf.select_font(size=12)
    pdf.multi_cell(0, 8, question)
    pdf.ln(5)
    
    # ===== ОТВЕТ =====
    pdf.select_font(style='B', size=12)
    pdf.cell(0, 8, 'Ответ:', 0, 1)
    pdf.select_font(size=12)
    pdf.multi_cell(0, 8, answer)
    pdf.ln(5)
    
    # ===== РИСК =====
    pdf.select_font(style='B', size=14)
    pdf.cell(0, 10, f'Общий риск: {result.overall_risk:.1f}%', 0, 1)
    pdf.ln(5)
    
    # ===== СТАТИСТИКА =====
    pdf.select_font(style='B', size=12)
    pdf.cell(0, 8, 'Статистика:', 0, 1)
    pdf.select_font(size=12)
    pdf.cell(0, 7, f'Предложений: {result.metadata["num_sentences"]}', 0, 1)
    pdf.cell(0, 7, f'Средняя схожесть: {result.metadata["mean_sentence_similarity"]:.2f}', 0, 1)
    pdf.ln(5)
    
    # ===== ФАКТОЛОГИЧЕСКАЯ ПРОВЕРКА =====
    if fact_results:
        pdf.select_font(style='B', size=12)
        pdf.cell(0, 8, 'Результаты проверки:', 0, 1)
        pdf.ln(2)
        
        confirmed = sum(1 for fr in fact_results if fr.status == "confirmed")
        partial = sum(1 for fr in fact_results if fr.status == "partial")
        contradicted = sum(1 for fr in fact_results if fr.status == "contradicted")
        no_source = sum(1 for fr in fact_results if fr.status == "no_source")
        total = len(fact_results)
        
        pdf.select_font(size=11)
        pdf.cell(0, 7, f'Всего проверено: {total}', 0, 1)
        pdf.cell(0, 7, f'✅ Подтверждено: {confirmed}', 0, 1)
        pdf.cell(0, 7, f'🟡 Частично: {partial}', 0, 1)
        pdf.cell(0, 7, f'❌ Противоречит: {contradicted}', 0, 1)
        pdf.cell(0, 7, f'❓ Нет источника: {no_source}', 0, 1)
        pdf.ln(5)
        
        # Детальный разбор
        pdf.select_font(style='B', size=12)
        pdf.cell(0, 8, 'Детальный разбор:', 0, 1)
        pdf.ln(2)
        
        for i, fr in enumerate(fact_results, 1):
            # Статус с эмодзи
            status_symbol = {
                "confirmed": "✅",
                "partial": "🟡",
                "contradicted": "❌",
                "no_source": "❓"
            }.get(fr.status, "•")
            
            pdf.select_font(style='B', size=11)
            pdf.cell(0, 7, f'{status_symbol} Предложение {i}:', 0, 1)
            pdf.select_font(size=10)
            
            # Текст предложения с переносом строк
            pdf.multi_cell(0, 6, fr.sentence)
            
            pdf.select_font(size=10)
            pdf.cell(0, 6, f'Статус: {fr.status}', 0, 1)
            
            if fr.similarity:
                pdf.cell(0, 6, f'Схожесть: {fr.similarity:.3f}', 0, 1)
            
            if fr.source_title:
                pdf.set_text_color(0, 0, 255)
                pdf.cell(0, 6, f'Источник: {fr.source_title}', 0, 1)
                pdf.set_text_color(0, 0, 0)
            
            if fr.source_url:
                pdf.set_font('helvetica' if not pdf.unicode_fonts_loaded else 'DejaVu', 'U', 8)
                # Разбиваем длинный URL
                url = fr.source_url
                if len(url) > 70:
                    parts = [url[i:i+70] for i in range(0, len(url), 70)]
                    for part in parts:
                        pdf.cell(0, 5, part, 0, 1)
                else:
                    pdf.cell(0, 5, url, 0, 1)
                pdf.select_font(size=10)
            
            if fr.explanation:
                pdf.select_font(style='I', size=9)
                pdf.multi_cell(0, 5, f'Пояснение: {fr.explanation}')
                pdf.select_font(size=10)
            
            pdf.ln(3)
            
            # Разделитель
            pdf.set_draw_color(200, 200, 200)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(2)
    
    # ===== ИТОГОВЫЕ ВЫВОДЫ =====
    pdf.select_font(style='B', size=12)
    pdf.cell(0, 8, 'Рекомендации:', 0, 1)
    pdf.select_font(size=11)
    
    recommendations = []
    if result.overall_risk > 60:
        recommendations.append("• Высокий риск галлюцинаций - требуется тщательная проверка")
    if fact_results:
        contradicted = sum(1 for fr in fact_results if fr.status == "contradicted")
        no_source = sum(1 for fr in fact_results if fr.status == "no_source")
        if contradicted > 0:
            recommendations.append(f"• Найдено {contradicted} противоречий с источниками")
        if no_source > 0:
            recommendations.append(f"• Для {no_source} утверждений не найдено источников")
    
    if not recommendations:
        recommendations.append("• Ответ выглядит согласованным, рекомендуем выборочно проверить ключевые факты")
    
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
