import io
from typing import List, Optional
from datetime import datetime
import os
from dataclasses import dataclass

from fpdf import FPDF
from semantic_analyzer import AnalysisResult, SentenceScore

# Определяем класс FactCheckResult прямо здесь
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
        # Используем встроенный шрифт с поддержкой Unicode через DejaVu
        font_dir = os.path.dirname(os.path.abspath(__file__))
        dejavu_regular = os.path.join(font_dir, 'DejaVuSansCondensed.ttf')
        dejavu_bold = os.path.join(font_dir, 'DejaVuSansCondensed-Bold.ttf')
        
        if os.path.exists(dejavu_regular) and os.path.exists(dejavu_bold):
            self.add_font('DejaVu', '', dejavu_regular, uni=True)
            self.add_font('DejaVu', 'B', dejavu_bold, uni=True)
            self.font_loaded = True
        else:
            self.font_loaded = False
        
    def header(self):
        if self.font_loaded:
            self.set_font('DejaVu', 'B', 16)
        else:
            self.set_font('helvetica', 'B', 16)
        self.cell(0, 10, 'Отчет о проверке галлюцинаций LLM', 0, 1, 'C')
        self.ln(10)
    
    def footer(self):
        self.set_y(-15)
        if self.font_loaded:
            self.set_font('DejaVu', '', 8)
        else:
            self.set_font('helvetica', '', 8)
        self.cell(0, 10, f'Страница {self.page_no()}', 0, 0, 'C')
    
    def chapter_title(self, title):
        self.set_font(self.font_loaded and 'DejaVu' or 'helvetica', 'B', 14)
        self.set_fill_color(230, 230, 230)
        self.cell(0, 10, title, 0, 1, 'L', 1)
        self.ln(5)

def build_pdf_bytes(
    question: str, 
    answer: str, 
    result: AnalysisResult, 
    fact_results: List[FactCheckResult],
    risk_threshold: float = 60.0
) -> bytes:
    """
    Формирует PDF-отчет о проверке с полной фактологической информацией.
    """
    pdf = PDF()
    pdf.add_page()
    
    # Выбираем шрифт в зависимости от наличия DejaVu
    main_font = 'DejaVu' if pdf.font_loaded else 'helvetica'
    
    # ===== ЗАГОЛОВОК =====
    pdf.set_font(main_font, 'B', 20)
    pdf.cell(0, 15, 'Анализ галлюцинаций LLM', 0, 1, 'C')
    pdf.set_font(main_font, '', 10)
    pdf.cell(0, 8, f'Сгенерировано: {datetime.now().strftime("%d.%m.%Y %H:%M")}', 0, 1, 'C')
    pdf.ln(10)
    
    # ===== ОСНОВНАЯ ИНФОРМАЦИЯ =====
    pdf.chapter_title('1. Исходные данные')
    pdf.set_font(main_font, 'B', 12)
    pdf.cell(0, 8, 'Вопрос:', 0, 1)
    pdf.set_font(main_font, '', 12)
    pdf.multi_cell(0, 8, question)
    pdf.ln(3)
    
    pdf.set_font(main_font, 'B', 12)
    pdf.cell(0, 8, 'Ответ:', 0, 1)
    pdf.set_font(main_font, '', 12)
    pdf.multi_cell(0, 8, answer)
    pdf.ln(5)
    
    # ===== МЕТРИКИ =====
    pdf.chapter_title('2. Метрики согласованности')
    
    # Общий риск
    pdf.set_font(main_font, 'B', 14)
    risk_color = (255, 100, 100) if result.overall_risk > 60 else (100, 100, 100)
    pdf.set_text_color(*risk_color)
    pdf.cell(0, 10, f'Общий риск: {result.overall_risk:.1f}%', 0, 1)
    pdf.set_text_color(0, 0, 0)
    
    # Интерпретация
    pdf.set_font(main_font, '', 12)
    risk = result.overall_risk
    if risk < 20:
        interpretation = "✅ Низкий риск. Ответ в целом согласован с вопросом."
    elif risk < 50:
        interpretation = "⚠️ Умеренный риск. Рекомендуется выборочная проверка ключевых фактов."
    elif risk < 80:
        interpretation = "⚠️⚠️ Повышенный риск. Проверьте основные утверждения и цифры."
    else:
        interpretation = "❌ Очень высокий риск. Ответ может содержать существенные неточности."
    
    pdf.multi_cell(0, 8, interpretation)
    pdf.ln(3)
    
    # Детальные метрики
    pdf.set_font(main_font, '', 11)
    pdf.cell(0, 7, f"• Схожесть вопрос-ответ: {result.metadata['qa_similarity']:.3f}", 0, 1)
    pdf.cell(0, 7, f"• Средняя схожесть предложений: {result.metadata['mean_sentence_similarity']:.3f}", 0, 1)
    pdf.cell(0, 7, f"• Стандартное отклонение: {result.metadata['std_sentence_similarity']:.3f}", 0, 1)
    pdf.cell(0, 7, f"• Всего предложений: {result.metadata['num_sentences']}", 0, 1)
    pdf.ln(5)
    
    # ===== ФАКТОЛОГИЧЕСКАЯ ПРОВЕРКА =====
    pdf.chapter_title('3. Фактологическая проверка')
    
    # Статистика по фактам
    if fact_results:
        confirmed = sum(1 for fr in fact_results if fr.status == "confirmed")
        partial = sum(1 for fr in fact_results if fr.status == "partial")
        contradicted = sum(1 for fr in fact_results if fr.status == "contradicted")
        no_source = sum(1 for fr in fact_results if fr.status == "no_source")
        total = len(fact_results)
        
        pdf.set_font(main_font, 'B', 12)
        pdf.cell(0, 8, f'Всего проверено предложений: {total}', 0, 1)
        pdf.set_font(main_font, '', 11)
        pdf.cell(0, 7, f'✅ Полностью подтверждены: {confirmed}', 0, 1)
        pdf.cell(0, 7, f'🟡 Частично подтверждены: {partial}', 0, 1)
        pdf.cell(0, 7, f'❌ Противоречат источникам: {contradicted}', 0, 1)
        pdf.cell(0, 7, f'❓ Источники не найдены: {no_source}', 0, 1)
        pdf.ln(5)
        
        # Детальный разбор каждого предложения
        pdf.set_font(main_font, 'B', 12)
        pdf.cell(0, 8, 'Детальный разбор предложений:', 0, 1)
        pdf.ln(2)
        
        for i, fr in enumerate(fact_results, 1):
            # Статус с эмодзи
            status_emoji = {
                "confirmed": "✅",
                "partial": "🟡",
                "contradicted": "❌",
                "no_source": "❓"
            }.get(fr.status, "•")
            
            # Заголовок предложения
            pdf.set_font(main_font, 'B', 11)
            pdf.set_fill_color(240, 240, 240)
            pdf.cell(0, 8, f'{status_emoji} Предложение {i}:', 0, 1, 'L', 1)
            
            # Текст предложения - используем multi_cell для длинных текстов
            pdf.set_font(main_font, '', 10)
            pdf.multi_cell(0, 6, fr.sentence)
            
            # Информация о проверке
            pdf.set_font(main_font, '', 10)
            
            # Статус проверки
            status_text = {
                "confirmed": "Статус: подтверждено",
                "partial": "Статус: частично подтверждено",
                "contradicted": "Статус: противоречит источникам",
                "no_source": "Статус: источники не найдены"
            }.get(fr.status, "Статус: неизвестен")
            pdf.multi_cell(0, 6, status_text)  # Заменили cell на multi_cell
            
            # Семантическая схожесть
            if fr.similarity:
                pdf.multi_cell(0, 6, f"Семантическая схожесть: {fr.similarity:.3f}")  # Заменили cell на multi_cell
            
            # Источник
            if fr.source_title:
                pdf.set_text_color(0, 0, 255)
                pdf.multi_cell(0, 6, f"Источник: {fr.source_title}")  # Заменили cell на multi_cell
                pdf.set_text_color(0, 0, 0)
                if fr.source_url:
                    pdf.set_font(main_font, 'U', 8)
                    # Разбиваем длинный URL на несколько строк
                    url_text = fr.source_url
                    if len(url_text) > 80:
                        # Простой способ разбить URL
                        parts = []
                        for j in range(0, len(url_text), 80):
                            parts.append(url_text[j:j+80])
                        url_text = '\n'.join(parts)
                    pdf.multi_cell(0, 5, url_text)  # Заменили cell на multi_cell
                    pdf.set_font(main_font, '', 10)
            
            # Числа/даты
            if fr.numbers_status != "no_numbers" and (fr.sentence_numbers or fr.source_numbers):
                numbers_text = {
                    "match": "✅ Числа совпадают",
                    "partial": "🟡 Числа совпадают частично",
                    "mismatch": "❌ Числа не совпадают"
                }.get(fr.numbers_status, "")
                
                if numbers_text:
                    pdf.multi_cell(0, 6, numbers_text)  # Заменили cell на multi_cell
                    pdf.set_font(main_font, '', 9)
                    if fr.sentence_numbers:
                        pdf.multi_cell(0, 5, f"  В ответе: {', '.join(fr.sentence_numbers)}")  # Заменили cell на multi_cell
                    if fr.source_numbers:
                        pdf.multi_cell(0, 5, f"  В источнике: {', '.join(fr.source_numbers)}")  # Заменили cell на multi_cell
                    pdf.set_font(main_font, '', 10)
            
            # Объяснение
            if fr.explanation:
                pdf.set_font(main_font, '', 9)
                pdf.multi_cell(0, 5, f"Пояснение: {fr.explanation}")
            
            pdf.ln(5)
            
            # Разделитель
            pdf.set_draw_color(200, 200, 200)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(3)
    else:
        pdf.set_font(main_font, '', 12)
        pdf.multi_cell(0, 8, "Фактологическая проверка не проводилась или не дала результатов.")
        pdf.ln(5)
    
    # ===== РИСКОВАННЫЕ ПРЕДЛОЖЕНИЯ =====
    risky_sentences = [s for s in result.sentence_scores if s.risk >= risk_threshold]
    
    if risky_sentences:
        pdf.chapter_title('4. Предложения с повышенным риском')
        
        for i, s in enumerate(risky_sentences, 1):
            pdf.set_font(main_font, 'B', 11)
            pdf.cell(0, 7, f'{i}. Риск: {s.risk:.1f}% (схожесть: {s.similarity:.2f})', 0, 1)
            pdf.set_font(main_font, '', 11)
            pdf.multi_cell(0, 7, s.sentence)
            pdf.ln(3)
    
    # ===== ИТОГОВЫЕ ВЫВОДЫ =====
    pdf.chapter_title('5. Рекомендации')
    pdf.set_font(main_font, '', 11)
    
    recommendations = []
    if result.overall_risk > 60:
        recommendations.append("• Высокий риск галлюцинаций - требуется тщательная проверка всех фактов")
    if fact_results:
        contradicted = sum(1 for fr in fact_results if fr.status == "contradicted")
        no_source = sum(1 for fr in fact_results if fr.status == "no_source")
        if contradicted > 0:
            recommendations.append(f"• Найдено {contradicted} противоречий с источниками - проверьте эти утверждения")
        if no_source > 0:
            recommendations.append(f"• Для {no_source} утверждений не найдено источников - требуется ручная проверка")
    if result.metadata['std_sentence_similarity'] > 0.2:
        recommendations.append("• Высокий разброс в схожести предложений - ответ стилистически неоднороден")
    
    if not recommendations:
        recommendations.append("• Ответ выглядит согласованным, но рекомендуем выборочно проверить ключевые факты")
    
    for rec in recommendations:
        pdf.multi_cell(0, 7, rec)
    
    # Получаем PDF как байты
    pdf_output = pdf.output(dest='S')
    
    # Конвертируем в bytes в зависимости от типа
    if isinstance(pdf_output, bytes):
        return pdf_output
    elif isinstance(pdf_output, bytearray):
        return bytes(pdf_output)
    elif isinstance(pdf_output, str):
        return pdf_output.encode('latin-1')
    else:
        return bytes(pdf_output)
