import io
from typing import List
from datetime import datetime

from fpdf import FPDF
from semantic_analyzer import AnalysisResult, SentenceScore

class PDF(FPDF):
    def __init__(self):
        super().__init__()
        # Добавляем поддержку Unicode (кириллицы)
        self.add_font('DejaVu', '', 'DejaVuSansCondensed.ttf', uni=True)
        self.add_font('DejaVu', 'B', 'DejaVuSansCondensed-Bold.ttf', uni=True)
        
    def header(self):
        # Заголовок с поддержкой Unicode
        self.set_font('DejaVu', 'B', 16)
        self.cell(0, 10, 'Отчет о проверке галлюцинаций LLM', 0, 1, 'C')
        self.ln(10)
    
    def footer(self):
        self.set_y(-15)
        self.set_font('DejaVu', '', 8)
        self.cell(0, 10, f'Страница {self.page_no()}', 0, 0, 'C')

def build_pdf_bytes(question: str, answer: str, result: AnalysisResult, risk_threshold: float = 60.0) -> bytes:
    """
    Формирует PDF-отчет о проверке.
    """
    pdf = PDF()
    pdf.add_page()
    
    # Используем шрифт с поддержкой Unicode
    pdf.set_font('DejaVu', '', 12)
    
    # Дата и время
    pdf.set_font('DejaVu', '', 10)
    pdf.cell(0, 10, f'Дата: {datetime.now().strftime("%d.%m.%Y %H:%M")}', 0, 1)
    pdf.ln(5)
    
    # Вопрос
    pdf.set_font('DejaVu', 'B', 12)
    pdf.cell(0, 10, 'Вопрос:', 0, 1)
    pdf.set_font('DejaVu', '', 12)
    # Важно: multi_cell для длинного текста
    pdf.multi_cell(0, 10, question)
    pdf.ln(5)
    
    # Ответ
    pdf.set_font('DejaVu', 'B', 12)
    pdf.cell(0, 10, 'Ответ:', 0, 1)
    pdf.set_font('DejaVu', '', 12)
    pdf.multi_cell(0, 10, answer)
    pdf.ln(5)
    
    # Общий риск
    pdf.set_font('DejaVu', 'B', 14)
    pdf.cell(0, 10, f'Общий риск галлюцинаций: {result.overall_risk:.1f}%', 0, 1)
    pdf.ln(5)
    
    # Интерпретация риска
    pdf.set_font('DejaVu', '', 12)
    risk = result.overall_risk
    if risk < 20:
        interpretation = "Низкий риск. Ответ в целом согласован с вопросом."
    elif risk < 50:
        interpretation = "Умеренный риск. Рекомендуется выборочная проверка ключевых фактов."
    elif risk < 80:
        interpretation = "Повышенный риск. Проверьте основные утверждения и цифры."
    else:
        interpretation = "Очень высокий риск. Ответ может содержать существенные неточности."
    
    pdf.multi_cell(0, 10, f'Интерпретация: {interpretation}')
    pdf.ln(5)
    
    # Статистика
    pdf.set_font('DejaVu', 'B', 12)
    pdf.cell(0, 10, 'Статистика:', 0, 1)
    pdf.set_font('DejaVu', '', 12)
    pdf.cell(0, 10, f'Количество предложений: {result.metadata["num_sentences"]}', 0, 1)
    pdf.cell(0, 10, f'Средняя схожесть: {result.metadata["mean_sentence_similarity"]:.2f}', 0, 1)
    pdf.ln(5)
    
    # Рискованные предложения
    risky_sentences = [s for s in result.sentence_scores if s.risk >= risk_threshold]
    
    if risky_sentences:
        pdf.set_font('DejaVu', 'B', 12)
        pdf.cell(0, 10, 'Предложения с повышенным риском:', 0, 1)
        pdf.ln(3)
        
        for i, s in enumerate(risky_sentences, 1):
            pdf.set_font('DejaVu', 'B', 11)
            pdf.cell(0, 8, f'{i}. Риск: {s.risk:.1f}% (схожесть: {s.similarity:.2f})', 0, 1)
            pdf.set_font('DejaVu', '', 11)
            pdf.multi_cell(0, 8, s.sentence)
            pdf.ln(3)
    else:
        pdf.set_font('DejaVu', '', 12)
        pdf.cell(0, 10, 'Предложений с повышенным риском не обнаружено.', 0, 1)
    
    # Возвращаем PDF как байты
    return pdf.output(dest='S').encode('latin-1')
