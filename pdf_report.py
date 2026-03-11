import io
from typing import List
from datetime import datetime
import os

from fpdf import FPDF
from semantic_analyzer import AnalysisResult, SentenceScore

class PDF(FPDF):
    def __init__(self):
        super().__init__()
        # Используем встроенный шрифт с поддержкой Unicode через DejaVu
        # Если файлы шрифтов есть в папке - они подключатся
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

def build_pdf_bytes(question: str, answer: str, result: AnalysisResult, risk_threshold: float = 60.0) -> bytes:
    """
    Формирует PDF-отчет о проверке с поддержкой кириллицы.
    """
    pdf = PDF()
    pdf.add_page()
    
    # Выбираем шрифт в зависимости от наличия DejaVu
    if pdf.font_loaded:
        main_font = 'DejaVu'
    else:
        main_font = 'helvetica'
    
    # Дата и время
    pdf.set_font(main_font, '', 10)
    pdf.cell(0, 10, f'Дата: {datetime.now().strftime("%d.%m.%Y %H:%M")}', 0, 1)
    pdf.ln(5)
    
    # Вопрос
    pdf.set_font(main_font, 'B', 12)
    pdf.cell(0, 10, 'Вопрос:', 0, 1)
    pdf.set_font(main_font, '', 12)
    
    # Для helvetica нужно кодировать, для DejaVu - нет
    if main_font == 'helvetica':
        pdf.multi_cell(0, 10, question.encode('latin-1', 'ignore').decode('latin-1'))
    else:
        pdf.multi_cell(0, 10, question)
    pdf.ln(5)
    
    # Ответ
    pdf.set_font(main_font, 'B', 12)
    pdf.cell(0, 10, 'Ответ:', 0, 1)
    pdf.set_font(main_font, '', 12)
    
    if main_font == 'helvetica':
        pdf.multi_cell(0, 10, answer.encode('latin-1', 'ignore').decode('latin-1'))
    else:
        pdf.multi_cell(0, 10, answer)
    pdf.ln(5)
    
    # Общий риск
    pdf.set_font(main_font, 'B', 14)
    pdf.cell(0, 10, f'Общий риск галлюцинаций: {result.overall_risk:.1f}%', 0, 1)
    pdf.ln(5)
    
    # Интерпретация риска
    pdf.set_font(main_font, '', 12)
    risk = result.overall_risk
    if risk < 20:
        interpretation = "Низкий риск. Ответ в целом согласован с вопросом."
    elif risk < 50:
        interpretation = "Умеренный риск. Рекомендуется выборочная проверка ключевых фактов."
    elif risk < 80:
        interpretation = "Повышенный риск. Проверьте основные утверждения и цифры."
    else:
        interpretation = "Очень высокий риск. Ответ может содержать существенные неточности."
    
    if main_font == 'helvetica':
        pdf.multi_cell(0, 10, interpretation.encode('latin-1', 'ignore').decode('latin-1'))
    else:
        pdf.multi_cell(0, 10, interpretation)
    pdf.ln(5)
    
    # Статистика
    pdf.set_font(main_font, 'B', 12)
    pdf.cell(0, 10, 'Статистика:', 0, 1)
    pdf.set_font(main_font, '', 12)
    pdf.cell(0, 10, f'Количество предложений: {result.metadata["num_sentences"]}', 0, 1)
    pdf.cell(0, 10, f'Средняя схожесть: {result.metadata["mean_sentence_similarity"]:.2f}', 0, 1)
    pdf.ln(5)
    
    # Рискованные предложения
    risky_sentences = [s for s in result.sentence_scores if s.risk >= risk_threshold]
    
    if risky_sentences:
        pdf.set_font(main_font, 'B', 12)
        pdf.cell(0, 10, 'Предложения с повышенным риском:', 0, 1)
        pdf.ln(3)
        
        for i, s in enumerate(risky_sentences, 1):
            pdf.set_font(main_font, 'B', 11)
            pdf.cell(0, 8, f'{i}. Риск: {s.risk:.1f}% (схожесть: {s.similarity:.2f})', 0, 1)
            pdf.set_font(main_font, '', 11)
            
            if main_font == 'helvetica':
                pdf.multi_cell(0, 8, s.sentence.encode('latin-1', 'ignore').decode('latin-1'))
            else:
                pdf.multi_cell(0, 8, s.sentence)
            pdf.ln(3)
    else:
        pdf.set_font(main_font, '', 12)
        text = "Предложений с повышенным риском не обнаружено."
        if main_font == 'helvetica':
            pdf.cell(0, 10, text.encode('latin-1', 'ignore').decode('latin-1'), 0, 1)
        else:
            pdf.cell(0, 10, text, 0, 1)
    
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
