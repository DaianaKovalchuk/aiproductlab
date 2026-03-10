from __future__ import annotations

from datetime import datetime
from typing import List
import os

from fpdf import FPDF

from semantic_analyzer import AnalysisResult, SentenceScore


# Используем системный шрифт Windows, который умеет Unicode/кириллицу.
FONT_FAMILY = "ArialUni"
FONT_PATH = r"C:\Windows\Fonts\arial.ttf"


def _ensure_unicode_font(pdf: "PDFReport", style: str = "", size: int = 11):
    """
    fpdf по умолчанию использует шрифты без поддержки Unicode.
    Здесь подключаем системный шрифт Arial (TTF) из Windows,
    который поддерживает русский язык.
    """
    if os.path.exists(FONT_PATH):
        if FONT_FAMILY not in pdf.fonts:
            pdf.add_font(FONT_FAMILY, "", FONT_PATH, uni=True)
            pdf.add_font(FONT_FAMILY, "B", FONT_PATH, uni=True)
            pdf.add_font(FONT_FAMILY, "I", FONT_PATH, uni=True)
        pdf.set_font(FONT_FAMILY, style, size)
    else:
        # Фоллбек: вернёмся к Helvetica (может не поддерживать кириллицу,
        # но хоть не упадём).
        base = "Helvetica"
        pdf.set_font(base, style, size)


class PDFReport(FPDF):
    def header(self):
        _ensure_unicode_font(self, "B", 14)
        self.cell(0, 10, "Отчёт по риску галлюцинаций LLM", ln=True, align="L")
        _ensure_unicode_font(self, "", 9)
        self.cell(0, 6, datetime.now().strftime("%d.%m.%Y %H:%M"), ln=True, align="L")
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        _ensure_unicode_font(self, "I", 8)
        page_text = f"Стр. {self.page_no()}"
        self.cell(0, 10, page_text, 0, 0, "C")


def _multi_cell_text(pdf: PDFReport, label: str, text: str, label_width: int = 35):
    _ensure_unicode_font(pdf, "B", 11)
    pdf.cell(label_width, 6, label, ln=0)
    _ensure_unicode_font(pdf, "", 11)
    pdf.multi_cell(0, 6, text)
    pdf.ln(2)


def _highlight_sentence(pdf: PDFReport, s: SentenceScore, threshold: float):
    """
    Вывод одного предложения. Если риск выше порога, подсвечиваем красным цветом.
    """
    if s.risk >= threshold:
        pdf.set_text_color(200, 0, 0)
    else:
        pdf.set_text_color(0, 0, 0)

    _ensure_unicode_font(pdf, "", 10)
    line = f"[риск {s.risk:.1f}% | sim {s.similarity:.2f}] {s.sentence}"
    pdf.multi_cell(0, 5, line)
    pdf.ln(1)

    # Сбрасываем цвет
    pdf.set_text_color(0, 0, 0)


def build_pdf_bytes(question: str, answer: str, result: AnalysisResult, risk_threshold: float = 60.0) -> bytes:
    """
    Собирает PDF-отчёт и возвращает его как bytes.
    """
    pdf = PDFReport()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Общее резюме
    _multi_cell_text(pdf, "Вопрос:", question)
    _multi_cell_text(pdf, "Ответ LLM:", answer)

    _ensure_unicode_font(pdf, "B", 12)
    pdf.cell(0, 8, f"Итоговый риск галлюцинаций: {result.overall_risk:.1f}%", ln=True)
    pdf.ln(2)

    _ensure_unicode_font(pdf, "", 11)
    pdf.cell(0, 6, f"Схожесть вопрос–ответ (cosine): {result.metadata.get('qa_similarity', 0.0):.2f}", ln=True)
    pdf.cell(
        0,
        6,
        f"Средняя схожесть по предложениям: {result.metadata.get('mean_sentence_similarity', 0.0):.2f} "
        f"(σ={result.metadata.get('std_sentence_similarity', 0.0):.2f})",
        ln=True,
    )
    pdf.cell(0, 6, f"Число предложений в ответе: {result.metadata.get('num_sentences', 0)}", ln=True)
    pdf.ln(4)

    # Объяснение
    _ensure_unicode_font(pdf, "B", 11)
    pdf.cell(0, 6, "Как интерпретировать score:", ln=True)
    _ensure_unicode_font(pdf, "", 10)
    pdf.multi_cell(
        0,
        5,
        (
            "0–20% — низкий риск галлюцинаций, ответ семантически согласован с вопросом.\n"
            "20–50% — умеренный риск, рекомендуется выборочная проверка фактов.\n"
            "50–80% — высокий риск, желательно проверить ключевые утверждения.\n"
            "80–100% — очень высокий риск, ответ вероятно содержит несоответствия или уход в сторону от вопроса."
        ),
    )
    pdf.ln(3)

    # Заголовок для предложений
    _ensure_unicode_font(pdf, "B", 11)
    pdf.cell(0, 6, "Предложения с повышенным риском галлюцинаций:", ln=True)
    pdf.ln(2)

    # Список предложений с подсветкой
    for s in result.sentence_scores:
        _highlight_sentence(pdf, s, risk_threshold)

    # В fpdf2 output(dest="S") уже возвращает bytes/bytearray,
    # поэтому просто приводим к bytes без дополнительного .encode().
    raw = pdf.output(dest="S")
    if isinstance(raw, (bytes, bytearray)):
        return bytes(raw)
    # На всякий случай fallback для старых версий fpdf
    return raw.encode("latin1")

