import io
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from dotenv import load_dotenv

from pdf_report import build_pdf_bytes
from semantic_analyzer import analyze_semantic_consistency, SentenceScore
from fact_checker_2026 import fact_check_sentences, FactCheckResult
import inspect

# Локально читаем .env, на Streamlit Cloud значения приходят из secrets.
load_dotenv()

# Если на Streamlit Cloud задан SERPER_API_KEY в st.secrets,
# пробрасываем его в переменные окружения.
if "SERPER_API_KEY" in getattr(st, "secrets", {}):
    os.environ.setdefault("SERPER_API_KEY", st.secrets["SERPER_API_KEY"])

st.set_page_config(
    page_title="LLM Hallucination Risk Checker",
    page_icon="🧠",
    layout="centered",
)

# ДИАГНОСТИКА
st.sidebar.header("🔧 Диагностика")

# 1. Проверка пути к файлу
try:
    import fact_checker_2026
    st.sidebar.write("📁 fact_checker_2026.py путь:", fact_checker_2026.__file__)
except Exception as e:
    st.sidebar.error(f"Не удалось импортировать fact_checker_2026: {e}")

# 2. Проверка сигнатуры FactCheckResult
try:
    sig = inspect.signature(FactCheckResult.__init__)
    params = list(sig.parameters.keys())
    st.sidebar.write("📋 Параметры __init__:", params)
    
    required = ['sentence_numbers', 'source_numbers', 'numbers_status']
    missing = [f for f in required if f not in params]
    if missing:
        st.sidebar.error(f"❌ ОТСУТСТВУЮТ: {missing}")
    else:
        st.sidebar.success("✅ Все поля на месте")
except Exception as e:
    st.sidebar.error(f"Ошибка проверки: {e}")

# 3. Проверка содержимого файла
try:
    with open(fact_checker_2026.__file__, 'r', encoding='utf-8') as f:
        content = f.read()[:500]
        if 'sentence_numbers' in content:
            st.sidebar.success("✅ 'sentence_numbers' найден в файле")
        else:
            st.sidebar.error("❌ 'sentence_numbers' НЕ найден в файле")
except Exception as e:
    st.sidebar.error(f"Ошибка чтения файла: {e}")

@st.cache_resource
def load_semantic_model():
    """Кэшируем модель для экономии памяти"""
    from semantic_analyzer import get_model
    return get_model()

def _compute_histogram_data(sentence_scores: List[SentenceScore]):
    sims = np.array([s.similarity for s in sentence_scores], dtype=float)
    sims_pct = sims * 100.0
    return sims_pct

def main():
    st.title("Проверка галлюцинаций LLM")
    st.caption(
        "Быстрая оценка риска галлюцинаций в ответах ChatGPT, Grok и других моделей. "
        "Инструмент для студентов, журналистов и маркетологов."
    )

    with st.form(key="qa_form"):
        question = st.text_area(
            "Вопрос, который вы задали LLM",
            height=120,
            placeholder="Например: Объясни причины кризиса 2008 года простым языком...",
        )
        answer = st.text_area(
            "Ответ LLM (ChatGPT, Grok и т.д.)",
            height=220,
            placeholder="Вставьте сюда полный ответ модели...",
        )

        submitted = st.form_submit_button("Проверить", use_container_width=True)

    if submitted:
        if not question.strip() and not answer.strip():
            st.warning("Введите вопрос и ответ, чтобы запустить проверку.")
            return
        if not question.strip():
            st.warning("Вы забыли ввести вопрос. Пожалуйста, добавьте формулировку запроса к LLM.")
            return
        if not answer.strip():
            st.warning("Вы забыли вставить ответ LLM. Скопируйте полный текст ответа и попробуйте снова.")
            return

        with st.spinner("Выполняется семантический анализ..."):
            try:
                result = analyze_semantic_consistency(question, answer)
            except Exception as e:
                st.error(f"Что-то пошло не так при семантическом анализе: {str(e)}")
                return

        try:
            with st.spinner("Проверяем факты по открытым источникам..."):
                fact_results: List[FactCheckResult] = fact_check_sentences(result.sentence_scores)
            fact_check_available = True
        except Exception as e:
            fact_results = []
            fact_check_available = False
            st.warning(f"Фактологическая проверка временно недоступна: {str(e)}")

        col_score, col_meta = st.columns([1, 1.2])
        with col_score:
            st.subheader("Итоговый риск галлюцинаций")
            st.metric(label="Риск (0–100%)", value=f"{result.overall_risk:.1f}%")

            risk = result.overall_risk
            if risk < 20:
                text = "Низкий риск. Ответ в целом согласован с вопросом."
            elif risk < 50:
                text = "Умеренный риск. Рекомендуется выборочная проверка ключевых фактов."
            elif risk < 80:
                text = "Повышенный риск. Проверьте основные утверждения и цифры."
            else:
                text = "Очень высокий риск. Ответ может содержать существенные неточности или уход от вопроса."
            st.write(text)

        with col_meta:
            st.subheader("Семантическая согласованность")
            st.write(
                f"Схожесть вопрос–ответ (cosine): **{result.metadata['qa_similarity']:.2f}**  "
                f"(0 = нет сходства, 1 = полное совпадение)"
            )
            st.write(
                f"Средняя схожесть по предложениям: **{result.metadata['mean_sentence_similarity']:.2f}**  "
                f"(σ = {result.metadata['std_sentence_similarity']:.2f})"
            )
            st.write(f"Число предложений в ответе: **{result.metadata['num_sentences']}**")

            if not fact_check_available:
                st.markdown("**Фактологическая проверка:** временно недоступна.")
            elif fact_results:
                confirmed = sum(1 for fr in fact_results if fr.status == "confirmed")
                partial = sum(1 for fr in fact_results if fr.status == "partial")
                contradicted = sum(1 for fr in fact_results if fr.status == "contradicted")
                no_source = sum(1 for fr in fact_results if fr.status == "no_source")
                total_fc = len(fact_results)

                st.markdown("**Фактологическая проверка:**")
                st.write(
                    f"- полностью подтверждены: **{confirmed}** из {total_fc}\n"
                    f"- частично подтверждены: **{partial}**\n"
                    f"- вызывают сомнения: **{contradicted}**\n"
                    f"- источников не найдено: **{no_source}**"
                )
            else:
                st.markdown("**Фактологическая проверка:** подходящих предложений не найдено.")

        st.markdown("---")
        st.subheader("Гистограмма согласованности по предложениям")
        sims_pct = _compute_histogram_data(result.sentence_scores)

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.hist(sims_pct, bins=8, color="#4C78A8", edgecolor="white")
        ax.set_xlabel("Семантическая схожесть с вопросом, %")
        ax.set_ylabel("Количество предложений")
        ax.set_xlim(0, 100)
        ax.grid(axis="y", alpha=0.2)
        st.pyplot(fig, use_container_width=True)

        low = np.mean(sims_pct < 40) * 100.0
        mid = np.mean((sims_pct >= 40) & (sims_pct < 70)) * 100.0
        high = np.mean(sims_pct >= 70) * 100.0

        st.markdown(
            f"- **Низкая схожесть (< 40%)**: примерно {low:.1f}% предложений\n"
            f"- **Средняя схожесть (40–70%)**: примерно {mid:.1f}% предложений\n"
            f"- **Высокая схожесть (> 70%)**: примерно {high:.1f}% предложений"
        )

        st.markdown("### Предложения с повышенным риском")
        risk_threshold = 60.0
        risky_sentences = [s for s in result.sentence_scores if s.risk >= risk_threshold]
        fc_by_sentence = {fr.sentence: fr for fr in fact_results} if fact_results else {}

        if not risky_sentences:
            st.success("Явных предложений с высоким семантическим риском не обнаружено.")
        else:
            for s in risky_sentences:
                fr = fc_by_sentence.get(s.sentence) if fact_check_available else None
                st.markdown(f"- **Риск {s.risk:.1f}%** — {s.sentence}")
                if fr:
                    status_text = {
                        "confirmed": "✅ Подтверждено",
                        "partial": "⚠️ Частично подтверждено",
                        "contradicted": "❌ Противоречит источникам",
                        "no_source": "❓ Источники не найдены"
                    }.get(fr.status, "")
                    st.markdown(f"  ↳ *{status_text}*")

        st.markdown("---")
        st.subheader("Экспорт отчёта")
        try:
            pdf_bytes = build_pdf_bytes(question, answer, result, risk_threshold=risk_threshold)
            st.download_button(
                label="Скачать PDF-отчёт",
                data=pdf_bytes,
                file_name="llm_hallucination_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"Не удалось сформировать PDF: {str(e)}")

if __name__ == "__main__":
    main()
