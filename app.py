import io
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from dotenv import load_dotenv

from pdf_report import build_pdf_bytes
from semantic_analyzer import analyze_semantic_consistency, SentenceScore
from fact_checker import fact_check_sentences, FactCheckResult


load_dotenv()

st.set_page_config(
    page_title="LLM Hallucination Risk Checker",
    page_icon="🧠",
    layout="centered",
)


def _compute_histogram_data(sentence_scores: List[SentenceScore]):
    sims = np.array([s.similarity for s in sentence_scores], dtype=float)
    # Переводим в проценты для гистограммы
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
            except Exception:
                st.error("Что-то пошло не так при семантическом анализе. Попробуйте ещё раз позже.")
                return

        # Фактологическая проверка поверх семантически рискованных / фактоёмких предложений
        try:
            with st.spinner("Проверяем факты по открытым источникам..."):
                fact_results: List[FactCheckResult] = fact_check_sentences(result.sentence_scores)
            fact_check_available = True
        except Exception:
            fact_results = []
            fact_check_available = False

        col_score, col_meta = st.columns([1, 1.2])
        with col_score:
            st.subheader("Итоговый риск галлюцинаций")
            st.metric(
                label="Риск (0–100%)",
                value=f"{result.overall_risk:.1f}%",
            )

            # Краткая интерпретация
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

            # Сводка фактологической проверки
            if not fact_check_available:
                st.markdown("**Фактологическая проверка:** временно недоступна. Попробуйте обновить страницу позже.")
            elif fact_results:
                confirmed = sum(1 for fr in fact_results if fr.status == "confirmed")
                partial = sum(1 for fr in fact_results if fr.status == "partial")
                contradicted = sum(1 for fr in fact_results if fr.status == "contradicted")
                no_source = sum(1 for fr in fact_results if fr.status == "no_source")
                total_fc = len(fact_results)

                st.markdown("**Фактологическая проверка (по открытым источникам):**")
                st.write(
                    f"- полностью подтверждены: **{confirmed}** из {total_fc}\n"
                    f"- частично подтверждены: **{partial}**\n"
                    f"- вызывают сомнения / противоречат: **{contradicted}**\n"
                    f"- подходящих источников не найдено: **{no_source}**"
                )
            else:
                st.markdown("**Фактологическая проверка:** подходящих для проверки предложений не найдено.")

        st.markdown("---")
        st.subheader("Гистограмма согласованности по предложениям")
        st.caption(
            "Каждый столбец показывает, сколько предложений из ответа имеют схожесть с вопросом "
            "в определённом диапазоне. Чем правее и выше столбцы, тем больше фраз близки по смыслу к вопросу."
        )

        sims_pct = _compute_histogram_data(result.sentence_scores)

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.hist(sims_pct, bins=8, color="#4C78A8", edgecolor="white")
        ax.set_xlabel("Семантическая схожесть с вопросом, %")
        ax.set_ylabel("Количество предложений")
        ax.set_xlim(0, 100)
        ax.grid(axis="y", alpha=0.2)

        st.pyplot(fig, use_container_width=True)

        # Простейшая аналитика по зонам схожести
        low = np.mean(sims_pct < 40) * 100.0
        mid = np.mean((sims_pct >= 40) & (sims_pct < 70)) * 100.0
        high = np.mean(sims_pct >= 70) * 100.0

        st.markdown(
            f"- **Низкая схожесть (< 40%)**: примерно {low:.1f}% предложений — потенциально рискованные зоны.\n"
            f"- **Средняя схожесть (40–70%)**: примерно {mid:.1f}% предложений — стоит выборочно проверить.\n"
            f"- **Высокая схожесть (> 70%)**: примерно {high:.1f}% предложений — обычно безопасные по смыслу фразы."
        )

        st.markdown("### Предложения с повышенным риском (семантика + факты)")
        st.caption(
            "Семантический риск показывает, насколько фраза уезжает по смыслу от вопроса. "
            "Фактологический статус основан на сравнении с русской Википедией и помогает понять, "
            "подтверждают ли открытые источники это утверждение."
        )

        risk_threshold = 60.0
        risky_sentences = [s for s in result.sentence_scores if s.risk >= risk_threshold]

        # Индексируем факт-чеки по тексту предложения для быстрого доступа
        fc_by_sentence = {fr.sentence: fr for fr in fact_results} if fact_results else {}

        if not risky_sentences:
            st.success("Явных предложений с высоким семантическим риском не обнаружено.")
        else:
            for s in risky_sentences:
                fr = fc_by_sentence.get(s.sentence) if fact_check_available else None
                st.markdown(
                    f"- **Семантический риск {s.risk:.1f}% (sim {s.similarity:.2f})** — {s.sentence}"
                )
                if fr:
                    if fr.status == "confirmed":
                        status_text = "Фактологически: источники в целом подтверждают утверждение."
                    elif fr.status == "partial":
                        status_text = "Фактологически: источники описывают близкий факт, но не дословно — интерпретируйте с осторожностью."
                    elif fr.status == "contradicted":
                        status_text = "Фактологически: источники говорят иначе — проверьте внимательно."
                    else:
                        status_text = "Фактологически: подходящих источников не найдено, нужна ручная проверка."

                    st.markdown(f"  ↳ *{status_text}*")
                    # Структурный разбор чисел/дат
                    if fr.numbers_status != "no_numbers":
                        if fr.numbers_status == "match":
                            num_text = "Числа/даты в ответе совпадают с источником."
                        elif fr.numbers_status == "partial":
                            num_text = (
                                "Часть чисел/дат совпадает с источником, но есть расхождения — проверьте внимательно."
                            )
                        else:
                            num_text = "Числа/даты в ответе не совпадают с источником — высокая вероятность ошибки."

                        st.markdown(f"  ↳ {num_text}")
                        st.markdown(
                            f"    · В ответе: `{', '.join(fr.sentence_numbers)}`  · В источнике: `{', '.join(fr.source_numbers)}`"
                        )

                    if fr.source_title:
                        if fr.source_url:
                            st.markdown(f"  ↳ Источник: **[{fr.source_title}]({fr.source_url})**")
                        else:
                            st.markdown(f"  ↳ Источник: **{fr.source_title}**")

        st.markdown("---")
        st.subheader("Экспорт подробного отчёта")
        st.caption(
            "PDF сохраняет ваш вопрос, ответ, общий риск и список рискованных предложений — удобно прикладывать "
            "к статьям, дипломным работам и редакционным заданиям."
        )

        try:
            pdf_bytes = build_pdf_bytes(question, answer, result, risk_threshold=risk_threshold)
            st.download_button(
                label="Скачать PDF-отчёт",
                data=pdf_bytes,
                file_name="llm_hallucination_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception:
            st.error("Не удалось сформировать PDF-отчёт. Попробуйте ещё раз позже или сократите текст.")


if __name__ == "__main__":
    main()

