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

# ПРИНУДИТЕЛЬНО ПЕРЕЗАПИСЫВАЕМ ФАЙЛ С ПРАВИЛЬНЫМ СОДЕРЖИМЫМ
try:
    # Получаем путь к файлу
    import fact_checker_2026
    file_path = fact_checker_2026.__file__
    
    # Читаем текущее содержимое для диагностики
    with open(file_path, 'r', encoding='utf-8') as f:
        current_content = f.read()
    
    # Создаем правильное содержимое файла
    correct_content = '''from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import os
import re
import requests
from sentence_transformers import SentenceTransformer
import numpy as np

from semantic_analyzer import SentenceScore, get_model


WIKIPEDIA_API_URL_TEMPLATE = "https://{lang}.wikipedia.org/w/api.php"
SERPER_URL = "https://google.serper.dev/search"


@dataclass
class FactCheckResult:
    """
    Результат фактологической проверки для одного предложения.

    Это плоская структура, которую удобно отображать в UI и PDF:
    - статус (confirmed / partial / contradicted / no_source);
    - лучшая найденная выдержка из источника;
    - числовые характеристики совпадения (similarity, numbers_status и т.п.).
    """

    sentence: str
    status: str  # "confirmed", "partial", "contradicted", "no_source"
    similarity: Optional[float]
    source_title: Optional[str]
    source_snippet: Optional[str]
    source_url: Optional[str]
    sentence_numbers: List[str]
    source_numbers: List[str]
    numbers_status: str  # "match", "partial", "mismatch", "no_numbers"
    explanation: str


def _looks_fact_dense(sentence: str) -> bool:
    """
    Эвристика: предложение «фактоёмкое», если содержит цифры, годы,
    проценты или последовательность слов с заглавной буквы (имена/организации).
    """
    if re.search(r"\d", sentence):
        return True

    # простая эвристика для имён/сущностей
    tokens = sentence.split()
    caps_runs = 0
    for t in tokens:
        if re.match(r"[A-ZА-ЯЁ][a-zа-яё]+", t):
            caps_runs += 1
            if caps_runs >= 2:
                return True
        else:
            caps_runs = 0

    return False


def _shorten_for_query(sentence: str, max_words: int = 15) -> str:
    words = sentence.split()
    if len(words) <= max_words:
        return sentence
    return " ".join(words[:max_words])


def _extract_numbers(text: str) -> List[str]:
    """
    Выделяет все числа (включая десятичные через точку/запятую) из строки.
    Используется для грубого сравнения фактических значений в ответе и источнике.
    """
    return re.findall(r"\d+(?:[.,]\d+)?", text)


def _wiki_candidates(query: str, top_k: int = 3) -> List[Tuple[str, str, str]]:
    """
    Ищем несколько кандидатов в ру- и эн-Википедии.
    Возвращаем список (lang, title, extract).
    """
    results: List[Tuple[str, str, str]] = []
    for lang in ("ru", "en"):
        api_url = WIKIPEDIA_API_URL_TEMPLATE.format(lang=lang)
        params_search = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "utf8": 1,
        }
        try:
            resp = requests.get(api_url, params=params_search, timeout=5)
            data = resp.json()
        except Exception:
            continue

        search = data.get("query", {}).get("search", [])
        if not search:
            continue

        for item in search[:top_k]:
            pageid = item["pageid"]
            title = item["title"]

            params_extract = {
                "action": "query",
                "prop": "extracts",
                "pageids": pageid,
                "exintro": 1,
                "explaintext": 1,
                "format": "json",
                "utf8": 1,
            }
            try:
                resp2 = requests.get(api_url, params=params_extract, timeout=5)
                data2 = resp2.json()
            except Exception:
                continue

            pages = data2.get("query", {}).get("pages", {})
            page = pages.get(str(pageid))
            if not page:
                continue

            extract = (page.get("extract") or "").strip()
            if not extract:
                continue

            # ограничим выдержку по длине
            snippet = extract.split("\n\n")[0][:600]
            results.append((lang, title, snippet))

    return results


def _web_candidates(query: str, max_results: int = 3) -> List[Tuple[str, str, str]]:
    """
    Ищем сниппеты через Serper.dev (Google Search).
    Требуется переменная окружения SERPER_API_KEY.
    Возвращает список (title, snippet, url).
    """
    api_key = os.environ.get("SERPER_API_KEY")
    if not api_key:
        return []

    try:
        resp = requests.post(
            SERPER_URL,
            headers={
                "X-API-KEY": api_key,
                "Content-Type": "application/json",
            },
            json={
                "q": query,
                "gl": "ru",
                "hl": "ru",
                "num": max_results,
            },
            timeout=8,
        )
        data = resp.json()
    except Exception:
        return []

    results: List[Tuple[str, str, str]] = []
    organic = data.get("organic", []) or data.get("organic_results", [])

    for item in organic[:max_results]:
        title = item.get("title") or ""
        snippet = item.get("snippet") or item.get("snippet_highlighted_words") or ""
        link = item.get("link") or item.get("url") or ""
        if not snippet:
            continue
        if isinstance(snippet, list):
            snippet_text = " ... ".join(snippet)
        else:
            snippet_text = str(snippet)
        results.append((title, snippet_text[:500], link))

    return results


def fact_check_sentences(
    sentences: List[SentenceScore], risk_threshold: float = 60.0
) -> List[FactCheckResult]:
    """
    Фактологическая проверка «подозрительных» предложений по открытым источникам.

    - выбираем предложения, которые либо:
      - имеют риск >= risk_threshold;
      - выглядят фактоёмкими (цифры, даты, имена и т.п.).
    - для каждого формируем короткий запрос и ищем статьи в ru/en.wikipedia.org
      и сниппеты веб-поиска через Serper.dev (если настроен SERPER_API_KEY);
    - сравниваем семантическую близость предложения и всех кандидатов;
    - учитываем совпадение/расхождение чисел и дат;
    - возвращаем статус на уровне предложения (confirmed / partial / contradicted / no_source)
      с коротким текстовым объяснением для пользователя.
    """
    model: SentenceTransformer = get_model()

    candidates: List[SentenceScore] = []
    for s in sentences:
        if s.risk >= risk_threshold or _looks_fact_dense(s.sentence):
            candidates.append(s)

    results: List[FactCheckResult] = []
    if not candidates:
        return results

    for s in candidates:
        query = _shorten_for_query(s.sentence)
        wiki_candidates = _wiki_candidates(query, top_k=3)
        web_candidates = _web_candidates(query, max_results=3)

        if not wiki_candidates and not web_candidates:
            results.append(
                FactCheckResult(
                    sentence=s.sentence,
                    status="no_source",
                    similarity=None,
                    source_title=None,
                    source_snippet=None,
                    source_url=None,
                    sentence_numbers=[],
                    source_numbers=[],
                    numbers_status="no_numbers",
                    explanation="Подходящие статьи в открытых источниках не найдены. Нужна ручная проверка.",
                )
            )
            continue

        # Семантическое сравнение предложения и всех кандидатов
        all_snippets: List[str] = []
        meta: List[Tuple[str, str, Optional[str]]] = []  # (label, title, url)

        for lang, title, snip in wiki_candidates:
            all_snippets.append(snip)
            meta.append((f"wikipedia-{lang}", f"{title} ({lang}.wikipedia)", None))

        for title, snip, url in web_candidates:
            all_snippets.append(snip)
            meta.append(("web", title, url))

        emb = model.encode([s.sentence] + all_snippets, convert_to_numpy=True, normalize_embeddings=True)
        sent_emb = emb[0]
        cand_embs = emb[1:]
        sims = np.dot(cand_embs, sent_emb)
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        label, best_title, best_url = meta[best_idx]
        best_snippet = all_snippets[best_idx]

        # Сравнение чисел/дат, если они есть
        sent_nums_list = _extract_numbers(s.sentence)
        src_nums_list = _extract_numbers(best_snippet)
        sent_nums = set(sent_nums_list)
        src_nums = set(src_nums_list)

        if not sent_nums and not src_nums:
            numbers_status = "no_numbers"
            numbers_conflict = False
        elif sent_nums & src_nums:
            if sent_nums == src_nums:
                numbers_status = "match"
            else:
                numbers_status = "partial"
            numbers_conflict = False
        else:
            numbers_status = "mismatch"
            numbers_conflict = True

        if best_sim >= 0.7 and not numbers_conflict:
            status = "confirmed"
            explanation = "Смысл предложения хорошо совпадает с описанием в Википедии, явных конфликтов по числам нет."
        elif best_sim >= 0.55 and not numbers_conflict:
            status = "partial"
            explanation = "Источники описывают похожий факт, но формулировки отличаются. Интерпретируйте с осторожностью."
        elif best_sim <= 0.35 or numbers_conflict:
            status = "contradicted"
            if numbers_conflict:
                explanation = "Цифры/годы в предложении отличаются от тех, что указаны в Википедии. Вероятна ошибка."
            else:
                explanation = "Описание в Википедии заметно отличается по смыслу. Проверьте факт."
        else:
            status = "no_source"
            explanation = "Источники дают неоднозначное соответствие. Рекомендуется ручная проверка."

        results.append(
            FactCheckResult(
                sentence=s.sentence,
                status=status,
                similarity=best_sim,
                source_title=best_title,
                source_snippet=best_snippet,
                source_url=best_url,
                sentence_numbers=sent_nums_list,
                source_numbers=src_nums_list,
                numbers_status=numbers_status,
                explanation=explanation,
            )
        )

    return results
'''
    
    # Записываем правильное содержимое
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(correct_content)
    
    st.sidebar.success("✅ Файл принудительно перезаписан правильной версией!")
    
    # Проверяем после записи
    with open(file_path, 'r', encoding='utf-8') as f:
        new_content = f.read()
        if 'sentence_numbers' in new_content:
            st.sidebar.success("✅ После перезаписи 'sentence_numbers' найден в файле!")
        else:
            st.sidebar.error("❌ После перезаписи 'sentence_numbers' ВСЁ ЕЩЁ не найден!")
            
except Exception as e:
    st.sidebar.error(f"Ошибка при перезаписи файла: {e}")

# После перезаписи перезагружаем модуль
import importlib
import fact_checker_2026
importlib.reload(fact_checker_2026)
from fact_checker_2026 import fact_check_sentences, FactCheckResult

st.sidebar.header("🔧 Диагностика после перезаписи")

# 1. Проверка пути к файлу
try:
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

# Локально читаем .env, на Streamlit Cloud значения приходят из secrets.
load_dotenv()
# ... остальной код main() ...
    
load_dotenv()

# Если на Streamlit Cloud задан SERPER_API_KEY в st.secrets,
# пробрасываем его в переменные окружения, чтобы fact_checker мог его использовать.
if "SERPER_API_KEY" in getattr(st, "secrets", {}):
    os.environ.setdefault("SERPER_API_KEY", st.secrets["SERPER_API_KEY"])

st.set_page_config(
    page_title="LLM Hallucination Risk Checker",
    page_icon="🧠",
    layout="centered",
)

@st.cache_resource
def load_semantic_model():
    """Кэшируем модель для экономии памяти"""
    from semantic_analyzer import get_model
    return get_model()

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
                # Модель уже загружена через кэш, просто вызываем функцию
                result = analyze_semantic_consistency(question, answer)
            except Exception as e:
                st.error(f"Что-то пошло не так при семантическом анализе: {str(e)}")
                return

        # Фактологическая проверка поверх семантически рискованных / фактоёмких предложений
        try:
            with st.spinner("Проверяем факты по открытым источникам..."):
                fact_results: List[FactCheckResult] = fact_check_sentences(result.sentence_scores)
            fact_check_available = True
        except Exception as e:
            fact_results = []
            fact_check_available = False
            st.warning(f"Фактологическая проверка временно недоступна: {str(e)}")

        # ... остальной код без изменений ...
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
        except Exception as e:
            st.error(f"Не удалось сформировать PDF-отчёт: {str(e)}. Попробуйте ещё раз позже или сократите текст.")


if __name__ == "__main__":
    main()







