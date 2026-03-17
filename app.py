import io
import os
import re
import requests
from typing import List, Optional, Tuple
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from semantic_analyzer import analyze_semantic_consistency, SentenceScore, get_model

# ========== НАСТРОЙКА СТРАНИЦЫ ==========
st.set_page_config(
    page_title="LLM Hallucination Risk Checker",
    page_icon="🧠",
    layout="centered",
)

# ========== ЗАГРУЗКА ПЕРЕМЕННЫХ ==========
load_dotenv()

# Если на Streamlit Cloud задан SERPER_API_KEY в st.secrets
if "SERPER_API_KEY" in getattr(st, "secrets", {}):
    os.environ.setdefault("SERPER_API_KEY", st.secrets["SERPER_API_KEY"])

# ========== КОНСТАНТЫ ==========
WIKIPEDIA_API_URL_TEMPLATE = "https://{lang}.wikipedia.org/w/api.php"
SERPER_URL = "https://google.serper.dev/search"

# ========== КЛАССЫ ==========
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

# ========== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ==========
@st.cache_resource
def load_semantic_model():
    """Кэшируем модель для экономии памяти"""
    return get_model()

def _looks_fact_dense(sentence: str) -> bool:
    if re.search(r"\d", sentence):
        return True
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
    return re.findall(r"\d+(?:[.,]\d+)?", text)

def _wiki_candidates(query: str, top_k: int = 3) -> List[Tuple[str, str, str]]:
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

            snippet = extract.split("\n\n")[0][:600]
            results.append((lang, title, snippet))

    return results

def _web_candidates(query: str, max_results: int = 3) -> List[Tuple[str, str, str]]:
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

        all_snippets: List[str] = []
        meta: List[Tuple[str, str, Optional[str]]] = []

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
        best_title, best_url = meta[best_idx][1], meta[best_idx][2]
        best_snippet = all_snippets[best_idx]

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



def _compute_histogram_data(sentence_scores: List[SentenceScore]):
    sims = np.array([s.similarity for s in sentence_scores], dtype=float)
    sims_pct = sims * 100.0
    return sims_pct

# ========== ОСНОВНОЕ ПРИЛОЖЕНИЕ ==========
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
                    f"- ✅ полностью подтверждены: **{confirmed}** из {total_fc}\n"
                    f"- 🟡 частично подтверждены: **{partial}**\n"
                    f"- ❌ противоречат источникам: **{contradicted}**\n"
                    f"- ❓ источников не найдено: **{no_source}**"
                )
            else:
                st.markdown("**Фактологическая проверка:** подходящих предложений не найдено.")

if __name__ == "__main__":
    main()

