from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import os
import re
import requests
import numpy as np

from semantic_analyzer import SentenceScore, get_model


WIKIPEDIA_API_URL_TEMPLATE = "https://{lang}.wikipedia.org/w/api.php"
SERPER_URL = "https://google.serper.dev/search"


@dataclass
class FactCheckResult:
    """
    Результат фактологической проверки для одного предложения.
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
    """Эвристика: предложение «фактоёмкое», если содержит цифры, годы, проценты"""
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
    """Выделяет все числа из строки."""
    return re.findall(r"\d+(?:[.,]\d+)?", text)


def _wiki_candidates(query: str, top_k: int = 3) -> List[Tuple[str, str, str]]:
    """Ищем кандидаты в Википедии."""
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
    """Ищем через Serper.dev."""
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
    Фактологическая проверка предложений.
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
        label, best_title, best_url = meta[best_idx]
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
