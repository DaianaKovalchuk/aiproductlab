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

# ========== PAGE CONFIGURATION ==========
st.set_page_config(
    page_title="LLM Hallucination Risk Checker",
    page_icon="🧠",
    layout="centered",
)

# ========== ENVIRONMENT VARIABLES ==========
load_dotenv()

# If SERPER_API_KEY is set in Streamlit Cloud secrets
if "SERPER_API_KEY" in getattr(st, "secrets", {}):
    os.environ.setdefault("SERPER_API_KEY", st.secrets["SERPER_API_KEY"])

# ========== CONSTANTS ==========
WIKIPEDIA_API_URL_TEMPLATE = "https://{lang}.wikipedia.org/w/api.php"
SERPER_URL = "https://google.serper.dev/search"

# ========== CLASSES ==========
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

# ========== HELPER FUNCTIONS ==========
@st.cache_resource
def load_semantic_model():
    """Cache the model to save memory"""
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
                    explanation="No relevant sources found. Manual verification needed.",
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
            explanation = "The statement closely matches Wikipedia sources with no numerical conflicts."
        elif best_sim >= 0.55 and not numbers_conflict:
            status = "partial"
            explanation = "Sources describe similar facts but wording differs. Interpret with caution."
        elif best_sim <= 0.35 or numbers_conflict:
            status = "contradicted"
            if numbers_conflict:
                explanation = "Numbers/dates in the statement differ from those in Wikipedia sources."
            else:
                explanation = "The description in Wikipedia significantly differs in meaning. Verify this fact."
        else:
            status = "no_source"
            explanation = "Sources provide ambiguous matches. Manual verification recommended."

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

# ========== MAIN APPLICATION ==========
def main():
    st.title("LLM Hallucination Checker")
    st.caption(
        "Quick hallucination risk assessment for ChatGPT, Grok, and other LLM responses. "
        "A tool for students, journalists, and marketers."
    )

    with st.form(key="qa_form"):
        question = st.text_area(
            "Question you asked the LLM",
            height=120,
            placeholder="e.g., Explain the causes of the 2008 financial crisis in simple terms...",
        )
        answer = st.text_area(
            "LLM Response (ChatGPT, Grok, etc.)",
            height=220,
            placeholder="Paste the complete model response here...",
        )

        submitted = st.form_submit_button("Check", use_container_width=True)

    if submitted:
        if not question.strip() and not answer.strip():
            st.warning("Please enter both question and answer to start the check.")
            return
        if not question.strip():
            st.warning("You forgot to enter the question. Please provide your query to the LLM.")
            return
        if not answer.strip():
            st.warning("You forgot to paste the LLM response. Copy the complete answer and try again.")
            return

        with st.spinner("Performing semantic analysis..."):
            try:
                result = analyze_semantic_consistency(question, answer)
            except Exception as e:
                st.error(f"Something went wrong during semantic analysis: {str(e)}")
                return

        try:
            with st.spinner("Checking facts against open sources..."):
                fact_results: List[FactCheckResult] = fact_check_sentences(result.sentence_scores)
            fact_check_available = True
            
        except Exception as e:
            fact_results = []
            fact_check_available = False
            st.warning(f"Fact check temporarily unavailable: {str(e)}")

        col_score, col_meta = st.columns([1, 1.2])
        with col_score:
            st.subheader("Overall Hallucination Risk")
            st.metric(label="Risk (0–100%)", value=f"{result.overall_risk:.1f}%")

            risk = result.overall_risk
            if risk < 20:
                text = "Low risk. Response is generally consistent with the question."
            elif risk < 50:
                text = "Moderate risk. Selective verification of key facts recommended."
            elif risk < 80:
                text = "Elevated risk. Verify main claims and numbers."
            else:
                text = "Very high risk. Response may contain significant inaccuracies or be off-topic."
            st.write(text)

        with col_meta:
            st.subheader("Semantic Coherence")
            st.write(
                f"Question–answer similarity (cosine): **{result.metadata['qa_similarity']:.2f}**  "
                f"(0 = no similarity, 1 = perfect match)"
            )
            st.write(
                f"Average sentence similarity: **{result.metadata['mean_sentence_similarity']:.2f}**  "
                f"(σ = {result.metadata['std_sentence_similarity']:.2f})"
            )
            st.write(f"Number of sentences: **{result.metadata['num_sentences']}**")

            if not fact_check_available:
                st.markdown("**Fact check:** temporarily unavailable.")
            elif fact_results:
                confirmed = sum(1 for fr in fact_results if fr.status == "confirmed")
                partial = sum(1 for fr in fact_results if fr.status == "partial")
                contradicted = sum(1 for fr in fact_results if fr.status == "contradicted")
                no_source = sum(1 for fr in fact_results if fr.status == "no_source")
                total_fc = len(fact_results)

                st.markdown("**Fact check results:**")
                st.write(
                    f"- ✅ fully confirmed: **{confirmed}** out of {total_fc}\n"
                    f"- 🟡 partially confirmed: **{partial}**\n"
                    f"- ❌ contradicts sources: **{contradicted}**\n"
                    f"- ❓ no sources found: **{no_source}**"
                )
            else:
                st.markdown("**Fact check:** no suitable sentences found for verification.")

        st.markdown("---")
        st.subheader("Sentence Coherence Histogram")
        st.caption(
            "Each bar shows how many sentences have similarity to the question in a given range. "
            "The further right and higher the bars, the more phrases are semantically close to the question."
        )

        sims_pct = _compute_histogram_data(result.sentence_scores)

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.hist(sims_pct, bins=8, color="#4C78A8", edgecolor="white")
        ax.set_xlabel("Semantic similarity with question, %")
        ax.set_ylabel("Number of sentences")
        ax.set_xlim(0, 100)
        ax.grid(axis="y", alpha=0.2)

        st.pyplot(fig, use_container_width=True)

        # Simple analytics by similarity zones
        low = np.mean(sims_pct < 40) * 100.0
        mid = np.mean((sims_pct >= 40) & (sims_pct < 70)) * 100.0
        high = np.mean(sims_pct >= 70) * 100.0

        st.markdown(
            f"- **Low similarity (< 40%)**: approximately {low:.1f}% of sentences — potentially risky areas.\n"
            f"- **Medium similarity (40–70%)**: approximately {mid:.1f}% of sentences — selective verification recommended.\n"
            f"- **High similarity (> 70%)**: approximately {high:.1f}% of sentences — generally safe phrases."
        )

        st.markdown("### High-Risk Sentences (Semantic + Factual)")
        st.caption(
            "Semantic risk shows how much a phrase deviates from the question. "
            "Factual status is based on comparison with Wikipedia sources."
        )

        risk_threshold = 60.0
        risky_sentences = [s for s in result.sentence_scores if s.risk >= risk_threshold]

        # Index fact checks by sentence text for quick access
        fc_by_sentence = {fr.sentence: fr for fr in fact_results} if fact_results else {}

        if not risky_sentences:
            st.success("No sentences with high semantic risk detected.")
        else:
            for s in risky_sentences:
                fr = fc_by_sentence.get(s.sentence) if fact_check_available else None
                st.markdown(
                    f"- **Semantic risk {s.risk:.1f}% (sim {s.similarity:.2f})** — {s.sentence}"
                )
                if fr:
                    if fr.status == "confirmed":
                        status_text = "Fact check: sources generally confirm the statement."
                    elif fr.status == "partial":
                        status_text = "Fact check: sources describe similar facts but not verbatim — interpret with caution."
                    elif fr.status == "contradicted":
                        status_text = "Fact check: sources contradict this — verify carefully."
                    else:
                        status_text = "Fact check: no suitable sources found, manual verification needed."

                    st.markdown(f"  ↳ *{status_text}*")
                    # Numbers/dates breakdown
                    if fr.numbers_status != "no_numbers":
                        if fr.numbers_status == "match":
                            num_text = "Numbers/dates match the source."
                        elif fr.numbers_status == "partial":
                            num_text = (
                                "Some numbers/dates match the source, but there are discrepancies — check carefully."
                            )
                        else:
                            num_text = "Numbers/dates don't match the source — high probability of error."

                        st.markdown(f"  ↳ {num_text}")
                        st.markdown(
                            f"    · In response: `{', '.join(fr.sentence_numbers)}`  · In source: `{', '.join(fr.source_numbers)}`"
                        )

                    if fr.source_title:
                        if fr.source_url:
                            st.markdown(f"  ↳ Source: **[{fr.source_title}]({fr.source_url})**")
                        else:
                            st.markdown(f"  ↳ Source: **{fr.source_title}**")

if __name__ == "__main__":
    main()
