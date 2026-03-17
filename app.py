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
    page_title="LLM Hallucination Checker",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ========== CUSTOM CSS ==========
st.markdown("""
<style>
    /* Main title styling */
    .main-title {
        font-size: 3rem !important;
        font-weight: 800 !important;
        background: linear-gradient(45deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem !important;
    }
    
    /* Subtitle styling */
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem !important;
    }
    
    /* Card styling */
    .card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 1.5rem;
        color: white;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    /* Risk meter styling */
    .risk-meter {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    
    .badge-confirmed {
        background: #d4edda;
        color: #155724;
    }
    
    .badge-partial {
        background: #fff3cd;
        color: #856404;
    }
    
    .badge-contradicted {
        background: #f8d7da;
        color: #721c24;
    }
    
    .badge-no-source {
        background: #e2e3e5;
        color: #383d41;
    }
    
    /* Stat cards */
    .stat-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        border: 1px solid #f0f0f0;
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    
    .stat-label {
        color: #666;
        font-size: 0.9rem;
    }
    
    /* Form styling */
    .stTextArea textarea {
        border-radius: 10px !important;
        border: 2px solid #f0f0f0 !important;
        transition: all 0.3s !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(45deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        transition: all 0.3s !important;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3) !important;
    }
    
    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4) !important;
    }
    
    /* Divider styling */
    .custom-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, #764ba2, #667eea, transparent);
        margin: 2rem 0;
    }
    
    /* Info boxes */
    .info-box {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Metrics styling */
    .metric-container {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #f0f0f0;
    }
</style>
""", unsafe_allow_html=True)

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
    # Custom title
    st.markdown('<h1 class="main-title">🔍 LLM Hallucination Checker</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Advanced AI response validation with semantic analysis and fact-checking</p>', unsafe_allow_html=True)
    
    # Welcome card
    with st.container():
        st.markdown("""
        <div class="card">
            <h3 style="color: white; margin-top: 0;">🎯 How it works</h3>
            <p style="color: rgba(255,255,255,0.9); margin-bottom: 0;">
                Our engine analyzes LLM responses using two complementary methods:<br>
                <strong>1. Semantic coherence</strong> — measures how well each sentence relates to your question<br>
                <strong>2. Factual verification</strong> — cross-references claims with Wikipedia and web sources
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    with st.form(key="qa_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            question = st.text_area(
                "📝 **Your Question**",
                height=150,
                placeholder="e.g., Explain the causes of the 2008 financial crisis...",
                help="Enter the exact question you asked the AI model"
            )
        
        with col2:
            answer = st.text_area(
                "🤖 **AI Response**",
                height=150,
                placeholder="Paste the complete model response here...",
                help="Copy and paste the entire response from ChatGPT, Grok, or any other LLM"
            )

        submitted = st.form_submit_button("🚀 Analyze Response", use_container_width=True)

    if submitted:
        if not question.strip() or not answer.strip():
            st.error("⚠️ Please provide both a question and an answer to analyze.")
            return

        # Progress steps
        progress_bar = st.progress(0, text="Initializing analysis...")
        
        with st.spinner("Performing semantic analysis..."):
            try:
                progress_bar.progress(30, text="Analyzing semantic coherence...")
                result = analyze_semantic_consistency(question, answer)
            except Exception as e:
                st.error(f"❌ Semantic analysis failed: {str(e)}")
                return

        try:
            progress_bar.progress(60, text="Checking facts against sources...")
            with st.spinner("Querying Wikipedia and web sources..."):
                fact_results: List[FactCheckResult] = fact_check_sentences(result.sentence_scores)
            fact_check_available = True
            progress_bar.progress(100, text="Analysis complete!")
            
        except Exception as e:
            fact_results = []
            fact_check_available = False
            st.warning(f"⚠️ Fact check temporarily unavailable: {str(e)}")
        
        progress_bar.empty()

        # Results header
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        st.markdown("## 📊 Analysis Results")
        
        # Risk meter and key metrics
        col_risk, col_stats = st.columns([1, 1])
        
        with col_risk:
            risk = result.overall_risk
            
            # Color-coded risk display
            if risk < 30:
                risk_color = "#28a745"
                risk_emoji = "🟢"
                risk_level = "Low Risk"
            elif risk < 60:
                risk_color = "#ffc107"
                risk_emoji = "🟡"
                risk_level = "Moderate Risk"
            else:
                risk_color = "#dc3545"
                risk_emoji = "🔴"
                risk_level = "High Risk"
            
            st.markdown(f"""
            <div class="risk-meter">
                <h3 style="margin-top: 0; color: {risk_color};">{risk_emoji} {risk_level}</h3>
                <div style="background: #f0f0f0; height: 30px; border-radius: 15px; margin: 10px 0;">
                    <div style="background: {risk_color}; width: {risk}%; height: 30px; border-radius: 15px; text-align: center; line-height: 30px; color: white; font-weight: bold;">
                        {risk:.1f}%
                    </div>
                </div>
                <p style="margin-bottom: 0; color: #666;">
                    { 'Response is generally consistent' if risk < 30 else 
                      'Selective verification recommended' if risk < 60 else 
                      'Critical review needed' }
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_stats:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.markdown("### 📈 Quick Stats")
            
            col_stat1, col_stat2 = st.columns(2)
            with col_stat1:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number">{result.metadata['num_sentences']}</div>
                    <div class="stat-label">Sentences</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number">{result.metadata['qa_similarity']:.2f}</div>
                    <div class="stat-label">QA Similarity</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_stat2:
                confirmed = sum(1 for fr in fact_results if fr.status == "confirmed") if fact_results else 0
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number">{confirmed}</div>
                    <div class="stat-label">Confirmed</div>
                </div>
                """, unsafe_allow_html=True)
                
                contradicted = sum(1 for fr in fact_results if fr.status == "contradicted") if fact_results else 0
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number">{contradicted}</div>
                    <div class="stat-label">Contradictions</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

        # Detailed metrics
        with st.expander("🔬 View Detailed Semantic Metrics", expanded=False):
            col_qa, col_sent = st.columns(2)
            
            with col_qa:
                st.markdown("**Question-Answer Similarity**")
                fig_qa, ax_qa = plt.subplots(figsize=(4, 2))
                ax_qa.barh(['Similarity'], [result.metadata['qa_similarity']], color='#667eea')
                ax_qa.set_xlim(0, 1)
                ax_qa.set_xlabel('Cosine Similarity')
                st.pyplot(fig_qa, use_container_width=True)
            
            with col_sent:
                st.markdown("**Sentence Similarity Distribution**")
                fig_sent, ax_sent = plt.subplots(figsize=(4, 2))
                sims = [s.similarity for s in result.sentence_scores]
                ax_sent.boxplot(sims, vert=False)
                ax_sent.set_xlabel('Similarity')
                st.pyplot(fig_sent, use_container_width=True)

        # Fact check summary with badges
        if fact_check_available and fact_results:
            st.markdown("### ✅ Fact Check Summary")
            
            col_badges, col_stats_detail = st.columns([1, 1])
            
            with col_badges:
                confirmed = sum(1 for fr in fact_results if fr.status == "confirmed")
                partial = sum(1 for fr in fact_results if fr.status == "partial")
                contradicted = sum(1 for fr in fact_results if fr.status == "contradicted")
                no_source = sum(1 for fr in fact_results if fr.status == "no_source")
                total_fc = len(fact_results)
                
                st.markdown(f"""
                <div style="display: flex; flex-wrap: wrap; gap: 10px;">
                    <span class="status-badge badge-confirmed">✅ Confirmed: {confirmed}</span>
                    <span class="status-badge badge-partial">🟡 Partial: {partial}</span>
                    <span class="status-badge badge-contradicted">❌ Contradicted: {contradicted}</span>
                    <span class="status-badge badge-no-source">❓ No source: {no_source}</span>
                </div>
                """, unsafe_allow_html=True)
            
            with col_stats_detail:
                accuracy = (confirmed / total_fc * 100) if total_fc > 0 else 0
                st.metric("Factual Accuracy", f"{accuracy:.1f}%", 
                         delta=f"{confirmed}/{total_fc} confirmed")

        # High-risk sentences
        st.markdown("### ⚠️ High-Risk Sentences")
        
        risk_threshold = 60.0
        risky_sentences = [s for s in result.sentence_scores if s.risk >= risk_threshold]
        fc_by_sentence = {fr.sentence: fr for fr in fact_results} if fact_results else {}

        if not risky_sentences:
            st.success("🎉 No high-risk sentences detected! The response appears to be well-aligned with your question.")
        else:
            st.warning(f"Found {len(risky_sentences)} sentence(s) that may require verification")
            
            for idx, s in enumerate(risky_sentences, 1):
                fr = fc_by_sentence.get(s.sentence) if fact_check_available else None
                
                with st.container():
                    col_marker, col_content = st.columns([0.05, 0.95])
                    
                    with col_marker:
                        st.markdown(f"**{idx}.**")
                    
                    with col_content:
                        st.markdown(f"**{s.sentence}**")
                        
                        # Risk indicator
                        if s.risk < 30:
                            risk_tag = "🟢 Low"
                        elif s.risk < 60:
                            risk_tag = "🟡 Moderate"
                        else:
                            risk_tag = "🔴 High"
                        
                        st.markdown(f"**Risk:** {risk_tag} ({s.risk:.1f}%) | **Similarity:** {s.similarity:.2f}")
                        
                        if fr:
                            if fr.status == "confirmed":
                                badge = '<span class="status-badge badge-confirmed">✅ Confirmed</span>'
                            elif fr.status == "partial":
                                badge = '<span class="status-badge badge-partial">🟡 Partial</span>'
                            elif fr.status == "contradicted":
                                badge = '<span class="status-badge badge-contradicted">❌ Contradicted</span>'
                            else:
                                badge = '<span class="status-badge badge-no-source">❓ No source</span>'
                            
                            st.markdown(f"**Fact check:** {badge} {fr.explanation}", unsafe_allow_html=True)
                            
                            if fr.source_title:
                                if fr.source_url:
                                    st.markdown(f"📚 **Source:** [{fr.source_title}]({fr.source_url})")
                                else:
                                    st.markdown(f"📚 **Source:** {fr.source_title}")
                        
                        st.markdown("---")

        # Histogram
        st.markdown("### 📊 Similarity Distribution")
        sims_pct = _compute_histogram_data(result.sentence_scores)
        
        fig_hist, ax_hist = plt.subplots(figsize=(10, 4))
        n, bins, patches = ax_hist.hist(sims_pct, bins=8, color='#667eea', edgecolor='white', alpha=0.7)
        
        # Color code the bars
        for i, patch in enumerate(patches):
            if bins[i] < 40:
                patch.set_facecolor('#dc3545')
            elif bins[i] < 70:
                patch.set_facecolor('#ffc107')
            else:
                patch.set_facecolor('#28a745')
        
        ax_hist.set_xlabel("Semantic Similarity with Question (%)", fontsize=11)
        ax_hist.set_ylabel("Number of Sentences", fontsize=11)
        ax_hist.set_xlim(0, 100)
        ax_hist.grid(axis="y", alpha=0.2)
        
        st.pyplot(fig_hist, use_container_width=True)
        
        # Quick interpretation
        low_pct = np.mean(sims_pct < 40) * 100
        mid_pct = np.mean((sims_pct >= 40) & (sims_pct < 70)) * 100
        high_pct = np.mean(sims_pct >= 70) * 100
        
        st.markdown(f"""
        <div class="info-box">
            <strong>📈 Distribution Insight:</strong><br>
            • 🔴 <strong>Low similarity (&lt;40%)</strong>: {low_pct:.1f}% — potential hallucinations<br>
            • 🟡 <strong>Medium similarity (40-70%)</strong>: {mid_pct:.1f}% — may need verification<br>
            • 🟢 <strong>High similarity (&gt;70%)</strong>: {high_pct:.1f}% — likely accurate
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
