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
import time

# ========== YOUR DESIGN CSS (PRESERVED 100%) ==========
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
</style>
""", unsafe_allow_html=True)

# ========== SIMPLIFIED SEMANTIC ANALYSIS (NO EXTERNAL semantic_analyzer) ==========
@st.cache_resource
def get_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L6-v2')

@dataclass
class SentenceScore:
    sentence: str
    similarity: float
    risk: float

def analyze_semantic_consistency(question: str, answer: str) -> 'AnalysisResult':
    model = get_model()
    
    # Split answer into sentences
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', answer)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    
    question_emb = model.encode(question)
    sentence_scores = []
    
    for sentence in sentences:
        sent_emb = model.encode(sentence)
        similarity = float(model.similarity(question_emb, sent_emb)[0][0])
        risk = max(0, min(100, (1 - similarity) * 100))
        sentence_scores.append(SentenceScore(sentence, similarity, risk))
    
    # Overall risk
    overall_risk = np.mean([s.risk for s in sentence_scores])
    
    class AnalysisResult:
        def __init__(self, sentence_scores, overall_risk):
            self.sentence_scores = sentence_scores
            self.overall_risk = overall_risk
            self.metadata = {'num_sentences': len(sentence_scores)}
    
    return AnalysisResult(sentence_scores, overall_risk)

# ========== FIXED FACT-CHECKING (RELEVANT SOURCES ONLY) ==========
@dataclass
class FactCheckResult:
    sentence: str
    status: str
    similarity: float
    source_title: Optional[str]
    source_snippet: str
    source_url: Optional[str]
    explanation: str

def fact_check_sentence(sentence: str) -> FactCheckResult:
    """FIXED: Search by sentence KEYWORDS, not random sources"""
    model = get_model()
    
    # Extract key terms for RELEVANT search
    words = re.findall(r'\b\w{4,}\b', sentence.lower())
    numbers = re.findall(r'\d+', sentence)
    query = ' '.join(words[:8] + numbers[:3])[:100]  # Relevant query
    
    # Wikipedia search (relevant pages only)
    candidates = []
    for lang in ["en", "ru"]:
        try:
            # Search Wikipedia for sentence keywords
            api_url = f"https://{lang}.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "utf8": 1,
                "srlimit": 3
            }
            resp = requests.get(api_url, params=params, timeout=5)
            data = resp.json()
            
            for item in data.get("query", {}).get("search", [])[:2]:
                # Get extract from relevant page
                params2 = {
                    "action": "query",
                    "prop": "extracts",
                    "pageids": item["pageid"],
                    "exintro": 1,
                    "explaintext": 1,
                    "format": "json"
                }
                resp2 = requests.get(api_url, params=params2, timeout=5)
                page_data = resp2.json().get("query", {}).get("pages", {}).get(str(item["pageid"]))
                
                if page_data and page_data.get("extract"):
                    snippet = page_data["extract"][:300]
                    candidates.append((item["title"], snippet, f"https://{lang}.wikipedia.org/wiki/{item['title'].replace(' ', '_')}"))
        except:
            continue
    
    if not candidates:
        return FactCheckResult(
            sentence=sentence,
            status="no_source",
            similarity=0.0,
            source_title=None,
            source_snippet="No relevant sources found",
            source_url=None,
            explanation="Manual verification needed"
        )
    
    # Find BEST semantic match
    sentence_emb = model.encode(sentence)
    best_match = max(candidates, key=lambda x: model.similarity(sentence_emb, model.encode(x[1]))[0][0])
    
    title, snippet, url = best_match
    sim_score = float(model.similarity(sentence_emb, model.encode(snippet))[0][0])
    
    # Clear status based on similarity
    if sim_score > 0.75:
        status = "confirmed"
        explanation = "Matches Wikipedia source closely"
    elif sim_score > 0.5:
        status = "partial"
        explanation = "Partially supported by source"
    else:
        status = "contradicted"
        explanation = "Differs significantly from source"
    
    return FactCheckResult(sentence, status, sim_score, title, snippet, url, explanation)

# ========== MAIN APP ==========
def main():
    st.set_page_config(page_title="LLM Hallucination Checker", page_icon="🔍", layout="centered")
    
    # Title (YOUR DESIGN)
    st.markdown('<h1 class="main-title">🔍 LLM Hallucination Checker</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Sentence-by-sentence analysis with source verification</p>', unsafe_allow_html=True)
    
    # How it works card (YOUR DESIGN)
    st.markdown("""
    <div class="card">
        <h3 style="color: white; margin-top: 0;">🎯 Every sentence analyzed</h3>
        <p style="color: rgba(255,255,255,0.9);">
            <strong>1. Semantic risk</strong> — how relevant to your question<br>
            <strong>2. Fact check</strong> — verified against Wikipedia sources
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # Input form
    with st.form(key="qa_form"):
        col1, col2 = st.columns(2)
        with col1:
            question = st.text_area("📝 **Your Question**", height=150, 
                                  placeholder="When was Tesla founded?")
        with col2:
            answer = st.text_area("🤖 **AI Response**", height=150,
                                placeholder="Tesla was founded in 2003...")
        submitted = st.form_submit_button("🚀 Analyze", use_container_width=True)

    if submitted and question.strip() and answer.strip():
        with st.spinner("Analyzing every sentence..."):
            # Semantic analysis
            result = analyze_semantic_consistency(question, answer)
            
            # Fact check ALL sentences
            fact_results = []
            progress_bar = st.progress(0)
            for i, sentence_score in enumerate(result.sentence_scores):
                fact_result = fact_check_sentence(sentence_score.sentence)
                fact_results.append(fact_result)
                progress_bar.progress((i+1) / len(result.sentence_scores))
            progress_bar.empty()

        # Overall risk (BUSINESS PLAN)
        st.markdown("## 📊 Overall Risk")
        col1, col2 = st.columns(2)
        with col1:
            risk_pct = result.overall_risk
            color = "#28a745" if risk_pct < 40 else "#ffc107" if risk_pct < 70 else "#dc3545"
            st.markdown(f"""
            <div class="risk-meter">
                <h3 style="color: {color}">Risk: {risk_pct:.0f}%</h3>
                <div style="background: #f0f0f0; height: 30px; border-radius: 15px;">
                    <div style="background: {color}; width: {risk_pct}%; height: 30px; border-radius: 15px; line-height: 30px; color: white;">
                        {risk_pct:.0f}%
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 📈 Quick Stats")
            risky_sentences = sum(1 for s in result.sentence_scores if s.risk >= 60)
            confirmed = sum(1 for f in fact_results if f.status == "confirmed")
            st.metric("⚠️ Risky Sentences", risky_sentences)
            st.metric("✅ Verified Facts", confirmed)

        # **CORE BUSINESS REQUIREMENT: Every sentence analysis**
        st.markdown('<div class="custom-divider"></div>')
        st.markdown("## 🔍 Sentence-by-Sentence Analysis")
        
        for i, (sentence_score, fact_result) in enumerate(zip(result.sentence_scores, fact_results), 1):
            with st.expander(f"**Sentence #{i}:** {sentence_score.sentence[:80]}..."):
                col1, col2, col3 = st.columns([1,1,2])
                
                with col1:
                    # 1. SEMANTIC RISK
                    risk_color = "🟢" if sentence_score.risk < 40 else "🟡" if sentence_score.risk < 70 else "🔴"
                    st.metric("Semantic Risk", f"{risk_color} {sentence_score.risk:.0f}%")
                
                with col2:
                    # 2. FACT CHECK STATUS
                    badge_class = {
                        "confirmed": "badge-confirmed", "partial": "badge-partial", 
                        "contradicted": "badge-contradicted", "no_source": "badge-no-source"
                    }[fact_result.status]
                    st.markdown(f'<span class="status-badge {badge_class}">{"✅" if fact_result.status=="confirmed" else "🟡" if fact_result.status=="partial" else "❌" if fact_result.status=="contradicted" else "❓"} {fact_result.status.title()}</span>', unsafe_allow_html=True)
                
                with col3:
                    # 3. ONE-SENTENCE ANALYTICS + SOURCE LINK
                    st.markdown(f"**Analytics:** {fact_result.explanation} **({fact_result.similarity:.0f}% match)**")
                    if fact_result.source_url:
                        st.markdown(f"**Source:** [{fact_result.source_title}]({fact_result.source_url})")
                    else:
                        st.markdown(f"**Source:** {fact_result.source_title}")

        # Histogram (YOUR DESIGN)
        st.markdown("## 📊 Response Coherence")
        fig, ax = plt.subplots(figsize=(10, 4))
        sims_pct = [s.risk for s in result.sentence_scores]
        n, bins, patches = ax.hist(sims_pct, bins=8, edgecolor='white')
        for i, patch in enumerate(patches):
            if bins[i] < 40: patch.set_facecolor('#28a745')
            elif bins[i] < 70: patch.set_facecolor('#ffc107')
            else: patch.set_facecolor('#dc3545')
        ax.set_xlabel("Risk Score (%)"); ax.set_ylabel("Sentences")
        st.pyplot(fig)

if __name__ == "__main__":
    main()
