import io
import os
import re
import requests
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
import json

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
    
    .badge-low {
        background: #d4edda;
        color: #155724;
    }
    
    .badge-medium {
        background: #fff3cd;
        color: #856404;
    }
    
    .badge-high {
        background: #f8d7da;
        color: #721c24;
    }
    
    .badge-confirmed {
        background: #d4edda;
        color: #155724;
    }
    
    .badge-questionable {
        background: #fff3cd;
        color: #856404;
    }
    
    .badge-debunked {
        background: #f8d7da;
        color: #721c24;
    }
    
    /* Sentence card */
    .sentence-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .sentence-text {
        font-size: 1.1rem;
        line-height: 1.5;
        color: #333;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #f0f0f0;
    }
    
    .metric-row {
        display: flex;
        gap: 1rem;
        margin: 0.5rem 0;
        flex-wrap: wrap;
    }
    
    .metric-item {
        flex: 1;
        min-width: 120px;
        padding: 0.5rem;
        border-radius: 8px;
        background: #f8f9fa;
    }
    
    .metric-label {
        font-size: 0.8rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-value {
        font-size: 1.2rem;
        font-weight: 700;
        color: #667eea;
    }
    
    .analysis-text {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1rem;
        border-left: 3px solid #667eea;
        font-style: italic;
        color: #555;
    }
    
    .fact-check-box {
        background: #e8f4fd;
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1rem;
        border-left: 3px solid #2196F3;
    }
    
    .source-link {
        color: #2196F3;
        text-decoration: none;
        font-weight: 500;
    }
    
    .source-link:hover {
        text-decoration: underline;
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
</style>
""", unsafe_allow_html=True)

# ========== ENVIRONMENT VARIABLES ==========
load_dotenv()

# ========== CONSTANTS ==========
WIKIPEDIA_API_URL = "https://ru.wikipedia.org/w/api.php"
WIKIDATA_API_URL = "https://www.wikidata.org/w/api.php"

# ========== CLASSES ==========
@dataclass
class FactCheckResult:
    sentence: str
    has_factual_content: bool
    extracted_entities: List[str]
    extracted_dates: List[str]
    wiki_match: Optional[str]
    wiki_snippet: Optional[str]
    wiki_url: Optional[str]
    verification_status: str  # "confirmed", "questionable", "debunked", "no_data"
    confidence: float

# ========== HELPER FUNCTIONS ==========
@st.cache_resource
def load_semantic_model():
    """Cache the model to save memory"""
    return get_model()

def extract_entities_and_dates(text: str) -> Tuple[List[str], List[str]]:
    """Extract potential entities (capitalized words) and dates from text"""
    # Extract dates (years, full dates)
    date_patterns = [
        r'\b\d{4}\s*г\.?\b',  # 2024 г.
        r'\b\d{1,2}\s+[а-яА-Я]+\s+\d{4}\b',  # 15 мая 2024
        r'\b\d{1,2}\.\d{1,2}\.\d{4}\b',  # 15.05.2024
        r'\b\d{4}\s+год\w*\b',  # 2024 год
    ]
    
    dates = []
    for pattern in date_patterns:
        dates.extend(re.findall(pattern, text, re.IGNORECASE))
    
    # Extract potential entities (capitalized words/phrases)
    # Look for sequences of capitalized words (potential names, places, events)
    entities = re.findall(r'\b[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)*\b', text)
    
    # Filter out common words that might be capitalized at start of sentence
    stop_words = {'Это', 'Он', 'Она', 'Они', 'Мы', 'Вы', 'Таким', 'Также', 'Кроме'}
    entities = [e for e in entities if e not in stop_words and len(e) > 2]
    
    return entities, dates

def search_wikipedia(query: str, lang: str = "ru") -> Optional[Dict[str, Any]]:
    """Search Wikipedia for a given query with better error handling"""
    try:
        # Search for pages
        search_params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "utf8": 1,
            "srlimit": 3
        }
        
        # Добавляем User-Agent (требуется Wikipedia)
        headers = {
            'User-Agent': 'LLMHallucinationChecker/1.0 (https://your-app-url.com)'
        }
        
        response = requests.get(
            f"https://{lang}.wikipedia.org/w/api.php",
            params=search_params,
            headers=headers,
            timeout=5
        )
        
        # Проверяем, что ответ успешный
        if response.status_code != 200:
            st.warning(f"Wikipedia returned status code {response.status_code} for query '{query}'")
            return None
            
        # Проверяем, что ответ - валидный JSON
        try:
            data = response.json()
        except ValueError as e:
            st.warning(f"Invalid JSON response from Wikipedia for query '{query}': {str(e)}")
            return None
        
        search_results = data.get("query", {}).get("search", [])
        if not search_results:
            return None
        
        # Get the first result
        first_result = search_results[0]
        page_title = first_result["title"]
        page_id = first_result["pageid"]
        
        # Get page extract
        extract_params = {
            "action": "query",
            "prop": "extracts",
            "exintro": 1,
            "explaintext": 1,
            "pageids": page_id,
            "format": "json",
            "utf8": 1
        }
        
        extract_response = requests.get(
            f"https://{lang}.wikipedia.org/w/api.php",
            params=extract_params,
            headers=headers,
            timeout=5
        )
        
        if extract_response.status_code != 200:
            return None
            
        try:
            extract_data = extract_response.json()
        except ValueError:
            return None
        
        pages = extract_data.get("query", {}).get("pages", {})
        page_data = pages.get(str(page_id), {})
        extract = page_data.get("extract", "")
        
        if extract:
            # Take first 500 chars as snippet
            snippet = extract[:500] + "..." if len(extract) > 500 else extract
            
            return {
                "title": page_title,
                "snippet": snippet,
                "url": f"https://{lang}.wikipedia.org/wiki/{page_title.replace(' ', '_')}",
                "pageid": page_id
            }
    
    except requests.exceptions.Timeout:
        st.warning(f"Wikipedia timeout for query '{query}'")
    except requests.exceptions.ConnectionError:
        st.warning(f"Wikipedia connection error for query '{query}'")
    except Exception as e:
        st.warning(f"Wikipedia search error for '{query}': {str(e)}")
    
    return None

def search_wikidata_by_date(date: str) -> Optional[Dict[str, Any]]:
    """Search Wikidata for events on a specific date"""
    try:
        # Clean up the date string
        date_clean = re.sub(r'\s*г\.?\s*$', '', date).strip()
        
        # Try to find year
        year_match = re.search(r'\b(\d{4})\b', date_clean)
        if year_match:
            year = year_match.group(1)
            
            # Search for events in that year
            search_params = {
                "action": "wbsearchentities",
                "search": f"events in {year}",
                "language": "ru",
                "format": "json",
                "limit": 3
            }
            
            response = requests.get(WIKIDATA_API_URL, params=search_params, timeout=5)
            data = response.json()
            
            if data.get("search"):
                # Look for articles in Russian Wikipedia
                for item in data["search"]:
                    if "wikipedia" in item.get("url", ""):
                        return {
                            "title": item.get("label", f"События {year} года"),
                            "snippet": f"Информация о событиях {year} года",
                            "url": item.get("url", ""),
                            "source": "wikidata"
                        }
    
    except Exception as e:
        st.warning(f"Wikidata search error for '{date}': {str(e)}")
    
    return None

def verify_sentence_facts(sentence: str) -> FactCheckResult:
    """Verify facts in a sentence using Wikipedia and Wikidata"""
    
    # Extract entities and dates
    entities, dates = extract_entities_and_dates(sentence)
    
    # Не проверяем слишком короткие предложения
    if len(sentence) < 10:
        return FactCheckResult(
            sentence=sentence,
            has_factual_content=False,
            extracted_entities=entities,
            extracted_dates=dates,
            wiki_match=None,
            wiki_snippet=None,
            wiki_url=None,
            verification_status="no_data",
            confidence=0.0
        )
    
    # Combine all potential search terms
    search_queries = []
    
    # Add full sentence for context (but shortened)
    words = sentence.split()
    if len(words) > 8:  # Уменьшили с 10 до 8
        search_queries.append(" ".join(words[:8]))
    else:
        search_queries.append(sentence)
    
    # Add entities
    search_queries.extend(entities[:2])  # Уменьшили с 3 до 2
    
    # Add dates with context
    for date in dates[:1]:  # Уменьшили с 2 до 1
        # Try to find a relevant entity to pair with date
        if entities:
            search_queries.append(f"{entities[0]} {date}")
        else:
            search_queries.append(date)
    
    # Try each query until we find a match
    best_match = None
    best_confidence = 0.0
    verification_status = "no_data"
    
    # Ограничиваем количество попыток
    for query in search_queries[:3]:  # Только первые 3 запроса
        if not query or len(query) < 3:
            continue
        
        # Не отправляем слишком длинные запросы
        if len(query) > 100:
            query = query[:100]
        
        # Try Russian Wikipedia first
        result = search_wikipedia(query, "ru")
        if result:
            # Simple confidence scoring based on query relevance
            confidence = 0.7  # Base confidence
            
            # Boost confidence if query contains both entity and date
            if any(d in query for d in dates) and any(e in query for e in entities):
                confidence = 0.9
            
            best_match = result
            best_confidence = confidence
            
            # Check if the sentence content appears in the snippet
            sentence_keywords = set(re.findall(r'\b\w{4,}\b', sentence.lower()))
            snippet_keywords = set(re.findall(r'\b\w{4,}\b', result["snippet"].lower()))
            
            common_keywords = sentence_keywords & snippet_keywords
            if len(common_keywords) >= 3:
                verification_status = "confirmed"
            else:
                verification_status = "questionable"
            
            break
        
        # If no Russian result, try English
        result = search_wikipedia(query, "en")
        if result:
            best_match = result
            best_confidence = 0.6
            verification_status = "questionable"
            break
        
        # Небольшая задержка между запросами
        import time
        time.sleep(0.5)
    
    return FactCheckResult(
        sentence=sentence,
        has_factual_content=len(entities) > 0 or len(dates) > 0,
        extracted_entities=entities,
        extracted_dates=dates,
        wiki_match=best_match["title"] if best_match else None,
        wiki_snippet=best_match["snippet"] if best_match else None,
        wiki_url=best_match["url"] if best_match else None,
        verification_status=verification_status,
        confidence=best_confidence
    )

def get_risk_level(risk_score: float) -> Tuple[str, str, str]:
    """Return risk level, color, and emoji based on risk score"""
    if risk_score < 30:
        return "Low", "badge-low", "🟢"
    elif risk_score < 60:
        return "Medium", "badge-medium", "🟡"
    else:
        return "High", "badge-high", "🔴"

def get_fact_status_badge(status: str) -> Tuple[str, str]:
    """Return badge class and emoji for fact status"""
    badges = {
        "confirmed": ("badge-confirmed", "✅"),
        "questionable": ("badge-questionable", "⚠️"),
        "debunked": ("badge-debunked", "❌"),
        "no_data": ("badge-medium", "❓")
    }
    return badges.get(status, ("badge-medium", "❓"))

def generate_analysis(sentence: str, similarity: float, risk_score: float, 
                     fact_result: Optional[FactCheckResult] = None) -> str:
    """Generate a one-sentence analysis based on semantic metrics and fact check"""
    
    semantic_part = ""
    if risk_score < 30:
        semantic_part = f"✅ Strongly aligned with question (similarity: {similarity:.2f})"
    elif risk_score < 60:
        semantic_part = f"⚡ Moderately relevant (similarity: {similarity:.2f})"
    else:
        semantic_part = f"⚠️ Low relevance to question (similarity: {similarity:.2f})"
    
    if fact_result and fact_result.has_factual_content:
        if fact_result.verification_status == "confirmed":
            return f"{semantic_part} | ✅ Fact-check: Confirmed in Wikipedia sources"
        elif fact_result.verification_status == "questionable":
            return f"{semantic_part} | ⚠️ Fact-check: Partially matches sources - verify details"
        elif fact_result.verification_status == "no_data":
            return f"{semantic_part} | ❓ Fact-check: No direct sources found for verification"
    
    return semantic_part

def _compute_histogram_data(sentence_scores: List[SentenceScore]):
    sims = np.array([s.similarity for s in sentence_scores], dtype=float)
    sims_pct = sims * 100.0
    return sims_pct

# ========== MAIN APPLICATION ==========
def main():
    # Custom title
    st.markdown('<h1 class="main-title">🔍 LLM Hallucination Checker</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Advanced AI response validation with semantic analysis and smart fact-checking</p>', unsafe_allow_html=True)
    
    # Welcome card
    with st.container():
        st.markdown("""
        <div class="card">
            <h3 style="color: white; margin-top: 0;">🎯 How it works</h3>
            <p style="color: rgba(255,255,255,0.9); margin-bottom: 0;">
                Our engine analyzes LLM responses using two complementary methods:<br>
                <strong>1. Semantic analysis</strong> — measures relevance to your question<br>
                <strong>2. Smart fact-checking</strong> — automatically detects entities (people, places, events) and dates, then verifies them against Wikipedia
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<hr style="margin: 2rem 0; opacity: 0.2;">', unsafe_allow_html=True)

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

        # Progress indicator
        progress_bar = st.progress(0, text="Initializing analysis...")
        
        with st.spinner("Performing semantic analysis..."):
            try:
                progress_bar.progress(30, text="Analyzing semantic coherence...")
                result = analyze_semantic_consistency(question, answer)
            except Exception as e:
                st.error(f"❌ Semantic analysis failed: {str(e)}")
                return
        
        # Fact-check sentences with factual content
        progress_bar.progress(60, text="Fact-checking claims against Wikipedia...")
        fact_results = {}
        
        for sentence_score in result.sentence_scores:
            fact_result = verify_sentence_facts(sentence_score.sentence)
            if fact_result.has_factual_content:
                fact_results[sentence_score.sentence] = fact_result
        
        progress_bar.progress(100, text="Analysis complete!")
        progress_bar.empty()

        # Results header
        st.markdown('<hr style="margin: 2rem 0; opacity: 0.2;">', unsafe_allow_html=True)
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
                risk_message = "Response is generally consistent with your question"
            elif risk < 60:
                risk_color = "#ffc107"
                risk_emoji = "🟡"
                risk_level = "Moderate Risk"
                risk_message = "Selective verification of key facts recommended"
            else:
                risk_color = "#dc3545"
                risk_emoji = "🔴"
                risk_level = "High Risk"
                risk_message = "Critical review needed - potential hallucinations detected"
            
            st.markdown(f"""
            <div class="risk-meter">
                <h3 style="margin-top: 0; color: {risk_color};">{risk_emoji} {risk_level}</h3>
                <div style="background: #f0f0f0; height: 30px; border-radius: 15px; margin: 10px 0;">
                    <div style="background: {risk_color}; width: {risk}%; height: 30px; border-radius: 15px; text-align: center; line-height: 30px; color: white; font-weight: bold;">
                        {risk:.1f}%
                    </div>
                </div>
                <p style="margin-bottom: 0; color: #666;">{risk_message}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_stats:
            st.markdown("### 📈 Quick Stats")
            
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            with col_stat1:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number">{result.metadata['num_sentences']}</div>
                    <div class="stat-label">Sentences</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_stat2:
                low_risk = sum(1 for s in result.sentence_scores if s.risk < 30)
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number">{low_risk}</div>
                    <div class="stat-label">Low Risk</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_stat3:
                high_risk = sum(1 for s in result.sentence_scores if s.risk >= 60)
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number">{high_risk}</div>
                    <div class="stat-label">High Risk</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_stat4:
                verified = sum(1 for fr in fact_results.values() if fr.verification_status == "confirmed")
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number">{verified}</div>
                    <div class="stat-label">Verified Facts</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('<hr style="margin: 2rem 0; opacity: 0.2;">', unsafe_allow_html=True)

        # Detailed sentence analysis
        st.markdown("## 📝 Sentence-by-Sentence Analysis")
        
        for idx, sentence_score in enumerate(result.sentence_scores, 1):
            risk_level, badge_class, emoji = get_risk_level(sentence_score.risk)
            fact_result = fact_results.get(sentence_score.sentence)
            analysis = generate_analysis(
                sentence_score.sentence, 
                sentence_score.similarity, 
                sentence_score.risk,
                fact_result
            )
            
            # Create a styled card for each sentence
            fact_html = ""
            if fact_result and fact_result.has_factual_content:
                fact_badge_class, fact_emoji = get_fact_status_badge(fact_result.verification_status)
                
                # Format extracted info
                extracted_info = []
                if fact_result.extracted_entities:
                    extracted_info.append(f"👤 {', '.join(fact_result.extracted_entities[:3])}")
                if fact_result.extracted_dates:
                    extracted_info.append(f"📅 {', '.join(fact_result.extracted_dates[:2])}")
                
                extracted_html = f"<div style='font-size:0.9rem; color:#666; margin-top:0.5rem;'>{' | '.join(extracted_info)}</div>" if extracted_info else ""
                
                # Source link if available
                source_html = ""
                if fact_result.wiki_url:
                    source_html = f"<div style='margin-top:0.5rem;'><a href='{fact_result.wiki_url}' target='_blank' class='source-link'>📚 Wikipedia: {fact_result.wiki_match}</a></div>"
                
                fact_html = f"""
                <div class="fact-check-box">
                    <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                        <span class="status-badge {fact_badge_class}">{fact_emoji} Fact: {fact_result.verification_status}</span>
                        <span style="font-size:0.8rem; color:#666;">confidence: {fact_result.confidence:.0%}</span>
                    </div>
                    {extracted_html}
                    {source_html}
                </div>
                """
            
            st.markdown(f"""
            <div class="sentence-card">
                <div class="sentence-text">
                    <strong>{idx}.</strong> {sentence_score.sentence}
                </div>
                <div class="metric-row">
                    <div class="metric-item">
                        <div class="metric-label">Risk Level</div>
                        <div class="metric-value">
                            <span class="status-badge {badge_class}">{emoji} {risk_level}</span>
                        </div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Risk Score</div>
                        <div class="metric-value">{sentence_score.risk:.1f}%</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Similarity</div>
                        <div class="metric-value">{sentence_score.similarity:.2f}</div>
                    </div>
                </div>
                <div class="analysis-text">
                    {analysis}
                </div>
                {fact_html}
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<hr style="margin: 2rem 0; opacity: 0.2;">', unsafe_allow_html=True)

        # Simple similarity distribution
        st.markdown("## 📊 Response Coherence Overview")
        
        sims_pct = _compute_histogram_data(result.sentence_scores)
        
        # Create figure with a larger size for better visibility
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Create histogram with better styling
        n, bins, patches = ax.hist(sims_pct, bins=8, edgecolor='white', linewidth=1.5)
        
        # Color code the bars for better understanding
        for i, patch in enumerate(patches):
            if bins[i] < 40:
                patch.set_facecolor('#dc3545')  # Red - low similarity
                patch.set_alpha(0.8)
            elif bins[i] < 70:
                patch.set_facecolor('#ffc107')  # Yellow - medium similarity
                patch.set_alpha(0.8)
            else:
                patch.set_facecolor('#28a745')  # Green - high similarity
                patch.set_alpha(0.8)
        
        # Add labels and title
        ax.set_xlabel("Similarity with Question (%)", fontsize=12, fontweight='bold')
        ax.set_ylabel("Number of Sentences", fontsize=12, fontweight='bold')
        ax.set_xlim(0, 100)
        ax.grid(axis="y", alpha=0.3, linestyle='--')
        
        # Add value labels on top of bars
        for i, (rect, bin_val) in enumerate(zip(patches, bins)):
            height = rect.get_height()
            if height > 0:
                ax.text(rect.get_x() + rect.get_width()/2., height + 0.1,
                        f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        # Display the plot
        st.pyplot(fig, use_container_width=True)
        
        # Simple interpretation with better formatting
        low_pct = np.mean(sims_pct < 40) * 100
        mid_pct = np.mean((sims_pct >= 40) & (sims_pct < 70)) * 100
        high_pct = np.mean(sims_pct >= 70) * 100
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 10px; padding: 1.5rem; margin: 1rem 0; border-left: 4px solid #667eea;">
            <h4 style="margin-top: 0; color: #333;">📈 Understanding the Results</h4>
            <p style="margin-bottom: 0.5rem;">The histogram shows how each sentence in the response relates to your question:</p>
            <ul style="margin-bottom: 0;">
                <li><span style="color: #dc3545; font-weight: bold;">🔴 Low similarity ({low_pct:.1f}%)</span> — sentences that may be off-topic or hallucinated</li>
                <li><span style="color: #ffc107; font-weight: bold;">🟡 Medium similarity ({mid_pct:.1f}%)</span> — sentences that are somewhat related but may need verification</li>
                <li><span style="color: #28a745; font-weight: bold;">🟢 High similarity ({high_pct:.1f}%)</span> — sentences that closely match your question</li>
            </ul>
            <p style="margin-top: 1rem; margin-bottom: 0; font-style: italic; color: #555;">
                <strong>Note:</strong> Blue fact-check boxes show automatic verification of names, dates, and events against Wikipedia.
            </p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
