import io
import os
import re
import requests
import time
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
import json

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

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
class SentenceScore:
    sentence: str
    similarity: float
    risk: float

@dataclass
class AnalysisResult:
    sentence_scores: List[SentenceScore]
    overall_risk: float
    metadata: Dict[str, Any]

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

# ========== MODEL LOADING WITH FALLBACK ==========
@st.cache_resource
def get_model_with_fallback():
    """Load sentence transformer model with fallback options"""
    from sentence_transformers import SentenceTransformer
    
    models_to_try = [
        'paraphrase-multilingual-MiniLM-L12-v2',
        'paraphrase-multilingual-MiniLM-L6-v2',
        'paraphrase-MiniLM-L3-v2',
        'all-MiniLM-L6-v2'
    ]
    
    for model_name in models_to_try:
        try:
            st.info(f"🔄 Trying to load model: {model_name}...")
            model = SentenceTransformer(model_name)
            st.success(f"✅ Successfully loaded model: {model_name}")
            return model
        except Exception as e:
            st.warning(f"⚠️ Failed to load {model_name}: {str(e)}")
            continue
    
    st.error("❌ Could not load any sentence transformer model.")
    return None

# ========== SEMANTIC ANALYSIS FUNCTIONS ==========
def analyze_semantic_consistency(question: str, answer: str) -> AnalysisResult:
    """Analyze semantic consistency between question and answer sentences"""
    model = get_model_with_fallback()
    if model is None:
        raise Exception("Failed to load semantic model")
    
    # Split answer into sentences
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', answer)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return AnalysisResult(
            sentence_scores=[],
            overall_risk=0.0,
            metadata={"num_sentences": 0}
        )
    
    # Encode question and all sentences
    embeddings = model.encode([question] + sentences, convert_to_numpy=True, normalize_embeddings=True)
    question_emb = embeddings[0]
    sentence_embs = embeddings[1:]
    
    # Calculate cosine similarities
    similarities = np.dot(sentence_embs, question_emb)
    
    # Calculate risk scores (inverse of similarity, scaled to 0-100)
    risks = (1 - similarities) * 100
    
    # Create sentence scores
    sentence_scores = [
        SentenceScore(sentence=sentences[i], similarity=float(similarities[i]), risk=float(risks[i]))
        for i in range(len(sentences))
    ]
    
    # Calculate overall risk (weighted average)
    overall_risk = float(np.mean(risks))
    
    return AnalysisResult(
        sentence_scores=sentence_scores,
        overall_risk=overall_risk,
        metadata={"num_sentences": len(sentences)}
    )

# ========== HELPER FUNCTIONS ==========
def extract_entities_and_dates(text: str) -> Tuple[List[str], List[str]]:
    """Extract potential entities and dates from text with improved accuracy"""
    # Extract dates (years, full dates)
    date_patterns = [
        r'\b\d{4}\s*г\.?\b',  # 2024 г.
        r'\b\d{1,2}\s+[а-яА-Я]+\s+\d{4}\b',  # 15 мая 2024
        r'\b\d{1,2}\.\d{1,2}\.\d{4}\b',  # 15.05.2024
        r'\b\d{4}\s+год\w*\b',  # 2024 год
        r'\b\d{4}\b',  # просто год
    ]
    
    dates = []
    for pattern in date_patterns:
        dates.extend(re.findall(pattern, text, re.IGNORECASE))
    
    # Словарь известных исторических событий и личностей
    known_entities = {
        'Крещение Руси': ['крещение', '988', 'владимир'],
        'Владимир': ['князь', 'красное солнышко', 'крещение'],
        'Иван Грозный': ['иван', 'грозный', 'царь'],
        'Петр I': ['петр', 'первый', 'великий', 'император'],
        'Екатерина II': ['екатерина', 'вторая', 'великая'],
        'Александр Невский': ['александр', 'невский', 'князь'],
        'Дмитрий Донской': ['дмитрий', 'донской', 'куликовская'],
        'Михаил Ломоносов': ['ломоносов', 'михаил', 'ученый'],
        'Александр Пушкин': ['пушкин', 'александр', 'поэт'],
    }
    
    entities = []
    text_lower = text.lower()
    
    # Проверяем известные сущности
    for entity, keywords in known_entities.items():
        if any(keyword in text_lower for keyword in keywords):
            entities.append(entity)
    
    # Находим все слова с заглавной буквы внутри предложения (не в начале)
    sentences = re.split(r'[.!?]+', text)
    for sentence in sentences:
        words = sentence.strip().split()
        for i, word in enumerate(words):
            # Проверяем, что слово начинается с заглавной буквы
            if re.match(r'^[А-ЯЁ][а-яё]*$', word) and len(word) > 2:
                # Игнорируем первое слово предложения и стоп-слова
                stop_words = {'Это', 'Он', 'Она', 'Они', 'Мы', 'Вы', 'Таким', 'Также', 'Кроме', 
                             'Все', 'То', 'Что', 'Как', 'Так', 'Где', 'Когда'}
                if i > 0 and word not in stop_words:
                    entities.append(word)
            
            # Находим составные сущности (несколько слов подряд с заглавных)
            if i < len(words) - 1:
                if re.match(r'^[А-ЯЁ][а-яё]*$', word) and re.match(r'^[А-ЯЁ][а-яё]*$', words[i+1]):
                    entities.append(f"{word} {words[i+1]}")
    
    # Убираем дубликаты
    entities = list(dict.fromkeys(entities))
    
    return entities, dates

def evaluate_relevance(sentence: str, title: str, snippet: str) -> float:
    """Оценивает релевантность найденной статьи"""
    sentence_lower = sentence.lower()
    title_lower = title.lower()
    snippet_lower = snippet.lower()
    
    # Ключевые слова из предложения
    keywords = set(re.findall(r'\b\w{4,}\b', sentence_lower))
    
    # Специальная обработка для исторических событий
    if 'крещение' in sentence_lower and 'руси' in sentence_lower:
        if 'крещение руси' in title_lower:
            return 1.0
        if 'крещение' in title_lower and 'руси' in title_lower:
            return 0.95
    
    if not keywords:
        return 0.0
    
    # Считаем совпадения в заголовке (более важны)
    title_matches = sum(2 for kw in keywords if kw in title_lower)
    
    # Считаем совпадения в сниппете
    snippet_matches = sum(1 for kw in keywords if kw in snippet_lower)
    
    # Бонус за точное совпадение ключевых фраз
    key_phrases = ['крещение руси', '988 год', 'князь владимир']
    for phrase in key_phrases:
        if phrase in sentence_lower and phrase in title_lower:
            title_matches += 5
    
    total = title_matches + snippet_matches
    max_possible = len(keywords) * 3 + 5
    
    return total / max_possible if max_possible > 0 else 0.0

def search_wikipedia(query: str, lang: str = "ru") -> Optional[Dict[str, Any]]:
    """Search Wikipedia for a given query with improved relevance"""
    try:
        headers = {
            'User-Agent': 'LLMHallucinationChecker/1.0 (https://hallucheck.streamlit.app)'
        }
        
        # Специальная обработка для известных исторических событий
        if "Крещение Руси" in query or "988" in query or ("крещение" in query.lower() and "руси" in query.lower()):
            # Прямой запрос к статье о Крещении Руси
            extract_params = {
                "action": "query",
                "titles": "Крещение Руси",
                "prop": "extracts",
                "exintro": 1,
                "explaintext": 1,
                "format": "json",
                "utf8": 1
            }
            
            response = requests.get(
                f"https://{lang}.wikipedia.org/w/api.php",
                params=extract_params,
                headers=headers,
                timeout=5
            )
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    pages = data.get("query", {}).get("pages", {})
                    for page_id, page in pages.items():
                        if page_id != "-1" and page.get("extract"):
                            return {
                                "title": "Крещение Руси",
                                "snippet": page["extract"][:500] + "...",
                                "url": f"https://{lang}.wikipedia.org/wiki/Крещение_Руси",
                                "pageid": page_id
                            }
                except:
                    pass
        
        # Стандартный поиск
        search_params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "utf8": 1,
            "srlimit": 5
        }
        
        response = requests.get(
            f"https://{lang}.wikipedia.org/w/api.php",
            params=search_params,
            headers=headers,
            timeout=5
        )
        
        if response.status_code != 200:
            return None
            
        try:
            data = response.json()
        except ValueError:
            return None
        
        search_results = data.get("query", {}).get("search", [])
        if not search_results:
            return None
        
        # Оцениваем релевантность результатов
        best_result = None
        best_relevance = 0
        best_result_data = None
        
        for result in search_results:
            title = result["title"]
            snippet = result.get("snippet", "")
            
            # Очищаем сниппет от HTML тегов
            snippet = re.sub(r'<[^>]+>', '', snippet)
            
            # Оцениваем релевантность
            relevance = 0
            
            # Проверяем точное совпадение запроса в заголовке
            if query.lower() in title.lower():
                relevance += 3
            
            # Проверяем отдельные слова запроса в заголовке
            query_words = set(query.lower().split())
            title_words = set(title.lower().split())
            common_words = query_words & title_words
            relevance += len(common_words) * 2
            
            # Проверяем наличие слов запроса в сниппете
            for word in query_words:
                if len(word) > 3 and word in snippet.lower():
                    relevance += 1
            
            if relevance > best_relevance:
                best_relevance = relevance
                best_result = result
                best_result_data = (title, snippet)
        
        if not best_result or best_relevance < 2:  # Минимальный порог релевантности
            return None
        
        # Получаем полный текст статьи
        extract_params = {
            "action": "query",
            "pageids": best_result["pageid"],
            "prop": "extracts",
            "exintro": 1,
            "explaintext": 1,
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
        page_data = pages.get(str(best_result["pageid"]), {})
        extract = page_data.get("extract", "")
        
        if extract:
            snippet = extract[:500] + "..." if len(extract) > 500 else extract
            return {
                "title": best_result["title"],
                "snippet": snippet,
                "url": f"https://{lang}.wikipedia.org/wiki/{best_result['title'].replace(' ', '_')}",
                "pageid": best_result["pageid"]
            }
    
    except Exception as e:
        st.warning(f"Wikipedia search error for '{query}': {str(e)}")
    
    return None

def search_wikidata_by_date(date: str) -> Optional[Dict[str, Any]]:
    """Search Wikidata for events on a specific date"""
    try:
        date_clean = re.sub(r'\s*г\.?\s*$', '', date).strip()
        year_match = re.search(r'\b(\d{4})\b', date_clean)
        
        if year_match:
            year = year_match.group(1)
            
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
    
    # Не проверяем слишком короткие предложения или предложения без сущностей
    if len(sentence) < 15 or (len(entities) == 0 and len(dates) == 0):
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
    
    # Создаем поисковые запросы на основе сущностей и дат
    search_queries = []
    
    # Сначала пробуем комбинации сущностей с датами
    if entities and dates:
        for entity in entities[:2]:
            for date in dates[:1]:
                search_queries.append(f"{entity} {date}")
    
    # Затем отдельные сущности
    search_queries.extend(entities[:3])
    
    # Затем даты с контекстом
    for date in dates[:1]:
        search_queries.append(f"события {date}")
        search_queries.append(date)
    
    # Пробуем каждый запрос
    best_match = None
    best_confidence = 0.0
    best_relevance = 0.0
    verification_status = "no_data"
    
    for query in search_queries[:4]:
        if not query or len(query) < 3:
            continue
        
        if len(query) > 100:
            query = query[:100]
        
        # Пробуем русскую Википедию
        result = search_wikipedia(query, "ru")
        if result:
            # Оцениваем релевантность
            relevance = evaluate_relevance(sentence, result["title"], result["snippet"])
            
            if relevance > best_relevance:
                best_relevance = relevance
                best_match = result
                
                # Рассчитываем confidence на основе релевантности
                confidence = 0.5 + relevance * 0.4
                best_confidence = min(confidence, 0.95)
                
                # Проверяем ключевые слова для определения статуса
                sentence_keywords = set(re.findall(r'\b\w{4,}\b', sentence.lower()))
                title_keywords = set(re.findall(r'\b\w{4,}\b', result["title"].lower()))
                snippet_keywords = set(re.findall(r'\b\w{4,}\b', result["snippet"].lower()))
                
                all_source_keywords = title_keywords | snippet_keywords
                common_keywords = sentence_keywords & all_source_keywords
                
                if len(common_keywords) >= 4 or relevance > 0.8:
                    verification_status = "confirmed"
                elif len(common_keywords) >= 2 or relevance > 0.5:
                    verification_status = "questionable"
                else:
                    verification_status = "no_data"
        
        # Небольшая задержка между запросами
        time.sleep(0.3)
    
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

        # Detailed sentence analysis - БЕЗ НОМЕРАЦИИ И С ПРОСТЫМ ВЫВОДОМ
        st.markdown("## 📝 Sentence Analysis")
        
        for sentence_score in result.sentence_scores:
            risk_level, badge_class, emoji = get_risk_level(sentence_score.risk)
            fact_result = fact_results.get(sentence_score.sentence)
            analysis = generate_analysis(
                sentence_score.sentence, 
                sentence_score.similarity, 
                sentence_score.risk,
                fact_result
            )
            
            # Простой вывод без сложного HTML
            st.markdown(f"**{sentence_score.sentence}**")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Risk Level", f"{emoji} {risk_level}")
            with col2:
                st.metric("Risk Score", f"{sentence_score.risk:.1f}%")
            with col3:
                st.metric("Similarity", f"{sentence_score.similarity:.2f}")
            
            st.info(analysis)
            
            if fact_result and fact_result.has_factual_content:
                with st.expander("🔍 Fact Check Details"):
                    st.write(f"**Status:** {fact_result.verification_status}")
                    st.write(f"**Confidence:** {fact_result.confidence:.0%}")
                    if fact_result.extracted_entities:
                        st.write(f"**Entities:** {', '.join(fact_result.extracted_entities[:3])}")
                    if fact_result.extracted_dates:
                        st.write(f"**Dates:** {', '.join(fact_result.extracted_dates[:2])}")
                    if fact_result.wiki_url:
                        st.write(f"**Source:** [{fact_result.wiki_match}]({fact_result.wiki_url})")
            
            st.divider()

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
