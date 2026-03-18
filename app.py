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

# ========== HELPER FUNCTIONS ==========
@st.cache_resource
def load_semantic_model():
    """Cache the model to save memory"""
    return get_model()

def get_risk_level(risk_score: float) -> Tuple[str, str, str]:
    """Return risk level, color, and emoji based on risk score"""
    if risk_score < 30:
        return "Low", "badge-low", "🟢"
    elif risk_score < 60:
        return "Medium", "badge-medium", "🟡"
    else:
        return "High", "badge-high", "🔴"

def generate_analysis(sentence: str, similarity: float, risk_score: float) -> str:
    """Generate a one-sentence analysis based on semantic metrics"""
    if risk_score < 30:
        return f"✅ Strongly aligned with your question (similarity: {similarity:.2f}) - low hallucination risk"
    elif risk_score < 60:
        return f"⚡ Moderately relevant but may need verification (similarity: {similarity:.2f})"
    else:
        return f"⚠️ Low relevance to your question (similarity: {similarity:.2f}) - potential hallucination"

def _compute_histogram_data(sentence_scores: List[SentenceScore]):
    sims = np.array([s.similarity for s in sentence_scores], dtype=float)
    sims_pct = sims * 100.0
    return sims_pct

# ========== MAIN APPLICATION ==========
def main():
    # Custom title
    st.markdown('<h1 class="main-title">🔍 LLM Hallucination Checker</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Advanced AI response validation with semantic analysis</p>', unsafe_allow_html=True)
    
    # Welcome card
    with st.container():
        st.markdown("""
        <div class="card">
            <h3 style="color: white; margin-top: 0;">🎯 How it works</h3>
            <p style="color: rgba(255,255,255,0.9); margin-bottom: 0;">
                Our engine analyzes LLM responses using semantic coherence:<br>
                <strong>Semantic analysis</strong> — measures how well each sentence relates to your question to identify potential hallucinations
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
                progress_bar.progress(50, text="Analyzing semantic coherence...")
                result = analyze_semantic_consistency(question, answer)
                progress_bar.progress(100, text="Analysis complete!")
            except Exception as e:
                st.error(f"❌ Semantic analysis failed: {str(e)}")
                return
        
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
            
            col_stat1, col_stat2, col_stat3 = st.columns(3)
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

        st.markdown('<hr style="margin: 2rem 0; opacity: 0.2;">', unsafe_allow_html=True)

        # Detailed sentence analysis
        st.markdown("## 📝 Sentence-by-Sentence Analysis")
        
        for idx, sentence_score in enumerate(result.sentence_scores, 1):
            risk_level, badge_class, emoji = get_risk_level(sentence_score.risk)
            analysis = generate_analysis(
                sentence_score.sentence, 
                sentence_score.similarity, 
                sentence_score.risk
            )
            
            # Create a styled card for each sentence
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
                <strong>Note:</strong> Higher similarity indicates better alignment with your question and lower hallucination risk.
            </p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
