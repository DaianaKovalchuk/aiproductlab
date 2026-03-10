import streamlit as st
import numpy as np
from dotenv import load_dotenv

from pdf_report import build_pdf_bytes
from semantic_analyzer import analyze_semantic_consistency
from fact_checker import fact_check_sentences, FactCheckResult

st.title("Тестовый запуск с импортами")
st.write("Проверяем импорт fact_checker")
