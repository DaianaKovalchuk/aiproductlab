from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


_MODEL: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """
    Lazy-load the sentence-transformers model to avoid slow startup.
    """
    global _MODEL
    if _MODEL is None:
        # Мультиязычная модель, чтобы лучше работать с русскими и англ. источниками
        _MODEL = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return _MODEL


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0, 0])


def _split_into_sentences(text: str) -> List[str]:
    """
    Очень простый сплиттер по предложениям.
    Для продакшена лучше заменить на spaCy / razdel и т.п.,
    но здесь достаточно базового варианта.
    """
    raw = [s.strip() for s in text.replace("?\n", "? ").replace("!\n", "! ").replace(".\n", ". ").split(".")]
    sentences: List[str] = []
    for s in raw:
        s = s.strip()
        if not s:
            continue
        # Возвращаем точку обратно, чтобы предложения выглядели естественно
        if not s.endswith((".", "?", "!")):
            s = s + "."
        sentences.append(s)
    return sentences


@dataclass
class SentenceScore:
    sentence: str
    similarity: float  # 0–1
    risk: float  # 0–100


@dataclass
class AnalysisResult:
    overall_risk: float  # 0–100
    question_embedding: np.ndarray
    answer_embedding: np.ndarray
    sentence_scores: List[SentenceScore]
    metadata: Dict[str, Any]


def analyze_semantic_consistency(question: str, answer: str) -> AnalysisResult:
    """
    Базовый семантический анализ согласованности ответа с вопросом.

    Механика:
    - считаем эмбеддинги вопроса и всего ответа;
    - делим ответ на предложения и считаем similarity каждого предложения с вопросом;
    - общий риск галлюцинаций = 100 - mean_similarity * 100, скорректированный дисперсией.
    """
    model = get_model()

    question = question.strip()
    answer = answer.strip()

    if not question or not answer:
        raise ValueError("Question and answer must be non-empty.")

    sentences = _split_into_sentences(answer)
    # Гарантируем хотя бы одно предложение
    if not sentences:
        sentences = [answer]

    texts_for_embedding = [question, answer] + sentences
    embeddings = model.encode(texts_for_embedding, convert_to_numpy=True, normalize_embeddings=True)

    q_emb = embeddings[0]
    a_emb = embeddings[1]
    sent_embs = embeddings[2:]

    # Сходство вопроса и ответа целиком
    qa_sim = _cosine(q_emb, a_emb)  # 0–1

    # Сходство вопроса с каждым предложением
    sent_sims: List[float] = []
    sent_scores: List[SentenceScore] = []
    for sent, emb in zip(sentences, sent_embs):
        sim = _cosine(q_emb, emb)
        sent_sims.append(sim)

    sent_sims_np = np.array(sent_sims, dtype=float)
    mean_sim = float(sent_sims_np.mean())
    std_sim = float(sent_sims_np.std()) if len(sent_sims_np) > 1 else 0.0

    # Базовый риск: чем ниже средняя схожесть, тем выше риск
    base_risk = 100.0 * (1.0 - mean_sim)

    # Усиливаем риск, если предложения очень неоднородны по смыслу (большая дисперсия)
    dispersion_factor = min(std_sim * 100.0, 30.0)  # ограничиваем вклад

    overall_risk = float(np.clip(base_risk + dispersion_factor, 0.0, 100.0))

    # Индивидуальный риск для предложения: инверсия схожести + поправка на разброс
    for sent, sim in zip(sentences, sent_sims_np):
        risk = 100.0 * (1.0 - float(sim))
        # если предложение сильно выбивается ниже средней схожести, немного увеличим риск
        if sim < mean_sim - std_sim:
            risk += 10.0
        risk = float(np.clip(risk, 0.0, 100.0))
        sent_scores.append(
            SentenceScore(
                sentence=sent,
                similarity=float(sim),
                risk=risk,
            )
        )

    metadata: Dict[str, Any] = {
        "qa_similarity": qa_sim,
        "mean_sentence_similarity": mean_sim,
        "std_sentence_similarity": std_sim,
        "num_sentences": len(sentences),
    }

    return AnalysisResult(
        overall_risk=overall_risk,
        question_embedding=q_emb,
        answer_embedding=a_emb,
        sentence_scores=sent_scores,
        metadata=metadata,
    )

