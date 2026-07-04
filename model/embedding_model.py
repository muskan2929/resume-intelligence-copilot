"""
embedding_model.py

Responsible for:
1. Loading the sentence-transformers embedding model (all-MiniLM-L6-v2).
2. Generating embeddings for arbitrary text (resume sections, job description).
3. Computing cosine similarity between embeddings.
4. Providing a convenience function that scores a full resume against
   a full job description, section by section.

This is the semantic "brain" of the project — it replaces old-school
keyword matching with meaning-based comparison.
"""

from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ------------------------------------------------------------------
# 1. MODEL LOADING (singleton pattern — load once, reuse everywhere)
# ------------------------------------------------------------------

_model = None  # module-level cache so we don't reload the model repeatedly


def load_embedding_model() -> SentenceTransformer:
    """
    Loads the MiniLM sentence embedding model.
    Uses a simple cache so the (relatively heavy) model is only
    downloaded/loaded into memory once per Colab session.

    Returns:
        A loaded SentenceTransformer model instance.
    """
    global _model
    if _model is None:
        print("⏳ Loading embedding model: all-MiniLM-L6-v2 ...")
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        print("✅ Embedding model loaded.")
    return _model


# ------------------------------------------------------------------
# 2. EMBEDDING GENERATION
# ------------------------------------------------------------------

def get_embedding(text: str) -> np.ndarray:
    """
    Converts a piece of text into a semantic embedding vector.

    Args:
        text: Any text (a sentence, paragraph, or skill list).

    Returns:
        A 1D numpy array representing the text's embedding
        (384-dimensional for MiniLM-L6-v2).
    """
    model = load_embedding_model()

    # Empty text would break similarity math, so guard against it
    if not text or not text.strip():
        text = " "

    embedding = model.encode(text, convert_to_numpy=True)
    return embedding


def get_embeddings_batch(texts: List[str]) -> np.ndarray:
    """
    Converts a list of texts into a batch of embeddings at once.
    Batching is much faster than calling get_embedding() in a loop.

    Args:
        texts: List of text strings.

    Returns:
        2D numpy array of shape (num_texts, embedding_dim).
    """
    model = load_embedding_model()
    safe_texts = [t if t and t.strip() else " " for t in texts]
    return model.encode(safe_texts, convert_to_numpy=True)


# ------------------------------------------------------------------
# 3. SIMILARITY COMPUTATION
# ------------------------------------------------------------------

def compute_similarity(text_a: str, text_b: str) -> float:
    """
    Computes cosine similarity between two pieces of text.

    Args:
        text_a: First text (e.g. resume section).
        text_b: Second text (e.g. job description).

    Returns:
        A similarity score between 0.0 and 1.0
        (higher = more semantically similar).
    """
    emb_a = get_embedding(text_a).reshape(1, -1)
    emb_b = get_embedding(text_b).reshape(1, -1)

    similarity = cosine_similarity(emb_a, emb_b)[0][0]

    # Cosine similarity can technically be slightly negative for
    # unrelated text — clip to a clean 0-1 range for easier scoring.
    return float(np.clip(similarity, 0.0, 1.0))


def compute_similarity_matrix(
    resume_skills: List[str], job_skills: List[str]
) -> np.ndarray:
    """
    Computes a full similarity matrix between every resume skill and
    every required job skill. This is the foundation for the
    "Missing Skills Heatmap" and "Skill Transferability Score" features
    built in later phases.

    Args:
        resume_skills: List of skills extracted from the resume.
        job_skills: List of required skills from the job description.

    Returns:
        2D numpy array of shape (len(resume_skills), len(job_skills)),
        where entry [i][j] = similarity between resume_skills[i] and
        job_skills[j].
    """
    if not resume_skills or not job_skills:
        return np.zeros((len(resume_skills), len(job_skills)))

    resume_embeddings = get_embeddings_batch(resume_skills)
    job_embeddings = get_embeddings_batch(job_skills)

    similarity_matrix = cosine_similarity(resume_embeddings, job_embeddings)
    return np.clip(similarity_matrix, 0.0, 1.0)


# ------------------------------------------------------------------
# 4. SECTION-LEVEL SCORING (resume sections vs job description)
# ------------------------------------------------------------------

def score_resume_sections_against_jd(
    resume_sections: Dict[str, str], job_description: str
) -> Dict[str, float]:
    """
    Compares each resume section (skills, experience, projects, etc.)
    against the full job description and returns a similarity score
    for each section.

    Args:
        resume_sections: Dict of section_name -> section_text
                          (output of resume_parser.split_into_sections).
        job_description: Full job description text.

    Returns:
        Dict mapping section_name -> similarity score (0.0 - 1.0).
    """
    scores = {}
    for section_name, section_text in resume_sections.items():
        scores[section_name] = compute_similarity(section_text, job_description)
    return scores


def compute_overall_match_score(
    resume_sections: Dict[str, str], job_description: str
) -> Dict[str, float]:
    """
    Computes the final "Resume Match Score" — an overall percentage
    plus a confidence score, based on weighted section similarities.

    Sections are weighted by importance: skills and experience matter
    more to recruiters than, say, the header/contact info.

    Args:
        resume_sections: Dict of section_name -> section_text.
        job_description: Full job description text.

    Returns:
        Dict with:
            - "overall_score": weighted overall match (0-100)
            - "confidence": how much of the resume had usable content (0-100)
            - "section_scores": raw similarity per section (0-1)
    """
    # Importance weight for each possible section.
    # Sections not present in the resume are simply skipped.
    weights = {
        "skills": 0.30,
        "experience": 0.25,
        "projects": 0.20,
        "summary": 0.10,
        "education": 0.05,
        "achievements": 0.05,
        "leadership": 0.05,
    }

    section_scores = score_resume_sections_against_jd(resume_sections, job_description)

    weighted_sum = 0.0
    total_weight_used = 0.0

    for section_name, score in section_scores.items():
        weight = weights.get(section_name, 0.02)  # small default weight
        weighted_sum += score * weight
        total_weight_used += weight

    # Normalize so the score isn't unfairly penalized if some
    # sections (e.g. "leadership") are missing from the resume.
    overall_score = (weighted_sum / total_weight_used) * 100 if total_weight_used else 0.0

    # Confidence reflects how much of the *expected* weight we
    # actually had data for (i.e. how complete the resume is).
    max_possible_weight = sum(weights.values())
    confidence = (total_weight_used / max_possible_weight) * 100

    return {
        "overall_score": round(overall_score, 2),
        "confidence": round(min(confidence, 100.0), 2),
        "section_scores": {k: round(v, 3) for k, v in section_scores.items()},
    }
