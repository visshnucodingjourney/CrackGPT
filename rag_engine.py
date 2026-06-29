"""
rag_engine.py
─────────────
Handles everything around turning uploaded notes into:
  1. Text chunks → embeddings → a FAISS index (per document)
  2. Adaptive retrieval: pulls a *variable* number of chunks depending on how
     quickly similarity scores drop off, instead of a fixed top-k.
  3. Gemini calls for question generation and answer scoring.
"""

import os
import re
import json
import pickle
import numpy as np
import faiss
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

_embedder = None


def get_embedder(model_name):
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(model_name)
    return _embedder


def configure_gemini(api_key):
    if api_key:
        genai.configure(api_key=api_key)


# ─── Text extraction ──────────────────────────────────────────────────────

def extract_text(filepath):
    ext = filepath.rsplit('.', 1)[-1].lower()
    if ext == 'pdf':
        reader = PdfReader(filepath)
        return '\n'.join(page.extract_text() or '' for page in reader.pages)
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


def chunk_text(text, chunk_size=800, overlap=120):
    """Simple sliding-window chunker on whitespace-normalized text."""
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# ─── Index build / load ───────────────────────────────────────────────────

def build_index(chunks, embed_model_name):
    embedder = get_embedder(embed_model_name)
    vectors = embedder.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine sim via normalized inner product
    index.add(vectors.astype('float32'))
    return index


def save_index(index, chunks, index_path, chunks_path):
    faiss.write_index(index, index_path)
    with open(chunks_path, 'wb') as f:
        pickle.dump(chunks, f)


def load_index(index_path, chunks_path):
    index = faiss.read_index(index_path)
    with open(chunks_path, 'rb') as f:
        chunks = pickle.load(f)
    return index, chunks


# ─── Adaptive retrieval ────────────────────────────────────────────────────

def adaptive_retrieve(query, index, chunks, embed_model_name,
                       min_k=2, max_k=6, gap_threshold=0.08):
    """
    Retrieves a *variable* number of chunks: starts with max_k candidates,
    then walks the similarity scores and stops as soon as the drop between
    consecutive scores exceeds gap_threshold (a "knee" in relevance) —
    but never returns fewer than min_k or more than max_k.
    """
    embedder = get_embedder(embed_model_name)
    qvec = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    k = min(max_k, len(chunks)) or 1
    scores, idxs = index.search(qvec.astype('float32'), k)
    scores, idxs = scores[0], idxs[0]

    keep = min_k
    for i in range(min_k, len(scores)):
        if scores[i - 1] - scores[i] > gap_threshold:
            break
        keep = i + 1

    selected = [chunks[i] for i in idxs[:keep] if i < len(chunks)]
    return selected


# ─── Gemini: question generation ──────────────────────────────────────────

QUESTION_GEN_PROMPT = """You are an expert technical interviewer. Using ONLY the
context below, generate {num_questions} interview questions that test real
understanding of the material (not just recall of exact wording).

For each question provide:
- "question": the interview question
- "ideal_answer": a thorough, correct model answer based on the context
- "difficulty": one of "easy", "medium", "hard"

Mix difficulties across the set. Respond with ONLY a JSON array, no markdown
fences, no commentary. Example format:
[{{"question": "...", "ideal_answer": "...", "difficulty": "medium"}}]

CONTEXT:
{context}
"""


def generate_questions(context_chunks, num_questions, model_name):
    context = '\n\n---\n\n'.join(context_chunks)
    prompt = QUESTION_GEN_PROMPT.format(num_questions=num_questions, context=context)

    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    raw = response.text.strip()
    raw = re.sub(r'^```json\s*|\s*```$', '', raw, flags=re.MULTILINE).strip()

    try:
        items = json.loads(raw)
    except json.JSONDecodeError:
        # Fall back: try to locate the first [ ... ] block
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        items = json.loads(match.group(0)) if match else []

    cleaned = []
    for item in items[:num_questions]:
        cleaned.append({
            'question': item.get('question', '').strip(),
            'ideal_answer': item.get('ideal_answer', '').strip(),
            'difficulty': item.get('difficulty', 'medium').strip().lower(),
        })
    return cleaned


# ─── Gemini: answer scoring ────────────────────────────────────────────────

SCORING_PROMPT = """You are grading an interview candidate's answer.

QUESTION:
{question}

IDEAL ANSWER (reference):
{ideal_answer}

CANDIDATE'S ANSWER:
{user_answer}

Score the candidate's answer from 0 to 10 based on correctness and
completeness versus the ideal answer. List specific missing points (as a
JSON array of short strings — empty array if nothing is missing). Give one
short paragraph of constructive feedback.

Respond with ONLY this JSON object, no markdown fences, no commentary:
{{"score": <number 0-10>, "missing_points": ["..."], "feedback": "..."}}
"""


def score_answer(question, ideal_answer, user_answer, model_name):
    if not user_answer or not user_answer.strip():
        return {
            'score': 0,
            'missing_points': ['No answer was submitted.'],
            'feedback': 'You did not provide an answer to this question.'
        }

    prompt = SCORING_PROMPT.format(
        question=question, ideal_answer=ideal_answer, user_answer=user_answer
    )
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    raw = response.text.strip()
    raw = re.sub(r'^```json\s*|\s*```$', '', raw, flags=re.MULTILINE).strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        result = json.loads(match.group(0)) if match else {
            'score': 0, 'missing_points': [], 'feedback': 'Could not score this answer automatically.'
        }

    score = float(result.get('score', 0))
    score = max(0.0, min(10.0, score))
    missing = result.get('missing_points', []) or []

    return {
        'score': score,
        'missing_points': missing,
        'feedback': result.get('feedback', '').strip(),
    }
