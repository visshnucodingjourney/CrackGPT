# CrackGPT — AI Interview Coach

Upload your study notes, get auto-generated interview questions via Adaptive RAG + Gemini, answer them, and receive instant scoring with missing points and ideal answers.

## Stack
Python · Flask · LangChain-style RAG (custom adaptive retriever) · FAISS · Sentence-Transformers · Google Gemini API · SQLite · Bootstrap · Chart.js

## How it works

1. **Upload** a PDF/TXT of your notes → text is chunked and embedded with
   `sentence-transformers/all-MiniLM-L6-v2`, then stored in a per-document
   FAISS index.
2. **Adaptive retrieval**: instead of a fixed top-k, the retriever pulls
   chunks until the similarity score drops sharply (a relevance "knee"),
   so each topic gets just enough context — no more, no less.
3. **Question generation**: Gemini is prompted with the retrieved context
   to produce a mixed-difficulty set of questions + ideal answers.
4. **Scoring**: each of your answers is graded by Gemini against the ideal
   answer — returning a 0–10 score, a list of missing points, and short
   feedback.
5. **Dashboard**: tracks your score over time across sessions with a
   Chart.js trend line.

## Local setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and add your GEMINI_API_KEY

python app.py
# visit http://localhost:5000
```

## Getting a Gemini API key

1. Go to https://aistudio.google.com/app/apikey
2. Create a key and paste it into `.env` as `GEMINI_API_KEY`

## Deployment (Render / Railway / Fly.io — any Flask-friendly host)

This repo is deploy-ready via the included `Procfile`:

```
web: gunicorn app:app
```

Steps (Render example):
1. Push this folder to a GitHub repo.
2. On Render: **New → Web Service** → connect the repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app`
5. Add environment variables: `GEMINI_API_KEY`, `SECRET_KEY`.
6. Deploy.

**Note on storage:** uploaded files, FAISS indexes, and the SQLite DB are
written to local disk (`uploads/`, `data/`, `crackgpt.db`). Most free-tier
hosts use ephemeral filesystems that reset on redeploy — fine for a demo,
but for persistence in production, swap `SQLALCHEMY_DATABASE_URI` for a
managed Postgres URL and point `UPLOAD_FOLDER`/`INDEX_FOLDER` at a mounted
volume or object storage (e.g. S3).

## Project structure

```
crackgpt/
├── app.py              # Flask routes
├── config.py            # settings (env-driven)
├── models.py             # SQLAlchemy models
├── rag_engine.py          # chunking, FAISS, adaptive retrieval, Gemini calls
├── requirements.txt
├── Procfile
├── .env.example
├── templates/
│   ├── base.html, index.html, quiz.html, results.html, dashboard.html, 404.html
├── static/css/style.css
├── uploads/              # raw uploaded notes (gitignored in practice)
└── data/                 # FAISS indexes + pickled chunks
```

## Notes / things to tune

- `QUESTIONS_PER_SESSION` (config.py) controls how many questions are generated per upload.
- `SIMILARITY_GAP_THRESHOLD` controls how aggressively adaptive retrieval cuts off chunks — lower = more chunks kept per query.
- Swap `GEMINI_MODEL` to `gemini-1.5-pro` in `.env` if you want higher-quality grading at the cost of latency/cost.
