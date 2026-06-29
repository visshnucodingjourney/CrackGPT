import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-me')
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        'SQLALCHEMY_DATABASE_URI', f"sqlite:///{os.path.join(BASE_DIR, 'crackgpt.db')}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
    GEMINI_MODEL = os.environ.get('GEMINI_MODEL', 'gemini-1.5-flash')

    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    INDEX_FOLDER = os.path.join(BASE_DIR, 'data')
    ALLOWED_EXTENSIONS = {'pdf', 'txt'}
    MAX_CONTENT_LENGTH = 20 * 1024 * 1024  # 20 MB

    EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

    # Adaptive RAG knobs
    MIN_RETRIEVED_CHUNKS = 2
    MAX_RETRIEVED_CHUNKS = 6
    SIMILARITY_GAP_THRESHOLD = 0.08  # stop adding chunks once relevance drops sharply

    QUESTIONS_PER_SESSION = 5
