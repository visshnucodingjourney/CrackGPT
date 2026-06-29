from datetime import datetime, timezone
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


def now():
    return datetime.now(timezone.utc)


class Document(db.Model):
    """An uploaded notes file (PDF or TXT), chunked + embedded into a FAISS index."""
    __tablename__ = 'documents'

    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    index_path = db.Column(db.String(255), nullable=False)   # path to .faiss file
    chunks_path = db.Column(db.String(255), nullable=False)  # path to pickled chunk texts
    num_chunks = db.Column(db.Integer, default=0)
    uploaded_at = db.Column(db.DateTime, default=now)

    sessions = db.relationship('QuizSession', backref='document', lazy='dynamic')


class QuizSession(db.Model):
    """One round of generated questions for a document."""
    __tablename__ = 'quiz_sessions'

    id = db.Column(db.Integer, primary_key=True)
    document_id = db.Column(db.Integer, db.ForeignKey('documents.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=now)
    completed_at = db.Column(db.DateTime)
    total_score = db.Column(db.Float, default=0.0)
    max_score = db.Column(db.Float, default=0.0)

    questions = db.relationship('Question', backref='session', lazy='dynamic',
                                 cascade='all, delete-orphan')

    @property
    def percentage(self):
        if not self.max_score:
            return 0
        return round((self.total_score / self.max_score) * 100, 1)

    @property
    def is_complete(self):
        return self.completed_at is not None


class Question(db.Model):
    """A single auto-generated interview question tied to a source chunk."""
    __tablename__ = 'questions'

    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('quiz_sessions.id'), nullable=False)
    question_text = db.Column(db.Text, nullable=False)
    ideal_answer = db.Column(db.Text, nullable=False)
    source_context = db.Column(db.Text)
    difficulty = db.Column(db.String(20), default='medium')  # easy, medium, hard
    order_index = db.Column(db.Integer, default=0)

    answer = db.relationship('Answer', backref='question', uselist=False,
                              cascade='all, delete-orphan')


class Answer(db.Model):
    """The user's submitted answer + Gemini's scoring of it."""
    __tablename__ = 'answers'

    id = db.Column(db.Integer, primary_key=True)
    question_id = db.Column(db.Integer, db.ForeignKey('questions.id'), nullable=False)
    user_answer = db.Column(db.Text)
    score = db.Column(db.Float, default=0.0)        # 0-10
    max_score = db.Column(db.Float, default=10.0)
    missing_points = db.Column(db.Text)              # newline-separated
    feedback = db.Column(db.Text)
    submitted_at = db.Column(db.DateTime, default=now)
