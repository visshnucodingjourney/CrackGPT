import os
import uuid
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify

from config import Config
from models import db, Document, QuizSession, Question, Answer
import rag_engine as rag


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['INDEX_FOLDER'], exist_ok=True)

    db.init_app(app)
    rag.configure_gemini(app.config['GEMINI_API_KEY'])

    with app.app_context():
        db.create_all()

    register_routes(app)
    return app


def allowed_file(filename, allowed_ext):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_ext


def register_routes(app):

    @app.route('/')
    def index():
        documents = Document.query.order_by(Document.uploaded_at.desc()).all()
        return render_template('index.html', documents=documents)

    # ─── Upload + index notes ──────────────────────────────────────────────

    @app.route('/upload', methods=['POST'])
    def upload():
        file = request.files.get('notes_file')
        if not file or file.filename == '':
            flash('Please choose a PDF or TXT file.', 'danger')
            return redirect(url_for('index'))

        if not allowed_file(file.filename, app.config['ALLOWED_EXTENSIONS']):
            flash('Only PDF or TXT files are supported.', 'danger')
            return redirect(url_for('index'))

        original_name = secure_filename(file.filename)
        unique_name = f"{uuid.uuid4().hex}_{original_name}"
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
        file.save(save_path)

        text = rag.extract_text(save_path)
        chunks = rag.chunk_text(text)

        if not chunks:
            flash('Could not extract any readable text from that file.', 'danger')
            return redirect(url_for('index'))

        index = rag.build_index(chunks, app.config['EMBEDDING_MODEL'])

        base = uuid.uuid4().hex
        index_path = os.path.join(app.config['INDEX_FOLDER'], f'{base}.faiss')
        chunks_path = os.path.join(app.config['INDEX_FOLDER'], f'{base}.pkl')
        rag.save_index(index, chunks, index_path, chunks_path)

        doc = Document(
            filename=unique_name,
            original_filename=original_name,
            index_path=index_path,
            chunks_path=chunks_path,
            num_chunks=len(chunks),
        )
        db.session.add(doc)
        db.session.commit()

        flash(f'"{original_name}" processed into {len(chunks)} chunks.', 'success')
        return redirect(url_for('generate_quiz', doc_id=doc.id))

    # ─── Generate questions (Adaptive RAG) ─────────────────────────────────

    @app.route('/generate/<int:doc_id>')
    def generate_quiz(doc_id):
        doc = Document.query.get_or_404(doc_id)
        index, chunks = rag.load_index(doc.index_path, doc.chunks_path)

        num_q = app.config['QUESTIONS_PER_SESSION']

        # Use a handful of broad seed queries so questions span the material,
        # not just whatever the first chunk happens to discuss.
        seed_queries = [
            'key concepts and definitions',
            'how this works in practice',
            'common pitfalls and edge cases',
            'comparisons and trade-offs',
            'practical applications',
        ]

        context_pool = []
        for q in seed_queries[:max(1, num_q // 2 + 1)]:
            context_pool.extend(rag.adaptive_retrieve(
                q, index, chunks, app.config['EMBEDDING_MODEL'],
                min_k=app.config['MIN_RETRIEVED_CHUNKS'],
                max_k=app.config['MAX_RETRIEVED_CHUNKS'],
                gap_threshold=app.config['SIMILARITY_GAP_THRESHOLD'],
            ))

        # de-dupe while preserving order
        seen = set()
        unique_context = []
        for c in context_pool:
            if c not in seen:
                seen.add(c)
                unique_context.append(c)

        try:
            generated = rag.generate_questions(
                unique_context or chunks[:4], num_q, app.config['GEMINI_MODEL']
            )
        except Exception as exc:
            flash(f'Question generation failed: {exc}', 'danger')
            return redirect(url_for('index'))

        if not generated:
            flash('Gemini did not return any questions. Try again.', 'danger')
            return redirect(url_for('index'))

        session = QuizSession(document_id=doc.id, max_score=len(generated) * 10.0)
        db.session.add(session)
        db.session.flush()

        for i, item in enumerate(generated):
            db.session.add(Question(
                session_id=session.id,
                question_text=item['question'],
                ideal_answer=item['ideal_answer'],
                difficulty=item['difficulty'],
                order_index=i,
            ))
        db.session.commit()

        return redirect(url_for('take_quiz', session_id=session.id))

    # ─── Take the quiz ──────────────────────────────────────────────────────

    @app.route('/quiz/<int:session_id>')
    def take_quiz(session_id):
        quiz_session = QuizSession.query.get_or_404(session_id)
        questions = quiz_session.questions.order_by(Question.order_index).all()
        return render_template('quiz.html', session=quiz_session, questions=questions)

    @app.route('/quiz/<int:session_id>/submit', methods=['POST'])
    def submit_quiz(session_id):
        quiz_session = QuizSession.query.get_or_404(session_id)
        questions = quiz_session.questions.order_by(Question.order_index).all()

        total = 0.0
        for q in questions:
            user_answer = request.form.get(f'answer_{q.id}', '').strip()
            try:
                result = rag.score_answer(
                    q.question_text, q.ideal_answer, user_answer,
                    app.config['GEMINI_MODEL']
                )
            except Exception as exc:
                result = {'score': 0, 'missing_points': [f'Scoring error: {exc}'], 'feedback': ''}

            answer = Answer(
                question_id=q.id,
                user_answer=user_answer,
                score=result['score'],
                max_score=10.0,
                missing_points='\n'.join(result['missing_points']),
                feedback=result['feedback'],
            )
            db.session.add(answer)
            total += result['score']

        quiz_session.total_score = total
        from datetime import datetime, timezone
        quiz_session.completed_at = datetime.now(timezone.utc)
        db.session.commit()

        return redirect(url_for('results', session_id=session_id))

    # ─── Results ─────────────────────────────────────────────────────────

    @app.route('/results/<int:session_id>')
    def results(session_id):
        quiz_session = QuizSession.query.get_or_404(session_id)
        questions = quiz_session.questions.order_by(Question.order_index).all()
        return render_template('results.html', session=quiz_session, questions=questions)

    # ─── Dashboard ───────────────────────────────────────────────────────

    @app.route('/dashboard')
    def dashboard():
        sessions = QuizSession.query.filter(
            QuizSession.completed_at.isnot(None)
        ).order_by(QuizSession.completed_at.asc()).all()

        chart_labels = [s.completed_at.strftime('%d %b %H:%M') for s in sessions]
        chart_scores = [s.percentage for s in sessions]

        return render_template(
            'dashboard.html',
            sessions=list(reversed(sessions)),
            chart_labels=chart_labels,
            chart_scores=chart_scores,
        )

    @app.errorhandler(404)
    def not_found(e):
        return render_template('404.html'), 404


app = create_app()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
