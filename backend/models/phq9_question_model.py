from .user_model import db


class Phq9Question(db.Model):
    """Model mapping to existing 'phq9_questions' table in PostgreSQL."""

    __tablename__ = 'phq9_questions'
    __table_args__ = {'schema': 'public'}

    question_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    question = db.Column(db.Text, nullable=False)

    def to_dict(self):
        return {
            'question_id': self.question_id,
            'question': self.question,
        }


