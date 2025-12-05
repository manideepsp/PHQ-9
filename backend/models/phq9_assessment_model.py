from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, date

from .user_model import db  # reuse the same SQLAlchemy instance


class Phq9Assessment(db.Model):
    """Model mapping to existing 'phq9_assessment' table in PostgreSQL."""

    __tablename__ = 'phq9_assessment'
    __table_args__ = {'schema': 'public'}

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, nullable=False)
    responses = db.Column(db.JSON)
    total_score = db.Column(db.Integer)
    submitted_at = db.Column(db.DateTime, default=datetime.utcnow)
    doctors_notes = db.Column(db.Text)
    patients_notes = db.Column(db.Text)

    def __repr__(self):
        return f'<Phq9Assessment id={self.id} user_id={self.user_id} date={self.assessment_date}>'


