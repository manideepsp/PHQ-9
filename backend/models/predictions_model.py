from .user_model import db
from datetime import datetime



class Prediction(db.Model):
    """Model mapping to existing 'predictions' table in PostgreSQL."""

    __tablename__ = 'predictions'
    __table_args__ = {'schema': 'public'}

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, nullable=False)
    phq9_total_score = db.Column(db.Integer)
    relapse = db.Column(db.Float)
    dsm5_mdd_assessment_enc = db.Column(db.Float)
    consultation_seq = db.Column(db.Integer)
    phq9_assessment_id = db.Column(db.Integer)
    is_predicted = db.Column(db.Boolean, nullable=False, default=False)
    created_at = db.Column(db.DateTime(timezone=True), default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'phq9_total_score': self.phq9_total_score,
            'relapse': self.relapse,
            'dsm5_mdd_assessment_enc': self.dsm5_mdd_assessment_enc,
            'consultation_seq': self.consultation_seq,
            'phq9_assessment_id': self.phq9_assessment_id,
            'is_predicted': self.is_predicted,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


