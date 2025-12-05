from .user_model import db
from datetime import datetime



class Interventions(db.Model):
    """Model mapping to existing 'predictions' table in PostgreSQL."""

    __tablename__ = 'interventions'
    __table_args__ = {'schema': 'public'}

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, nullable=False)
    intervention = db.Column(db.String)
    phq9_assessment_id = db.Column(db.Integer)
    created_at = db.Column(db.DateTime(timezone=True), default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'intervention': self.intervention,
            'phq9_assessment_id': self.phq9_assessment_id,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


