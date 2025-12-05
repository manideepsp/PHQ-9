from .user_model import db
from datetime import datetime

class DSM5Assessment(db.Model):
    """Model mapping to existing 'dsm_5_assessment' table in PostgreSQL."""

    __tablename__ = 'dsm_5_assessment'
    __table_args__ = {'schema': 'public'}

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, nullable=False)
    severity = db.Column(db.String(50), nullable=False)
    q9_flag = db.Column(db.Boolean, nullable=False)
    mdd_assessment = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), default=datetime.utcnow)

    def __repr__(self):
        return f'<DSM5Assessment id={self.id} user_id={self.user_id} severity={self.severity}>'

    def to_dict(self):
        """Convert DSM5Assessment object to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'severity': self.severity,
            'q9_flag': self.q9_flag,
            'mdd_assessment': self.mdd_assessment,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
