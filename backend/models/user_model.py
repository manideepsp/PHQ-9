from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class User(db.Model):
    """User model mapping to the existing 'user' table in PostgreSQL"""
    
    __tablename__ = 'user'
    __table_args__ = {'schema': 'public'}
    
    # Primary key
    user_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    
    # User details
    emailid = db.Column(db.String(255), unique=True, nullable=False)
    username = db.Column(db.String(100), unique=True, nullable=False)
    firstname = db.Column(db.String(100), nullable=False)
    lastname = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(50), nullable=False)
    industry = db.Column(db.String(100), nullable=False)
    profession = db.Column(db.String(100), nullable=False)
    
    # Security
    password_hash = db.Column(db.String(255), nullable=False)
    
    # Timestamps
    created_at = db.Column(db.DateTime(timezone=True), default=datetime.utcnow)
    
    # Role
    role = db.Column(db.String(255))
    
    def __repr__(self):
        return f'<User {self.username}>'
    
    def to_dict(self):
        """Convert user object to dictionary for JSON serialization"""
        return {
            'user_id': self.user_id,
            'emailid': self.emailid,
            'username': self.username,
            'firstname': self.firstname,
            'lastname': self.lastname,
            'age': self.age,
            'gender': self.gender,
            'industry': self.industry,
            'profession': self.profession,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'role': self.role
        }
