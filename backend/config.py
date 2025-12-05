import os

class Config:
    """Configuration class for Flask application"""
    
    # Database configuration
    SQLALCHEMY_DATABASE_URI = "postgresql://admin:admin123@192.168.1.81:5432/phq9_poc"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Flask configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    
    # CORS configuration
    CORS_ORIGINS = "*"  # Configure this properly for production
    
    # Qdrant configuration
    QDRANT_HOST = os.environ.get('QDRANT_HOST', 'http://192.168.1.81')
    QDRANT_API_KEY = os.environ.get('QDRANT_API_KEY', 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.8XMswPe58h0Yor83QOPtTQ9-r3aR5Y0wWVKgKO9vbg0')
    QDRANT_PORT = int(os.environ.get('QDRANT_PORT', '6333'))
    QDRANT_COLLECTION = os.environ.get('QDRANT_COLLECTION', 'pdf_bge_m3')
    
    # Model configuration
    EMBEDDING_MODEL = os.environ.get('EMBEDDING_MODEL', 'BAAI/bge-m3')
    LLM_MODEL = os.environ.get('LLM_MODEL', 'llama3:latest')
    LLM_HOST = os.environ.get('LLM_HOST', 'http://host.docker.internal:11434')
    TOP_K = int(os.environ.get('TOP_K', '5'))
    
    # Application settings
    DEBUG = True
    HOST = "0.0.0.0"
    PORT = 5050

    # === Dynamic ML paths ===
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    ML_ROOT = os.path.join(PROJECT_ROOT, 'backend', 'ML')

    @staticmethod
    def ml_path(*parts: str) -> str:
        """Build a path under backend/ML from project root, ensuring parent dirs exist on demand."""
        return os.path.join(Config.ML_ROOT, *parts)

    @staticmethod
    def ensure_dir(path: str) -> None:
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def ensure_parent(path: str) -> None:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)