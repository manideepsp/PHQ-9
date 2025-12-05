from flask import Flask
from flask_cors import CORS
from models.user_model import db
from routes.user_routes import user_bp
from routes.phq9_routes import phq9_bp
from config import Config


def create_app():
    """Application factory pattern"""
    
    # Create Flask app
    app = Flask(__name__)
    
    # Load configuration
    app.config.from_object(Config)
    
    # Initialize SQLAlchemy with app
    db.init_app(app)
    
    # Enable CORS for all domains (configure properly for production)
    CORS(app, origins=app.config['CORS_ORIGINS'])
    
    # Register blueprints
    app.register_blueprint(user_bp, url_prefix='/api')
    app.register_blueprint(phq9_bp, url_prefix='/api')
    
    # Create database tables (if they don't exist)
    with app.app_context():
        try:
            # This will create tables if they don't exist
            # Note: The tables already exist in PostgreSQL, so this is safe
            db.create_all()
            print("Database tables verified/created successfully")
        except Exception as e:
            print(f"Database connection error: {e}")
    
    return app

# Create the app instance
app = create_app()

@app.route('/')
def health_check():
    """Health check endpoint"""
    return {
        "message": "PHQ9-DSM5 Backend API is running",
        "status": "Success",
        "data": {
            "version": "1.0.0",
            "endpoints": ["/api/register", "/api/login", "/api/phq9/check-submission", "/api/phq9/train-model"]
        }
    }

@app.route('/health')
def health():
    """Health endpoint for monitoring"""
    return {
        "status": "healthy",
        "service": "phq9-dsm5-backend"
    }


if __name__ == "__main__":
    import sys
    
    print("Starting PHQ9-DSM5 Backend API...")
    print(f"Database URI: {app.config['SQLALCHEMY_DATABASE_URI']}")
    print(f"Debug mode: {app.config['DEBUG']}")
    print("Available endpoints:")
    print("  POST /api/register - User registration")
    print("  POST /api/login - User login")
    print("  GET / - Health check")
    print("  GET /health - Health status")
    
    # Check if running under debugger (VS Code, PyCharm, etc.)
    is_debugger = hasattr(sys, 'gettrace') and sys.gettrace() is not None
    
    if is_debugger:
        print("🐛 Running under debugger - Flask debug mode disabled to avoid conflicts")
        app.run(
            host=app.config['HOST'],
            port=app.config['PORT'],
            debug=False,  # Disable Flask's debug mode when running under debugger
            use_reloader=False  # Disable auto-reload when debugging
        )
    else:
        print("🚀 Running normally - Flask debug mode enabled")
        app.run(
            host=app.config['HOST'],
            port=app.config['PORT'],
            debug=app.config['DEBUG']
        )
