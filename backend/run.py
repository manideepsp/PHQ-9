#!/usr/bin/env python3
"""
Startup script for PHQ9-DSM5 Backend API
This script provides an easy way to start the Flask application
"""

import os
import sys
from app import app

def main():
    """Main startup function"""
    print("🚀 Starting PHQ9-DSM5 Backend API...")
    print("=" * 50)
    
    # Check if we're in the correct directory
    if not os.path.exists('app.py'):
        print("❌ Error: app.py not found. Please run this script from the backend directory.")
        sys.exit(1)
    
    # Display configuration
    print(f"📊 Configuration:")
    print(f"   Host: {app.config.get('HOST', '0.0.0.0')}")
    print(f"   Port: {app.config.get('PORT', 5050)}")
    print(f"   Debug: {app.config.get('DEBUG', False)}")
    print(f"   Database: {app.config.get('SQLALCHEMY_DATABASE_URI', 'Not configured')}")
    print("=" * 50)
    
    # Display available endpoints
    print("🔗 Available Endpoints:")
    print("   GET  /           - Health check")
    print("   GET  /health     - Health status")
    print("   POST /api/register - User registration")
    print("   POST /api/login    - User login")
    print("=" * 50)
    
    try:
        # Start the Flask application
        app.run(
            host=app.config.get('HOST', '0.0.0.0'),
            port=app.config.get('PORT', 5050),
            debug=app.config.get('DEBUG', False)
        )
    except KeyboardInterrupt:
        print("\n👋 Shutting down PHQ9-DSM5 Backend API...")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
