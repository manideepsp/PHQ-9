from flask import Blueprint, request, jsonify
from services.user_service import UserService

# Create blueprint for user routes
user_bp = Blueprint('user', __name__)

@user_bp.route('/register', methods=['POST'])
def register():
    """
    User registration endpoint
    
    Expected JSON payload:
    {
        "emailid": "john@example.com",
        "username": "john_doe",
        "firstname": "John",
        "lastname": "Doe",
        "age": 28,
        "gender": "Male",
        "industry": "Tech",
        "profession": "Developer",
        "password": "password123",
        "role": "user"
    }
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({
                "message": "No JSON data provided",
                "status": "Failed",
                "data": None
            }), 400
        
        # Call service to register user
        result = UserService.register_user(data)
        
        # Return appropriate status code based on result
        if result["status"] == "Success":
            return jsonify(result), 201
        else:
            return jsonify(result), 400
            
    except Exception as e:
        return jsonify({
            "message": f"Registration endpoint error: {str(e)}",
            "status": "Failed",
            "data": None
        }), 500

@user_bp.route('/login', methods=['POST'])
def login():
    """
    User login endpoint
    
    Expected JSON payload:
    {
        "username": "john_doe",
        "password": "password123"
    }
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({
                "message": "No JSON data provided",
                "status": "Failed",
                "data": None
            }), 400
        
        # Validate required fields
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({
                "message": "Username and password are required",
                "status": "Failed",
                "data": None
            }), 400
        
        # Call service to authenticate user
        result = UserService.login_user(username, password)
        
        # Return appropriate status code based on result
        if result["status"] == "Success":
            return jsonify(result), 200
        else:
            return jsonify(result), 401
            
    except Exception as e:
        return jsonify({
            "message": f"Login endpoint error: {str(e)}",
            "status": "Failed",
            "data": None
        }), 500
