# PHQ9-DSM5 Backend API

A Flask backend application for the PHQ9-DSM5 mental health assessment system.

## Features

- User Registration API
- User Login API
- PostgreSQL database integration
- Secure password hashing with bcrypt
- CORS enabled for frontend integration

## Setup Instructions

### Prerequisites

- Python 3.8+
- PostgreSQL database running on `localhost:5433`
- Database name: `mydb`
- Username: `postgres`
- Password: `admin123`
- ML model training - hit `/phq9/train-model` api

### Installation

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. Make sure PostgreSQL is running and the database is accessible
2. Start the Flask application:
   ```bash
   python app.py
   ```

The API will be available at `http://localhost:5000`

## API Endpoints

## ML Model training
- **GET** `/phq9/train-model` - Train ML Model

### Health Check
- **GET** `/` - API health check
- **GET** `/health` - Health status for monitoring

### User Authentication
- **POST** `/api/register` - User registration
- **POST** `/api/login` - User login

## API Usage Examples

### User Registration

**Endpoint:** `POST /api/register`

**Request Body:**
```json
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
```

**Success Response:**
```json
{
  "message": "User registered successfully",
  "status": "Success",
  "data": {
    "user_id": 1,
    "username": "john_doe"
  }
}
```

### User Login

**Endpoint:** `POST /api/login`

**Request Body:**
```json
{
  "username": "john_doe",
  "password": "password123"
}
```

**Success Response:**
```json
{
  "message": "Login successful",
  "status": "Success",
  "data": {
    "role": "user"
  }
}
```

## Database Schema

The application connects to an existing PostgreSQL database with the following tables:

- `user` - User information and authentication
- `phq9_questions` - PHQ9 assessment questions
- `phq9_assessment` - User PHQ9 assessment results
- `dsm_5_assessment` - DSM-5 assessment results

## Project Structure

```
backend/
├── app.py                 # Main Flask application
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── models/
│   ├── __init__.py
│   └── user_model.py      # User SQLAlchemy model
├── routes/
│   ├── __init__.py
│   └── user_routes.py     # API route definitions
├── services/
│   ├── __init__.py
│   └── user_service.py    # Business logic
└── utils/
    ├── __init__.py
    └── security.py        # Password hashing utilities
```

## Configuration

Database connection and other settings can be modified in `config.py`:

```python
SQLALCHEMY_DATABASE_URI = "postgresql://postgres:admin123@localhost:5433/mydb"
```

## Error Handling

All API responses follow a consistent format:

```json
{
  "message": "Error description",
  "status": "Failed",
  "data": null
}
```

## Development Notes

- Passwords are securely hashed using bcrypt
- CORS is enabled for frontend integration
- Database tables are automatically created if they don't exist
- The application uses SQLAlchemy ORM for database operations
