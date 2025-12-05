import bcrypt

def hash_password(password):
    """
    Hash a password using bcrypt
    
    Args:
        password (str): Plain text password to hash
        
    Returns:
        str: Hashed password string
    """
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(stored_hash, password):
    """
    Verify a password against its stored hash
    
    Args:
        stored_hash (str): The stored password hash from database
        password (str): Plain text password to verify
        
    Returns:
        bool: True if password matches, False otherwise
    """
    return bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8'))
