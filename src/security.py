from passlib.context import CryptContext
import secrets
import string

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain password against a hashed password."""
    return pwd_context.verify(plain_password, hashed_password)

def hash_password(password: str) -> str:
    """Hashes a plain password."""
    return pwd_context.hash(password)

def generate_secure_password(length: int = 16) -> str:
    """Generates a cryptographically strong random password."""
    alphabet = string.ascii_letters
    # Ensure the password contains at least one of each character type if desired (more complex)
    # For simplicity here, just random characters.
    return ''.join(secrets.choice(alphabet) for _ in range(length))
