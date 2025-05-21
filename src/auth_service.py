import logging
from typing import Optional
from uuid import UUID

from db_manager import DatabaseManager
from schema.schema import UserInDB
import security

logger = logging.getLogger(__name__)

def authenticate_user(db: DatabaseManager, email: str, password: str) -> Optional[UserInDB]:
    """Authenticates a user by email and password."""
    user = db.get_user_by_email(email)
    if not user:
        return None
    if not security.verify_password(password, user.hashed_password):
        return None
    return user

def send_password_email_placeholder(email_to: str, new_password: str):
    """Placeholder for sending password email. Logs to console."""
    logger.warning(f"EMAIL_PLACEHOLDER: For user '{email_to}', new password is: '{new_password}'")
    print(f"--- DEV ONLY: Email to {email_to} ---")
    print(f"Subject: Your New Account Password")
    print(f"Body: Your temporary password is: {new_password}")
    print(f"--- END DEV ONLY ---")

def register_new_user(db: DatabaseManager, email: str) -> tuple[Optional[UserInDB], Optional[str]]:
    """
    Registers a new user if email is @asnr.fr and not already registered.
    Generates a password and calls the email placeholder.
    Returns (UserInDB object or None, plain_password or None for display/logging if needed temporarily)
    """
    if not email.lower().endswith("@asnr.fr"):
        logger.error(f"Registration attempt with non-ASNR email: {email}")
        return None, None

    if db.get_user_by_email(email):
        logger.warning(f"Attempt to register already existing email: {email}")
        return None, None

    plain_password = security.generate_secure_password()
    hashed_password = security.hash_password(plain_password)

    try:
        created_user = db.create_user(email=email, hashed_password=hashed_password)
        if created_user:
            send_password_email_placeholder(email, plain_password)
            return created_user, plain_password
        else:
            logger.error(f"User creation returned None for {email}")
            return None, None
    except Exception as e:
        logger.error(f"Error during user creation for {email}: {e}")
        return None, None
