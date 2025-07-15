import logging
import os
from typing import Optional
from uuid import UUID

from db_manager import DatabaseManager
from schema.schema import UserInDB
import security

from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from display_texts import dt

logger = logging.getLogger(__name__)

def authenticate_user(db: DatabaseManager, email: str, password: str) -> Optional[UserInDB]:
    """Authenticates a user by email and password."""
    user = db.get_user_by_email(email)
    if not user:
        return None
    if not security.verify_password(password, user.hashed_password):
        return None
    return user

def send_password_email_with_sendgrid(email_to: str, new_password: str) -> bool:
    """Sends a password email using SendGrid."""
    sendgrid_api_key = os.environ.get("SENDGRID_API_KEY")
    from_email_address = os.environ.get("SENDGRID_FROM_EMAIL")

    if not sendgrid_api_key:
        logger.critical("SENDGRID_API_KEY environment variable not set. Cannot send email.")
        return False
    if not from_email_address:
        logger.critical("SENDGRID_FROM_EMAIL environment variable not set. Cannot send email.")
        return False

    message = Mail(
        from_email=from_email_address,
        to_emails=email_to,
        subject=dt.PASSWORD_EMAIL_SUBJECT,
        html_content=dt.PASSWORD_EMAIL_BODY.format(new_password=new_password)
    )
    print('Email send with', from_email_address, ' to ', email_to)
    try:
        sg = SendGridAPIClient(sendgrid_api_key)
        response = sg.send(message)
        if response.status_code >= 200 and response.status_code < 300:
            logger.info(f"Password email sent successfully to {email_to} via SendGrid. Status: {response.status_code}")
            return True
        else:
            logger.error(f"Failed to send password email to {email_to} via SendGrid. Status: {response.status_code}, Body: {response.body}")
            return False
    except Exception as e:
        logger.error(f"Error sending password email to {email_to} via SendGrid: {e}")
        return False

def register_new_user(db: DatabaseManager, email: str) -> tuple[Optional[UserInDB], Optional[str]]:
    """
    Registers a new user if email has a valid domain and does not already exist.
    Generates a password and calls the email placeholder.
    Returns (UserInDB object or None, plain_password or None for display/logging if needed temporarily)
    """
    if not email.lower().endswith(dt.EMAIL_DOMAIN):
        logger.error(f"Registration attempt with invalid email domain: {email}")
        return None, None

    if db.get_user_by_email(email):
        logger.warning(f"Attempt to register already existing email: {email}")
        return None, None

    plain_password = security.generate_secure_password()
    hashed_password = security.hash_password(plain_password)

    try:
        created_user = db.create_user(email=email, hashed_password=hashed_password)
        if created_user:
            if send_password_email_with_sendgrid(email, plain_password):
                logger.info(f"Successfully registered and sent password to {email}")
            else:
                logger.error(f"User {email} registered, but FAILED to send password email via SendGrid.")
            return created_user, plain_password
        else:
            logger.error(f"User creation returned None for {email}")
            return None, None
    except Exception as e:
        logger.error(f"Error during user creation for {email}: {e}")
        return None, None

def reset_user_password(db: DatabaseManager, email: str) -> tuple[bool, str]:
    """
    Resets the password for an existing user, updates it in the database, and sends the new password via email.
    Returns (success, message) for UI feedback.
    """
    user = db.get_user_by_email(email)
    if not user:
        return False, getattr(dt, "RESET_PASSWORD_USER_NOT_FOUND", "User not found.")
    plain_password = security.generate_secure_password()
    hashed_password = security.hash_password(plain_password)
    updated = db.update_user_password(email, hashed_password)
    if not updated:
        return False, getattr(dt, "RESET_PASSWORD_UPDATE_FAILED", "Failed to update password.")
    email_sent = send_password_email_with_sendgrid(email, plain_password)
    if email_sent:
        return True, dt.RESET_PASSWORD_EMAIL_SENT.format(email=email)
    else:
        # If email fails, show the password directly (not ideal for production, but fallback for demo/dev)
        # return True, getattr(dt, "RESET_PASSWORD_EMAIL_FAILED", "Email failed. Your new password is: {password}").format(password=plain_password)
        pass
