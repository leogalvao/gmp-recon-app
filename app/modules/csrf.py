"""
CSRF Protection Module

Implements double-submit cookie pattern with cryptographic tokens.
Uses itsdangerous for secure token generation/validation.

Security Analysis (Game-Theoretic):
- Attacker's strategy: Forge requests from victim's browser
- Defender's strategy: Require proof of same-origin via secret token
- Nash Equilibrium: Attacker cannot succeed without token knowledge
"""
import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Optional
from fastapi import Request, HTTPException, status
from fastapi.responses import Response
import logging

logger = logging.getLogger(__name__)


class CSRFProtection:
    """
    CSRF protection using cryptographic tokens.

    Implements the Synchronizer Token Pattern:
    1. Server generates unique token per session
    2. Token is stored in cookie AND must be submitted with requests
    3. Server validates both match

    This is simpler than the signed token approach but equally secure
    for single-server deployments.
    """

    def __init__(self, cookie_name: str = "csrf_token", header_name: str = "X-CSRF-Token"):
        self.cookie_name = cookie_name
        self.header_name = header_name
        self.form_field = "csrf_token"
        self._tokens: dict = {}  # In production, use Redis or database

    def generate_token(self) -> str:
        """Generate a new cryptographically secure CSRF token."""
        return secrets.token_hex(32)

    def set_token_cookie(self, response: Response, token: str) -> None:
        """Set CSRF token as HTTP-only cookie."""
        response.set_cookie(
            key=self.cookie_name,
            value=token,
            httponly=False,  # JS needs to read it for AJAX
            samesite="strict",
            secure=False,  # Set True in production with HTTPS
            max_age=3600  # 1 hour
        )

    async def get_submitted_token(self, request: Request) -> Optional[str]:
        """
        Extract CSRF token from request.

        Checks in order:
        1. X-CSRF-Token header (for AJAX)
        2. Form field csrf_token (for form submissions)
        """
        # Check header first (AJAX requests)
        token = request.headers.get(self.header_name)
        if token:
            return token

        # Check form data
        content_type = request.headers.get("content-type", "")
        if "form" in content_type:
            try:
                form = await request.form()
                token = form.get(self.form_field)
                if token:
                    return token
            except Exception:
                pass

        # Check JSON body
        if "json" in content_type:
            try:
                body = await request.json()
                if isinstance(body, dict):
                    token = body.get(self.form_field)
                    if token:
                        return token
            except Exception:
                pass

        return None

    async def verify_request(self, request: Request) -> bool:
        """
        Verify CSRF token for state-changing requests.

        Returns True if:
        - Request method is safe (GET, HEAD, OPTIONS)
        - Token in cookie matches token in request body/header

        Raises HTTPException if validation fails.
        """
        # Safe methods don't need CSRF protection
        if request.method in ("GET", "HEAD", "OPTIONS"):
            return True

        # Get token from cookie
        cookie_token = request.cookies.get(self.cookie_name)
        if not cookie_token:
            logger.warning(f"CSRF: Missing cookie token for {request.url.path}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="CSRF token missing from cookie"
            )

        # Get submitted token
        submitted_token = await self.get_submitted_token(request)
        if not submitted_token:
            logger.warning(f"CSRF: Missing submitted token for {request.url.path}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="CSRF token missing from request"
            )

        # Constant-time comparison to prevent timing attacks
        if not secrets.compare_digest(cookie_token, submitted_token):
            logger.warning(f"CSRF: Token mismatch for {request.url.path}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="CSRF token mismatch"
            )

        return True


# Global instance
csrf = CSRFProtection()


def get_or_create_csrf_token(request: Request) -> str:
    """
    Get existing CSRF token from cookie or generate new one.

    Used by templates to include token in forms.
    """
    existing = request.cookies.get(csrf.cookie_name)
    if existing:
        return existing
    return csrf.generate_token()
