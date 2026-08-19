"""
apps.authentication.views
===========================

Ship's complete authentication API.

Token architecture: access tokens are short-lived, stateless JWTs
(`rest_framework_simplejwt.tokens.AccessToken`), never persisted.
Refresh tokens are long-lived, opaque, high-entropy random strings;
only their SHA-256 digest is ever persisted, in `RefreshToken`
(models.py), which is also what makes rotation, revocation, and
family-based reuse detection possible.

Email delivery (verification / password reset) is deliberately isolated
behind the two small functions in the "Email delivery hooks" section
below rather than inlined — swap their bodies for a real call into
Ship's future notifications/email service; nothing else in this file
needs to change. Neither function ever logs a raw token.
"""

from __future__ import annotations

import logging
import secrets
import uuid
from datetime import datetime, timedelta
from datetime import timezone as dt_timezone
from typing import Any

from django.conf import settings
from django.db import transaction
from django.utils import timezone
from rest_framework import generics, permissions, status
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.throttling import ScopedRateThrottle
from rest_framework.views import APIView
from rest_framework_simplejwt.tokens import AccessToken

from apps.authentication.models import (
    RefreshToken,
    RefreshTokenRevocationReason,
    SecurityToken,
    SecurityTokenPurpose,
    User,
)
from apps.authentication.serializers import (
    EmailResendSerializer,
    EmailVerifySerializer,
    LoginSerializer,
    LogoutSerializer,
    PasswordChangeSerializer,
    PasswordResetConfirmSerializer,
    PasswordResetRequestSerializer,
    RefreshSerializer,
    RegistrationSerializer,
    SessionRevokeSerializer,
    TokenResponseSerializer,
    UserSerializer,
)

logger = logging.getLogger(__name__)

_GENERIC_RESET_REQUEST_MESSAGE = {
    "detail": "If an account exists for that email, a password reset link has been sent."
}
_GENERIC_RESEND_MESSAGE = {
    "detail": "If an account exists for that email and is not yet verified, a verification link has been sent."
}

VERIFICATION_TOKEN_LIFETIME: timedelta = getattr(
    settings, "AUTH_EMAIL_VERIFICATION_TOKEN_LIFETIME", timedelta(hours=24)
)
PASSWORD_RESET_TOKEN_LIFETIME: timedelta = getattr(
    settings, "AUTH_PASSWORD_RESET_TOKEN_LIFETIME", timedelta(hours=1)
)
REFRESH_TOKEN_LIFETIME: timedelta = getattr(
    settings, "AUTH_REFRESH_TOKEN_LIFETIME", timedelta(days=30)
)


# ---------------------------------------------------------------------------
# Email delivery hooks (integration seam — see module docstring)
# ---------------------------------------------------------------------------


def _send_verification_email(user: User, raw_token: str) -> None:
    logger.info("Verification email queued for user_id=%s", user.id)


def _send_password_reset_email(user: User, raw_token: str) -> None:
    logger.info("Password reset email queued for user_id=%s", user.id)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _client_ip(request: Request) -> str | None:
    forwarded = request.META.get("HTTP_X_FORWARDED_FOR")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.META.get("REMOTE_ADDR")


def _issue_token_pair(
    user: User, request: Request, family_id: uuid.UUID | None = None
) -> dict[str, Any]:
    """
    Issues one access token (stateless JWT) and one refresh token
    (persisted only as a hash). `family_id` is carried forward across
    rotation so the whole chain can be revoked together if reuse is
    ever detected (see RefreshSerializer).
    """
    raw_refresh = secrets.token_urlsafe(48)
    now = timezone.now()

    refresh_record = RefreshToken.objects.create(
        user=user,
        token_hash=RefreshToken.hash_token(raw_refresh),
        expires_at=now + REFRESH_TOKEN_LIFETIME,
        user_agent=request.META.get("HTTP_USER_AGENT", "")[:255],
        ip_address=_client_ip(request),
        **({"family_id": family_id} if family_id is not None else {}),
    )

    access_token = AccessToken.for_user(user)

    return {
        "access": str(access_token),
        "access_expires_at": datetime.fromtimestamp(access_token["exp"], tz=dt_timezone.utc),
        "refresh": raw_refresh,
        "refresh_expires_at": refresh_record.expires_at,
        "session_id": refresh_record.id,
    }


def _issue_security_token(user: User, purpose: str, lifetime: timedelta) -> str:
    """Invalidates any previously issued, still-unused token of the same purpose, then issues a fresh one."""
    SecurityToken.objects.filter(user=user, purpose=purpose, used_at__isnull=True).update(
        used_at=timezone.now()
    )
    raw_token = secrets.token_urlsafe(32)
    SecurityToken.objects.create(
        user=user,
        purpose=purpose,
        token_hash=SecurityToken.hash_token(raw_token),
        expires_at=timezone.now() + lifetime,
    )
    return raw_token


def _debug_extra(**tokens: str) -> dict[str, str]:
    """Only ever populated when DEBUG=True — never expose raw tokens in production responses."""
    return {f"debug_{k}": v for k, v in tokens.items()} if settings.DEBUG else {}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class RegisterView(generics.CreateAPIView):
    """POST /register/ — public. Creates a non-privileged account and starts email verification."""

    queryset = User.objects.all()
    serializer_class = RegistrationSerializer
    permission_classes = [permissions.AllowAny]

    def create(self, request: Request, *args, **kwargs) -> Response:
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        with transaction.atomic():
            user = serializer.save()
            raw_token = _issue_security_token(
                user, SecurityTokenPurpose.EMAIL_VERIFICATION, VERIFICATION_TOKEN_LIFETIME
            )

        _send_verification_email(user, raw_token)

        data = UserSerializer(user).data
        data.update(_debug_extra(verification_token=raw_token))
        return Response(data, status=status.HTTP_201_CREATED)


# ---------------------------------------------------------------------------
# Login / logout / refresh
# ---------------------------------------------------------------------------


class LoginView(APIView):
    """POST /login/ — public. Issues an access/refresh token pair on valid, unlocked, active credentials."""

    permission_classes = [permissions.AllowAny]

    def post(self, request: Request) -> Response:
        serializer = LoginSerializer(data=request.data, context={"request": request})
        serializer.is_valid(raise_exception=True)
        user: User = serializer.validated_data["user"]

        with transaction.atomic():
            user.failed_login_attempts = 0
            user.locked_until = None
            user.last_login = timezone.now()
            user.save(update_fields=["failed_login_attempts", "locked_until", "last_login", "updated_at"])
            tokens = _issue_token_pair(user, request)

        return Response(TokenResponseSerializer(tokens).data, status=status.HTTP_200_OK)


class LogoutView(APIView):
    """
    POST /logout/ — authenticated. Revokes the caller's own refresh
    token so it can never be used to obtain a new access token again.
    The access token the client is currently holding remains valid
    (by design — it is stateless and short-lived) until it naturally
    expires; clients should discard it locally on logout.
    """

    permission_classes = [permissions.IsAuthenticated]

    def post(self, request: Request) -> Response:
        serializer = LogoutSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        token_hash = RefreshToken.hash_token(serializer.validated_data["refresh"])
        record = RefreshToken.objects.filter(
            token_hash=token_hash, user=request.user, revoked_at__isnull=True
        ).first()
        if record is not None:
            record.revoke(RefreshTokenRevocationReason.LOGOUT)

        return Response(status=status.HTTP_204_NO_CONTENT)


class RefreshView(APIView):
    """
    POST /refresh/ — public (the whole point is to work without a
    valid access token). Rotates the presented refresh token: revokes
    it and issues a new access/refresh pair sharing the same
    `family_id`, so reuse of the old one is detectable.
    """

    permission_classes = [permissions.AllowAny]

    def post(self, request: Request) -> Response:
        serializer = RefreshSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        old_record: RefreshToken = serializer.validated_data["refresh_record"]

        with transaction.atomic():
            old_record.revoke(RefreshTokenRevocationReason.ROTATED)
            tokens = _issue_token_pair(old_record.user, request, family_id=old_record.family_id)

        return Response(TokenResponseSerializer(tokens).data, status=status.HTTP_200_OK)


# ---------------------------------------------------------------------------
# Current user / profile
# ---------------------------------------------------------------------------


class MeView(generics.RetrieveUpdateAPIView):
    """GET/PATCH /me/ — authenticated. Read or partially update the caller's own profile."""

    serializer_class = UserSerializer
    permission_classes = [permissions.IsAuthenticated]
    http_method_names = ["get", "patch", "head", "options"]

    def get_object(self) -> User:
        return self.request.user


# ---------------------------------------------------------------------------
# Password change / reset
# ---------------------------------------------------------------------------


class PasswordChangeView(APIView):
    """
    POST /password/change/ — authenticated. Verifies the current
    password, applies the new one via `set_password()`, and revokes
    every other active session as a standard "changing your password
    logs out other devices" security behavior.
    """

    permission_classes = [permissions.IsAuthenticated]

    def post(self, request: Request) -> Response:
        serializer = PasswordChangeSerializer(data=request.data, context={"request": request})
        serializer.is_valid(raise_exception=True)

        user = request.user
        with transaction.atomic():
            user.set_password(serializer.validated_data["new_password"])
            user.save()
            RefreshToken.objects.filter(user=user, revoked_at__isnull=True).update(
                revoked_at=timezone.now(),
                revoked_reason=RefreshTokenRevocationReason.PASSWORD_CHANGE,
            )

        return Response({"detail": "Password changed successfully."}, status=status.HTTP_200_OK)


class PasswordResetRequestView(APIView):
    """
    POST /password/reset/request/ — public. Always returns the same
    generic response regardless of whether the email matches an
    account, to avoid account enumeration.
    """

    permission_classes = [permissions.AllowAny]
    throttle_classes = [ScopedRateThrottle]
    throttle_scope = "password-reset"

    def post(self, request: Request) -> Response:
        serializer = PasswordResetRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        user = User.objects.filter(
            email__iexact=serializer.validated_data["email"], is_active=True
        ).first()
        extra: dict[str, str] = {}
        if user is not None:
            with transaction.atomic():
                raw_token = _issue_security_token(
                    user, SecurityTokenPurpose.PASSWORD_RESET, PASSWORD_RESET_TOKEN_LIFETIME
                )
            _send_password_reset_email(user, raw_token)
            extra = _debug_extra(reset_token=raw_token)

        return Response({**_GENERIC_RESET_REQUEST_MESSAGE, **extra}, status=status.HTTP_200_OK)


class PasswordResetConfirmView(APIView):
    """
    POST /password/reset/confirm/ — public. Consumes a single-use
    reset token, sets the new password, and revokes every active
    session (the account may have just been compromised, so any
    session issued before the reset should not be trusted going
    forward).
    """

    permission_classes = [permissions.AllowAny]

    def post(self, request: Request) -> Response:
        serializer = PasswordResetConfirmSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        token: SecurityToken = serializer.validated_data["security_token"]
        user = token.user
        with transaction.atomic():
            token.mark_used()
            user.set_password(serializer.validated_data["new_password"])
            user.save()
            RefreshToken.objects.filter(user=user, revoked_at__isnull=True).update(
                revoked_at=timezone.now(),
                revoked_reason=RefreshTokenRevocationReason.PASSWORD_CHANGE,
            )

        return Response({"detail": "Password reset successfully."}, status=status.HTTP_200_OK)


# ---------------------------------------------------------------------------
# Email verification
# ---------------------------------------------------------------------------


class EmailVerifyView(APIView):
    """POST /email/verify/ — public (the token itself proves identity). Consumes a single-use verification token."""

    permission_classes = [permissions.AllowAny]

    def post(self, request: Request) -> Response:
        serializer = EmailVerifySerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        token: SecurityToken = serializer.validated_data["security_token"]
        with transaction.atomic():
            token.mark_used()
            token.user.mark_email_verified()

        return Response({"detail": "Email verified successfully."}, status=status.HTTP_200_OK)


class EmailResendView(APIView):
    """
    POST /email/resend/ — public. Always returns the same generic
    response regardless of whether the email matches an account or is
    already verified, to avoid account enumeration.
    """

    permission_classes = [permissions.AllowAny]
    throttle_classes = [ScopedRateThrottle]
    throttle_scope = "email-verification-resend"

    def post(self, request: Request) -> Response:
        serializer = EmailResendSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        user = User.objects.filter(
            email__iexact=serializer.validated_data["email"], is_active=True
        ).first()
        extra: dict[str, str] = {}
        if user is not None and not user.is_email_verified:
            raw_token = _issue_security_token(
                user, SecurityTokenPurpose.EMAIL_VERIFICATION, VERIFICATION_TOKEN_LIFETIME
            )
            _send_verification_email(user, raw_token)
            extra = _debug_extra(verification_token=raw_token)

        return Response({**_GENERIC_RESEND_MESSAGE, **extra}, status=status.HTTP_200_OK)


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------


class SessionRevokeView(APIView):
    """
    POST /sessions/revoke/ — authenticated. Revokes one of the caller's
    own sessions by `session_id` (RefreshToken.id). Always responds
    204 whether or not a matching active session was found, so a
    caller cannot use this endpoint to probe another user's session
    ids.
    """

    permission_classes = [permissions.IsAuthenticated]

    def post(self, request: Request) -> Response:
        serializer = SessionRevokeSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        record = RefreshToken.objects.filter(
            pk=serializer.validated_data["session_id"],
            user=request.user,
            revoked_at__isnull=True,
        ).first()
        if record is not None:
            record.revoke(RefreshTokenRevocationReason.USER_REVOKED)

        return Response(status=status.HTTP_204_NO_CONTENT)


class SessionRevokeAllView(APIView):
    """POST /sessions/revoke-all/ — authenticated. Revokes every active session belonging to the caller."""

    permission_classes = [permissions.IsAuthenticated]

    def post(self, request: Request) -> Response:
        RefreshToken.objects.filter(user=request.user, revoked_at__isnull=True).update(
            revoked_at=timezone.now(),
            revoked_reason=RefreshTokenRevocationReason.USER_REVOKED,
        )
        return Response(status=status.HTTP_204_NO_CONTENT)