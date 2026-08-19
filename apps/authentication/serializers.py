"""
apps.authentication.serializers
=================================

DRF serializers for Ship's complete authentication API.

No serializer here can read or write `is_staff`, `is_superuser`,
`groups`, `user_permissions`, `email_verified_at` (as input),
`failed_login_attempts`, `locked_until`, `password_changed_at`, or any
token/hash field — those are simply never listed in any `fields`, so
DRF cannot touch them regardless of request payload.

Password hashing is always delegated to `User.set_password()` (which
also keeps `password_changed_at` in sync — see models.py). No
serializer here reimplements hashing, and Django's configured
`AUTH_PASSWORD_VALIDATORS` are always run via
`django.contrib.auth.password_validation.validate_password`.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from django.conf import settings
from django.contrib.auth import authenticate
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError as DjangoValidationError
from django.utils import timezone
from rest_framework import serializers

from apps.authentication.models import (
    RefreshToken,
    RefreshTokenRevocationReason,
    SecurityToken,
    SecurityTokenPurpose,
    User,
)

# Policy constants for login lockout. These are business policy, not
# persistence concerns — `User.failed_login_attempts`/`locked_until` are
# policy-free counters (see models.py); the threshold/duration applied
# to them lives here until a dedicated service layer owns it.
# LOGIN_LOCKOUT_THRESHOLD: int = getattr(settings, "AUTH_LOGIN_LOCKOUT_THRESHOLD", 10)
# LOGIN_LOCKOUT_DURATION: timedelta = getattr(
#     settings, "AUTH_LOGIN_LOCKOUT_DURATION", timedelta(minutes=15)
# )
DEFAULT_LOGIN_LOCKOUT_THRESHOLD = 10
DEFAULT_LOGIN_LOCKOUT_DURATION = timedelta(minutes=15)


_GENERIC_LOGIN_ERROR = "Unable to log in with the provided credentials."
_GENERIC_TOKEN_ERROR = "Invalid or expired token."


def _resolve_security_token(raw_token: str, purpose: str) -> SecurityToken:
    """
    Shared by email-verification and password-reset confirmation so the
    lookup/validity logic (and its error message) exists in exactly one
    place. Deliberately returns the same generic error whether the
    token doesn't exist, was already used, or has expired — telling
    those apart would leak information to an attacker probing tokens.
    """
    token_hash = SecurityToken.hash_token(raw_token)
    try:
        token = SecurityToken.objects.select_related("user").get(
            token_hash=token_hash, purpose=purpose
        )
    except SecurityToken.DoesNotExist:
        raise serializers.ValidationError({"token": _GENERIC_TOKEN_ERROR})
    if not token.is_valid:
        raise serializers.ValidationError({"token": _GENERIC_TOKEN_ERROR})
    return token


def _register_failed_login(email: str) -> None:
    """
    Best-effort lockout bookkeeping for a failed login attempt. Silently
    no-ops if the email doesn't match a real account — this must never
    create or reveal anything about an account that doesn't exist.
    """
    user = User.objects.filter(email__iexact=email).first()
    if user is None:
        return
    user.failed_login_attempts += 1
    # if user.failed_login_attempts >= LOGIN_LOCKOUT_THRESHOLD:
    #     user.locked_until = timezone.now() + LOGIN_LOCKOUT_DURATION
    lockout_threshold = getattr(
        settings,
        "AUTH_LOGIN_LOCKOUT_THRESHOLD",
        DEFAULT_LOGIN_LOCKOUT_THRESHOLD,
    )

    lockout_duration = getattr(
        settings,
        "AUTH_LOGIN_LOCKOUT_DURATION",
        DEFAULT_LOGIN_LOCKOUT_DURATION,
    )

    if user.failed_login_attempts >= lockout_threshold:
        user.locked_until = timezone.now() + lockout_duration
    user.save(update_fields=["failed_login_attempts", "locked_until", "updated_at"])


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class RegistrationSerializer(serializers.ModelSerializer):
    """
    Creates a new, non-privileged account. Password handling is
    delegated entirely to `User.objects.create_user` — this serializer
    only validates input.
    """

    password = serializers.CharField(
        write_only=True, style={"input_type": "password"}, trim_whitespace=False
    )
    password_confirm = serializers.CharField(
        write_only=True, style={"input_type": "password"}, trim_whitespace=False
    )

    class Meta:
        model = User
        fields = ["id", "email", "password", "password_confirm", "first_name", "last_name"]
        read_only_fields = ["id"]

    def validate_email(self, value: str) -> str:
        normalized = User.objects.normalize_email(value)
        if User.objects.filter(email__iexact=normalized).exists():
            raise serializers.ValidationError("A user with this email already exists.")
        return normalized

    def validate(self, attrs: dict[str, Any]) -> dict[str, Any]:
        if attrs["password"] != attrs["password_confirm"]:
            raise serializers.ValidationError({"password_confirm": "Passwords do not match."})

        candidate = User(
            email=attrs.get("email", ""),
            first_name=attrs.get("first_name", ""),
            last_name=attrs.get("last_name", ""),
        )
        try:
            validate_password(attrs["password"], user=candidate)
        except DjangoValidationError as exc:
            raise serializers.ValidationError({"password": list(exc.messages)})

        return attrs

    def create(self, validated_data: dict[str, Any]) -> User:
        validated_data.pop("password_confirm")
        password = validated_data.pop("password")
        return User.objects.create_user(password=password, **validated_data)


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------


class LoginSerializer(serializers.Serializer):
    """
    Authenticates via Django's authentication framework
    (`django.contrib.auth.authenticate`), which already: normalizes the
    case-insensitive lookup (via `UserManager.get_by_natural_key`),
    respects `is_active` (`ModelBackend.user_can_authenticate`), and
    performs a dummy password hash when the email doesn't match anyone
    — so failure for "wrong password" and "no such account" take
    comparable time and return an identical, generic error.

    `is_locked` is checked *after* a successful credential match, so an
    account-lock disclosure only ever reaches someone who already
    proved they know the correct password — it cannot be used to probe
    which emails exist.
    """

    email = serializers.EmailField()
    password = serializers.CharField(write_only=True, trim_whitespace=False)

    def validate(self, attrs: dict[str, Any]) -> dict[str, Any]:
        request = self.context.get("request")
        user = authenticate(request=request, email=attrs["email"], password=attrs["password"])

        if user is None:
            _register_failed_login(attrs["email"])
            raise serializers.ValidationError(_GENERIC_LOGIN_ERROR)

        if user.is_locked:
            raise serializers.ValidationError(
                "This account is temporarily locked due to repeated failed login attempts."
            )

        attrs["user"] = user
        return attrs


# ---------------------------------------------------------------------------
# Token responses
# ---------------------------------------------------------------------------


class TokenResponseSerializer(serializers.Serializer):
    """
    Safe output shape for a freshly issued (or rotated) token pair.
    `access` is a short-lived, stateless JWT. `refresh` is the ONE
    moment the raw refresh token is ever available — only its SHA-256
    digest is persisted (`RefreshToken.token_hash`); this response is
    the client's only chance to see the raw value.
    """

    access = serializers.CharField(read_only=True)
    access_expires_at = serializers.DateTimeField(read_only=True)
    refresh = serializers.CharField(read_only=True)
    refresh_expires_at = serializers.DateTimeField(read_only=True)
    session_id = serializers.UUIDField(
        read_only=True, help_text="RefreshToken.id, for later individual revocation."
    )


class RefreshSerializer(serializers.Serializer):
    """
    Validates a presented refresh token and resolves it to its
    `RefreshToken` row. Any token that is found but not currently
    active (already rotated, revoked, or expired) is treated as a
    potential compromise: the *entire rotation family* is revoked here,
    not just this one row, so a stolen-and-already-used refresh token
    cannot be replayed even in a race against the legitimate client.
    """

    refresh = serializers.CharField(write_only=True, trim_whitespace=False)

    def validate(self, attrs: dict[str, Any]) -> dict[str, Any]:
        token_hash = RefreshToken.hash_token(attrs["refresh"])
        try:
            record = RefreshToken.objects.select_related("user").get(token_hash=token_hash)
        except RefreshToken.DoesNotExist:
            raise serializers.ValidationError({"refresh": "Invalid or expired refresh token."})

        if not record.is_active:
            RefreshToken.objects.filter(
                family_id=record.family_id, revoked_at__isnull=True
            ).update(
                revoked_at=timezone.now(),
                revoked_reason=RefreshTokenRevocationReason.REUSE_DETECTED,
            )
            raise serializers.ValidationError({"refresh": "Invalid or expired refresh token."})

        if not record.user.is_active:
            raise serializers.ValidationError({"refresh": "Invalid or expired refresh token."})

        attrs["refresh_record"] = record
        return attrs


class LogoutSerializer(serializers.Serializer):
    """Identifies which refresh token/session to end."""

    refresh = serializers.CharField(write_only=True, trim_whitespace=False)


class SessionRevokeSerializer(serializers.Serializer):
    """Identifies a session (by its RefreshToken.id) to revoke. Ownership is enforced in the view."""

    session_id = serializers.UUIDField()


# ---------------------------------------------------------------------------
# Current user / profile
# ---------------------------------------------------------------------------


class UserSerializer(serializers.ModelSerializer):
    """
    Safe representation of a user: used both to read the authenticated
    user's profile and to accept profile updates. Only `first_name`/
    `last_name` are writable — everything else, including `email`, is
    read-only. Privilege fields (`is_staff`, `is_superuser`, `groups`,
    `user_permissions`) and internal security counters
    (`failed_login_attempts`, `locked_until`, `password_changed_at`)
    are not listed at all, so they can never be read or written here.
    """

    class Meta:
        model = User
        fields = [
            "id",
            "email",
            "first_name",
            "last_name",
            "is_active",
            "email_verified_at",
            "created_at",
            "updated_at",
        ]
        read_only_fields = [
            "id",
            "email",
            "is_active",
            "email_verified_at",
            "created_at",
            "updated_at",
        ]


# ---------------------------------------------------------------------------
# Password change / reset
# ---------------------------------------------------------------------------


class PasswordChangeSerializer(serializers.Serializer):
    old_password = serializers.CharField(write_only=True, trim_whitespace=False)
    new_password = serializers.CharField(write_only=True, trim_whitespace=False)
    new_password_confirm = serializers.CharField(write_only=True, trim_whitespace=False)

    def validate_old_password(self, value: str) -> str:
        user = self.context["request"].user
        if not user.check_password(value):
            raise serializers.ValidationError("Current password is incorrect.")
        return value

    def validate(self, attrs: dict[str, Any]) -> dict[str, Any]:
        if attrs["new_password"] != attrs["new_password_confirm"]:
            raise serializers.ValidationError({"new_password_confirm": "Passwords do not match."})

        user = self.context["request"].user
        try:
            validate_password(attrs["new_password"], user=user)
        except DjangoValidationError as exc:
            raise serializers.ValidationError({"new_password": list(exc.messages)})

        return attrs


class PasswordResetRequestSerializer(serializers.Serializer):
    """
    Intentionally validates only that `email` is a well-formed email
    address — never whether it belongs to an account. The view returns
    an identical response either way.
    """

    email = serializers.EmailField()


class PasswordResetConfirmSerializer(serializers.Serializer):
    token = serializers.CharField(write_only=True, trim_whitespace=False)
    new_password = serializers.CharField(write_only=True, trim_whitespace=False)
    new_password_confirm = serializers.CharField(write_only=True, trim_whitespace=False)

    def validate(self, attrs: dict[str, Any]) -> dict[str, Any]:
        if attrs["new_password"] != attrs["new_password_confirm"]:
            raise serializers.ValidationError({"new_password_confirm": "Passwords do not match."})

        token = _resolve_security_token(attrs["token"], SecurityTokenPurpose.PASSWORD_RESET)

        try:
            validate_password(attrs["new_password"], user=token.user)
        except DjangoValidationError as exc:
            raise serializers.ValidationError({"new_password": list(exc.messages)})

        attrs["security_token"] = token
        return attrs


# ---------------------------------------------------------------------------
# Email verification
# ---------------------------------------------------------------------------


class EmailVerifySerializer(serializers.Serializer):
    token = serializers.CharField(write_only=True, trim_whitespace=False)

    def validate(self, attrs: dict[str, Any]) -> dict[str, Any]:
        token = _resolve_security_token(attrs["token"], SecurityTokenPurpose.EMAIL_VERIFICATION)
        attrs["security_token"] = token
        return attrs


class EmailResendSerializer(serializers.Serializer):
    """Same enumeration-safe shape as PasswordResetRequestSerializer, kept separate for clarity of intent."""

    email = serializers.EmailField()