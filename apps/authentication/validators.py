"""
apps.authentication.validators
================================

Reusable, side-effect-free "is this input/state valid?" checks for the
authentication domain. Deliberately does NOT reimplement anything
Django already does well:
    - Email format: Django's `EmailField` (used throughout
      serializers.py) already validates this.
    - Password strength: `django.contrib.auth.password_validation.
      validate_password()`, driven by the project's configured
      `AUTH_PASSWORD_VALIDATORS`, is called directly below — never
      duplicated or reimplemented.

Every validator raises `django.core.exceptions.ValidationError` (never
a DRF exception), so this module has no dependency on Django REST
Framework and is usable from `services.py`, the Django admin, a future
management command, or a serializer — not just an API view. Existing
`serializers.py` currently reimplements the email-uniqueness and
password-confirmation checks below inline (see each function's
docstring for exactly which lines) rather than calling this module —
noted, not silently duplicated, since I was told not to edit
`serializers.py` this turn.
"""

from __future__ import annotations

from django.contrib.auth.password_validation import validate_password as django_validate_password
from django.core.exceptions import ValidationError

from apps.authentication.models import SecurityToken, User


def validate_email_available(email: str) -> str:
    """
    Normalizes `email` (via `UserManager.normalize_email`, the same
    normalization `User.objects.create_user` applies) and raises if a
    case-insensitive match already exists. Returns the normalized
    email for the caller to use.

    This exact check is currently duplicated inline in
    `RegistrationSerializer.validate_email`. The database's
    `UniqueConstraint(Lower("email"))` (models.py) remains the
    authoritative guarantee regardless — this is a fast, friendly
    pre-check, not a replacement for it. See
    `services.register_user` for how the race between this check and
    the actual insert is handled.
    """
    normalized = User.objects.normalize_email(email)
    if User.objects.filter(email__iexact=normalized).exists():
        raise ValidationError("A user with this email already exists.", code="email_taken")
    return normalized


def validate_password_confirmation(password: str, password_confirm: str) -> None:
    """
    Currently duplicated inline wherever a serializer compares
    `password`/`password_confirm` or `new_password`/
    `new_password_confirm` (RegistrationSerializer,
    PasswordChangeSerializer, PasswordResetConfirmSerializer).
    """
    if password != password_confirm:
        raise ValidationError("Passwords do not match.", code="password_mismatch")


def validate_new_password(
    password: str, user: User, *, forbid_reuse_of_current: bool = False
) -> None:
    """
    Runs Django's configured `AUTH_PASSWORD_VALIDATORS` via
    `validate_password()` — the single source of truth for password
    strength, never reimplemented here.

    Adds one genuine rule Django doesn't provide:
    `forbid_reuse_of_current` rejects a new password identical to the
    account's current one. Used by `services.change_password` (where
    the task explicitly requires this); left off by default (and by
    `services.confirm_password_reset`) since a user who has forgotten
    their password has no current password to meaningfully compare
    against from their own perspective.
    """
    django_validate_password(password, user=user)
    if forbid_reuse_of_current and user.pk and user.check_password(password):
        raise ValidationError(
            "New password must be different from your current password.",
            code="password_unchanged",
        )


def validate_raw_token_format(raw_token: str, *, min_length: int = 16) -> None:
    """
    Lightweight defensive check on a client-supplied raw token
    (verification/reset/refresh) before it's hashed and used in a
    database lookup — rejects obviously-malformed input (empty,
    whitespace, too short to plausibly be a `secrets.token_urlsafe`
    output) without a wasted query. Not a substitute for the real
    validity check, which requires the database (see
    `get_valid_security_token` below and `services.refresh_credentials`).
    """
    if not raw_token or not raw_token.strip():
        raise ValidationError("Token must not be empty.", code="token_empty")
    if len(raw_token) < min_length:
        raise ValidationError("Token is malformed.", code="token_malformed")


def validate_active_state_transition(user: User, *, target_active: bool) -> None:
    """
    Rejects a no-op activate/deactivate call so `services.deactivate_user`/
    `reactivate_user` fail clearly instead of silently no-op-saving.
    """
    if user.is_active == target_active:
        state = "active" if target_active else "inactive"
        raise ValidationError(f"User is already {state}.", code="no_state_change")


def get_valid_security_token(raw_token: str, purpose: str) -> SecurityToken:
    """
    Resolves a raw token to its `SecurityToken` row and confirms it is
    currently valid (`SecurityToken.is_valid`: not used, not expired)
    for the given `purpose`. Read-only — does not call `mark_used()`;
    consuming the token is a workflow step owned by `services.py`.

    Deliberately returns the identical error for "no such token",
    "already used", and "expired" — distinguishing them would let an
    attacker learn more about a token they don't legitimately hold.

    Currently duplicated inline as `_resolve_security_token()` in
    `serializers.py` (used by `EmailVerifySerializer` and
    `PasswordResetConfirmSerializer`).
    """
    validate_raw_token_format(raw_token)
    token_hash = SecurityToken.hash_token(raw_token)
    try:
        token = SecurityToken.objects.select_related("user").get(
            token_hash=token_hash, purpose=purpose
        )
    except SecurityToken.DoesNotExist:
        raise ValidationError("Invalid or expired token.", code="invalid_token")
    if not token.is_valid:
        raise ValidationError("Invalid or expired token.", code="invalid_token")
    return token