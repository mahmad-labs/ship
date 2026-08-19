"""
apps.authentication.services
==============================

The authentication APPLICATION SERVICE layer: every workflow that spans
credential handling, token/session records, verification state,
transactions, and Core audit events lives here — not in views or
serializers.

INTEGRATION STATUS (read before wiring): `views.py`/`serializers.py`
already implement every one of these workflows inline (from the prior
turn) and I was told not to redesign those files this turn. The
functions below are the complete, correct, canonical versions and are
NOT YET CALLED by `views.py` — that requires a small follow-up edit to
`views.py` (deleting its private `_issue_token_pair`/
`_issue_security_token`/`_send_*_email` helpers and calling the
equivalents here instead) that is out of scope for this response. Each
function below states exactly which existing private helper it
supersedes.

Token architecture (unchanged from prior turns): access tokens are
short-lived, stateless JWTs (`rest_framework_simplejwt`), never
persisted. Refresh tokens are long-lived, opaque, `secrets`-generated
strings; only their SHA-256 digest (`RefreshToken.hash_token`) is ever
persisted.

Email change: NOT implemented here. `User.email` has no update path in
this file or in the approved `models.py`/`serializers.py` — changing
the login identifier safely requires a dedicated confirmation workflow
(verify the NEW address before it becomes authoritative, handle the
case where it collides with another account, decide what happens to
existing sessions) that doesn't exist in the approved architecture yet.
Inventing it here would mean designing new persistent state
(models.py is explicitly out of scope this turn) — flagged, not
silently built.

Auditing policy: every function below documents whether it writes to
Core's `AuditEvent`. Not everything does — `AuditEvent` is a
business/compliance-facing log, and Core's own design explicitly warns
against using it for high-frequency plumbing. Routine refresh-token
rotation and failed-login attempts are NOT individually audited (the
former is normal traffic; the latter is already recorded as a live
counter on `User.failed_login_attempts`) — but REUSE of an
already-rotated refresh token (a genuine security incident) IS audited.
No audit call anywhere in this file includes a password, password
hash, raw token, or token hash.
"""

from __future__ import annotations

import logging
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from datetime import timezone as dt_timezone
from typing import Any

from django.conf import settings
from django.contrib.auth import authenticate
from django.core.exceptions import ValidationError
from django.db import IntegrityError, transaction
from django.http import HttpRequest
from django.utils import timezone
from rest_framework_simplejwt.tokens import AccessToken

from apps.authentication.models import (
    RefreshToken,
    RefreshTokenRevocationReason,
    SecurityToken,
    SecurityTokenPurpose,
    User,
)
from apps.authentication.validators import (
    get_valid_security_token,
    validate_active_state_transition,
    validate_email_available,
    validate_new_password,
)
from apps.core.models import ActorType, AuditAction, AuditEvent

logger = logging.getLogger(__name__)

REFRESH_TOKEN_LIFETIME: timedelta = getattr(settings, "AUTH_REFRESH_TOKEN_LIFETIME", timedelta(days=30))
VERIFICATION_TOKEN_LIFETIME: timedelta = getattr(
    settings, "AUTH_EMAIL_VERIFICATION_TOKEN_LIFETIME", timedelta(hours=24)
)
PASSWORD_RESET_TOKEN_LIFETIME: timedelta = getattr(
    settings, "AUTH_PASSWORD_RESET_TOKEN_LIFETIME", timedelta(hours=1)
)
# LOGIN_LOCKOUT_THRESHOLD: int = getattr(settings, "AUTH_LOGIN_LOCKOUT_THRESHOLD", 10)
# LOGIN_LOCKOUT_DURATION: timedelta = getattr(settings, "AUTH_LOGIN_LOCKOUT_DURATION", timedelta(minutes=15))
DEFAULT_LOGIN_LOCKOUT_THRESHOLD = 10
DEFAULT_LOGIN_LOCKOUT_DURATION = timedelta(minutes=15)


# ---------------------------------------------------------------------------
# Domain exceptions
# ---------------------------------------------------------------------------
# Predictable, catchable exception types for authentication-specific
# failures that aren't ordinary field validation (those raise Django's
# own `ValidationError` from validators.py instead). A future
# views.py/serializers.py refactor is expected to catch these and
# translate them to the appropriate DRF response — none of that
# translation happens here, since this module has no DRF dependency.


class AuthenticationError(Exception):
    """Base class for authentication-service-level errors."""


class EmailAlreadyRegisteredError(AuthenticationError):
    """Raised when registration loses a race against another concurrent registration for the same email."""


class InvalidCredentialsError(AuthenticationError):
    """Login failed: unknown email, wrong password, or inactive account (deliberately indistinguishable)."""


class AccountLockedError(AuthenticationError):
    """Credentials were correct, but the account is temporarily locked."""


class InvalidTokenError(AuthenticationError):
    """A refresh/verification/reset token was not found, already used, revoked, or expired."""


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TokenPair:
    """Mirrors `serializers.TokenResponseSerializer`'s shape exactly, so it can serialize this directly."""

    access: str
    access_expires_at: datetime
    refresh: str
    refresh_expires_at: datetime
    session_id: uuid.UUID


def _client_ip(request: HttpRequest) -> str | None:
    forwarded = request.META.get("HTTP_X_FORWARDED_FOR")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.META.get("REMOTE_ADDR")


def _audit(
    *,
    actor: User,
    action: str,
    resource: Any,
    event: str,
    metadata: dict | None = None,
    correlation_id: uuid.UUID | None = None,
) -> None:
    """
    Thin, consistent wrapper around `AuditEvent.objects.create` so every
    call site here supplies `resource_type`/`resource_id` the same way
    and never has to remember the (actor_type, actor_id) shape.
    `metadata`/`changes` must never contain a password, hash, or token
    — enforced by convention (every call site below is a small,
    reviewable literal dict), not by code in this function.
    """
    AuditEvent.objects.create(
        actor_type=ActorType.USER,
        actor_id=str(actor.id),
        action=action,
        resource_type="authentication.User",
        resource_id=str(getattr(resource, "id", resource)),
        metadata={"event": event, **(metadata or {})},
        correlation_id=correlation_id,
    )


def issue_authentication_credentials(
    user: User, request: HttpRequest, *, family_id: uuid.UUID | None = None
) -> TokenPair:
    """
    Issues one access token (stateless JWT) and one refresh token
    (persisted only as a hash). `family_id` is carried forward across
    rotation (pass the old record's `family_id`) so the whole chain can
    be revoked together if reuse is ever detected — see
    `refresh_credentials`.

    Supersedes `views._issue_token_pair`.
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

    return TokenPair(
        access=str(access_token),
        access_expires_at=datetime.fromtimestamp(access_token["exp"], tz=dt_timezone.utc),
        refresh=raw_refresh,
        refresh_expires_at=refresh_record.expires_at,
        session_id=refresh_record.id,
    )


def _issue_security_token(user: User, purpose: str, lifetime: timedelta) -> str:
    """
    Invalidates any previously issued, still-unused token of the same
    purpose (so a user never has two simultaneously valid reset/
    verification links), then issues a fresh one. Returns the RAW
    token — callers decide whether/how to expose it (see each public
    function's docstring).

    Supersedes `views._issue_security_token`.
    """
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


# ---------------------------------------------------------------------------
# Email delivery hooks (integration seam)
# ---------------------------------------------------------------------------
# Deliberately isolated so a real call into Ship's future notifications/
# email application is a one-line swap here, with nothing else in this
# file (or in views.py, once refactored to call these) needing to
# change. Authentication does not import or depend on that future app.
# Neither function ever logs a raw token. Supersedes
# `views._send_verification_email`/`views._send_password_reset_email`.


def send_verification_email(user: User, raw_token: str) -> None:
    logger.info("Verification email queued for user_id=%s", user.id)


def send_password_reset_email(user: User, raw_token: str) -> None:
    logger.info("Password reset email queued for user_id=%s", user.id)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_user(
    *, email: str, password: str, first_name: str = "", last_name: str = ""
) -> tuple[User, str]:
    """
    Creates a non-privileged account and starts email verification.
    Returns `(user, raw_verification_token)` — the caller (view) decides
    whether the raw token is ever exposed in a response (e.g. only
    under `DEBUG=True`, as the existing `RegisterView` already does).

    Never accepts `is_staff`/`is_superuser`/groups/permissions — those
    aren't parameters of this function at all, so there is no code path
    by which a caller could grant them here.

    Concurrency: `validate_email_available` is a pre-check, not the
    guarantee — two requests can both pass it before either inserts.
    The database's `UniqueConstraint(Lower("email"))` is the real
    backstop; its `IntegrityError` is caught here and converted to
    `EmailAlreadyRegisteredError` so the race produces a clean,
    predictable error instead of a raw 500.

    Audited as `AuditAction.CREATE` (actor = the new user themself).
    """
    normalized_email = validate_email_available(email)
    candidate = User(email=normalized_email, first_name=first_name, last_name=last_name)
    validate_new_password(password, candidate)

    try:
        with transaction.atomic():
            user = User.objects.create_user(
                email=normalized_email, password=password, first_name=first_name, last_name=last_name
            )
            raw_token = _issue_security_token(
                user, SecurityTokenPurpose.EMAIL_VERIFICATION, VERIFICATION_TOKEN_LIFETIME
            )
            _audit(actor=user, action=AuditAction.CREATE, resource=user, event="user_registered")
    except IntegrityError as exc:
        raise EmailAlreadyRegisteredError("A user with this email already exists.") from exc

    send_verification_email(user, raw_token)
    return user, raw_token


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------


def _register_failed_login(email: str) -> None:
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


def authenticate_user(*, email: str, password: str, request: HttpRequest) -> User:
    """
    Authenticates via Django's authentication framework, which already
    normalizes the case-insensitive lookup (`UserManager.
    get_by_natural_key`), respects `is_active`
    (`ModelBackend.user_can_authenticate`), and performs a dummy
    password hash on a non-existent email so failure timing doesn't
    distinguish "no such account" from "wrong password".

    On success: resets lockout state, records `last_login`, and writes
    one `AuditAction.LOGIN` event. On failure: records the failed
    attempt against the matching account if one exists (silently a
    no-op otherwise) and raises — individual failed attempts are not
    separately audited; the running counter on `User` already is the
    record of them.

    `is_locked` is checked only *after* a successful credential match,
    so the lockout disclosure only ever reaches someone who already
    proved they know the correct password.

    Supersedes `LoginSerializer.validate`.
    """
    user = authenticate(request=request, email=email, password=password)
    if user is None:
        _register_failed_login(email)
        raise InvalidCredentialsError("Unable to log in with the provided credentials.")

    if user.is_locked:
        raise AccountLockedError(
            "This account is temporarily locked due to repeated failed login attempts."
        )

    user.failed_login_attempts = 0
    user.locked_until = None
    user.last_login = timezone.now()
    user.save(update_fields=["failed_login_attempts", "locked_until", "last_login", "updated_at"])

    _audit(actor=user, action=AuditAction.LOGIN, resource=user, event="login_succeeded")
    return user


# ---------------------------------------------------------------------------
# Refresh / logout / sessions
# ---------------------------------------------------------------------------


def refresh_credentials(*, raw_refresh_token: str, request: HttpRequest) -> TokenPair:
    """
    Validates, rotates, and reissues. Any presented token that is found
    but not currently active (already rotated, revoked, or expired) is
    treated as a potential compromise: the entire rotation family is
    revoked, not just this one row, so a stolen-and-already-used
    refresh token can't be replayed even in a race against the
    legitimate client. That specific outcome IS audited
    (`reuse_detected`) — routine successful rotation is not, to keep
    the audit table for meaningful events rather than every API call.

    Supersedes `RefreshSerializer.validate` + `RefreshView.post`.
    """
    token_hash = RefreshToken.hash_token(raw_refresh_token)
    try:
        record = RefreshToken.objects.select_related("user").get(token_hash=token_hash)
    except RefreshToken.DoesNotExist:
        raise InvalidTokenError("Invalid or expired refresh token.")

    if not record.is_active:
        RefreshToken.objects.filter(family_id=record.family_id, revoked_at__isnull=True).update(
            revoked_at=timezone.now(), revoked_reason=RefreshTokenRevocationReason.REUSE_DETECTED
        )
        _audit(
            actor=record.user,
            action=AuditAction.OTHER,
            resource=record.user,
            event="refresh_token_reuse_detected",
            metadata={"family_id": str(record.family_id)},
        )
        raise InvalidTokenError("Invalid or expired refresh token.")

    if not record.user.is_active:
        raise InvalidTokenError("Invalid or expired refresh token.")

    with transaction.atomic():
        record.revoke(RefreshTokenRevocationReason.ROTATED)
        tokens = issue_authentication_credentials(record.user, request, family_id=record.family_id)

    return tokens


def logout_user(*, user: User, raw_refresh_token: str) -> None:
    """
    Revokes the caller's own refresh token so it can never be used to
    obtain a new access token again. The access token the client is
    currently holding remains valid — by design, it is stateless and
    short-lived; it is not revocable server-side. Clients must discard
    it locally on logout, and it will stop working on its own within
    `SIMPLE_JWT["ACCESS_TOKEN_LIFETIME"]`.

    Idempotent: silently succeeds whether or not a matching active
    token was found (never reveals which). Only audited when something
    was actually revoked.

    Supersedes the body of `LogoutView.post`.
    """
    token_hash = RefreshToken.hash_token(raw_refresh_token)
    record = RefreshToken.objects.filter(
        token_hash=token_hash, user=user, revoked_at__isnull=True
    ).first()
    if record is not None:
        record.revoke(RefreshTokenRevocationReason.LOGOUT)
        _audit(actor=user, action=AuditAction.OTHER, resource=user, event="logout")


def revoke_session(*, user: User, session_id: uuid.UUID) -> None:
    """
    Revokes one of `user`'s own sessions by `RefreshToken.id`. Scoped to
    `user=user` at the query level — never trusts that `session_id`
    belongs to the caller, so this can never revoke another user's
    session regardless of what id is supplied. Idempotent; audited only
    if a matching active session was found.

    Supersedes the body of `SessionRevokeView.post`.
    """
    record = RefreshToken.objects.filter(pk=session_id, user=user, revoked_at__isnull=True).first()
    if record is not None:
        record.revoke(RefreshTokenRevocationReason.USER_REVOKED)
        _audit(
            actor=user,
            action=AuditAction.OTHER,
            resource=user,
            event="session_revoked",
            metadata={"session_id": str(session_id)},
        )


def revoke_all_sessions(
    *, user: User, reason: str = RefreshTokenRevocationReason.USER_REVOKED
) -> int:
    """
    Revokes every active session belonging to `user`. Returns the count
    revoked. `reason` defaults to USER_REVOKED (self-service "log out
    everywhere") but is also reused by `change_password`/
    `confirm_password_reset` with `PASSWORD_CHANGE`.

    Supersedes the body of `SessionRevokeAllView.post` and the
    session-invalidation step inline in `PasswordChangeView`/
    `PasswordResetConfirmView`.
    """
    count = RefreshToken.objects.filter(user=user, revoked_at__isnull=True).update(
        revoked_at=timezone.now(), revoked_reason=reason
    )
    if count:
        _audit(
            actor=user,
            action=AuditAction.OTHER,
            resource=user,
            event="all_sessions_revoked",
            metadata={"count": count, "reason": reason},
        )
    return count


# ---------------------------------------------------------------------------
# Password change / reset
# ---------------------------------------------------------------------------


def change_password(*, user: User, current_password: str, new_password: str) -> None:
    """
    Verifies the current password via `check_password()` (never a
    manual comparison), validates the new one (rejecting reuse of the
    current password — required by this task for change, not reset),
    applies it via `set_password()` (which also updates
    `password_changed_at` — see models.py), and revokes every active
    session, since a password change should end every other logged-in
    session.

    Supersedes the body of `PasswordChangeView.post`.
    """
    if not user.check_password(current_password):
        raise InvalidCredentialsError("Current password is incorrect.")

    validate_new_password(new_password, user, forbid_reuse_of_current=True)

    with transaction.atomic():
        user.set_password(new_password)
        user.save()
        revoke_all_sessions(user=user, reason=RefreshTokenRevocationReason.PASSWORD_CHANGE)
        _audit(actor=user, action=AuditAction.UPDATE, resource=user, event="password_changed")


def request_password_reset(*, email: str) -> str | None:
    """
    Always completes without raising, regardless of whether `email`
    matches an account — the caller must return an identical response
    either way to avoid account enumeration. Returns the raw reset
    token if (and only if) a matching active account was found, or
    `None` otherwise; the caller (view) decides whether to ever expose
    that raw value (only under `DEBUG=True`, as the existing
    `PasswordResetRequestView` already does).

    Supersedes the body of `PasswordResetRequestView.post`.
    """
    user = User.objects.filter(email__iexact=email, is_active=True).first()
    if user is None:
        return None

    with transaction.atomic():
        raw_token = _issue_security_token(
            user, SecurityTokenPurpose.PASSWORD_RESET, PASSWORD_RESET_TOKEN_LIFETIME
        )
        _audit(
            actor=user, action=AuditAction.OTHER, resource=user, event="password_reset_requested"
        )

    send_password_reset_email(user, raw_token)
    return raw_token


def confirm_password_reset(*, raw_token: str, new_password: str) -> None:
    """
    Consumes a single-use password-reset token, applies the new
    password, and revokes every active session (the account may just
    have been compromised, so no session issued before the reset should
    be trusted afterward).

    Supersedes the body of `PasswordResetConfirmView.post` +
    `PasswordResetConfirmSerializer.validate`.
    """
    try:
        token = get_valid_security_token(raw_token, SecurityTokenPurpose.PASSWORD_RESET)
    except ValidationError as exc:
        raise InvalidTokenError(str(exc.message)) from exc
    user = token.user
    validate_new_password(new_password, user)

    with transaction.atomic():
        token.mark_used()
        user.set_password(new_password)
        user.save()
        revoke_all_sessions(user=user, reason=RefreshTokenRevocationReason.PASSWORD_CHANGE)
        _audit(actor=user, action=AuditAction.UPDATE, resource=user, event="password_reset_confirmed")


# ---------------------------------------------------------------------------
# Email verification
# ---------------------------------------------------------------------------


def verify_email(*, raw_token: str) -> None:
    """
    Consumes a single-use email-verification token and sets
    `User.email_verified_at`.

    Supersedes the body of `EmailVerifyView.post` +
    `EmailVerifySerializer.validate`.
    """
    try:
        token = get_valid_security_token(raw_token, SecurityTokenPurpose.EMAIL_VERIFICATION)
    except ValidationError as exc:
        raise InvalidTokenError(str(exc.message)) from exc

    with transaction.atomic():
        token.mark_used()
        token.user.mark_email_verified()
        _audit(actor=token.user, action=AuditAction.UPDATE, resource=token.user, event="email_verified")


def resend_email_verification(*, email: str) -> str | None:
    """
    Same enumeration-safe contract as `request_password_reset`: always
    completes, never raises for an unknown/already-verified email,
    returns the raw token only when a new one was actually issued.
    Request-rate abuse (someone hammering this endpoint) is a view-layer
    concern already handled by `EmailResendView`'s `ScopedRateThrottle`
    — not duplicated here.

    Supersedes the body of `EmailResendView.post`.
    """
    user = User.objects.filter(email__iexact=email, is_active=True).first()
    if user is None or user.is_email_verified:
        return None

    raw_token = _issue_security_token(
        user, SecurityTokenPurpose.EMAIL_VERIFICATION, VERIFICATION_TOKEN_LIFETIME
    )
    send_verification_email(user, raw_token)
    return raw_token


# ---------------------------------------------------------------------------
# Account activation state
# ---------------------------------------------------------------------------
# No existing URL exposes these (see module docstring). Provided because
# the task explicitly requires them and `User.is_active` already fully
# supports the operation; wiring an endpoint to them is future work.


def deactivate_user(*, user: User, actor: User | None = None, reason: str = "") -> None:
    """
    Sets `is_active=False` and revokes every active session (a
    deactivated account should not retain live sessions). `actor`
    defaults to `user` (self-deactivation); pass a different, staff/
    superuser `User` for an administrator-initiated deactivation — the
    audit record reflects whichever actor performed the action.
    """
    validate_active_state_transition(user, target_active=False)
    revoke_reason = (
        RefreshTokenRevocationReason.USER_REVOKED
        if actor is None or actor == user
        else RefreshTokenRevocationReason.ADMIN_REVOKED
    )
    with transaction.atomic():
        user.is_active = False
        user.save(update_fields=["is_active", "updated_at"])
        revoke_all_sessions(user=user, reason=revoke_reason)
        _audit(
            actor=actor or user,
            action=AuditAction.UPDATE,
            resource=user,
            event="account_deactivated",
            metadata={"reason": reason} if reason else None,
        )


def reactivate_user(*, user: User, actor: User | None = None) -> None:
    """Sets `is_active=True`. Does not restore prior sessions — the user must log in again."""
    validate_active_state_transition(user, target_active=True)
    with transaction.atomic():
        user.is_active = True
        user.save(update_fields=["is_active", "updated_at"])
        _audit(actor=actor or user, action=AuditAction.UPDATE, resource=user, event="account_reactivated")