"""
apps.authentication.models
===========================

Ship's complete authentication foundation: identity, credentials,
account/security state, and the persistent state needed to back a
proper token-based API authentication flow.

Scope discipline: this app owns "who can authenticate, with what
credentials, in what security state, over what sessions" — nothing
about organizations, workspaces, store membership, billing, or any
other business-domain concept. Those depend on this app; this app must
never depend on them (or on any Ship app other than `core`).

Models:
    User            Identity, credentials, permissions, account state.
    SecurityToken   Hashed, single-use tokens for email verification
                    and password reset.
    RefreshToken    Hashed, revocable refresh-token/session records
                    backing stateless, short-lived API access tokens.

All three build on `apps.core.models.BaseModel` for UUID identity and
created_at/updated_at, rather than introducing a second identity
scheme. This is Core's one intended dependency direction: authentication
depends on core, core never depends on authentication.

Security invariants enforced throughout this file:
    - No plaintext passwords (Django's password hashers only, via
      `set_password()`/`AbstractBaseUser`).
    - No raw verification/reset tokens persisted — only a SHA-256
      digest (`SecurityToken.token_hash`).
    - No raw refresh tokens persisted — only a SHA-256 digest
      (`RefreshToken.token_hash`).
    - No access tokens (JWTs) persisted anywhere — they are intended
      to be short-lived and stateless, verified by signature alone.
    - No secrets in JSONField, no secrets logged, nothing here stores
      or forwards values to Core's `AuditEvent`.
"""

from __future__ import annotations

import hashlib
import uuid
from typing import Any

from django.contrib.auth.base_user import AbstractBaseUser, BaseUserManager
from django.contrib.auth.models import PermissionsMixin
from django.db import models
from django.db.models.functions import Lower
from django.utils import timezone

from apps.core.models import BaseModel


def _hash_token(raw_token: str) -> str:
    """
    Shared digest used by both `SecurityToken` and `RefreshToken`.

    SHA-256 is sufficient here (not a password hasher like PBKDF2/
    Argon2) because these are high-entropy, randomly generated tokens
    issued by the server (e.g. `secrets.token_urlsafe(32)` in the
    future service layer) — not user-chosen, low-entropy secrets. The
    threat this defends against is "someone reads the database and
    tries to replay a token," not offline brute-forcing of a
    human-guessable value.
    """
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# User
# ---------------------------------------------------------------------------


class UserManager(BaseUserManager):
    """
    The single, authoritative manager for `User`. Owns email
    normalization and guarantees passwords are always set through
    Django's password hashers — never assigned directly.
    """

    use_in_migrations = True

    def _create_user(self, email: str, password: str | None, **extra_fields: Any) -> "User":
        if not email:
            raise ValueError("Users must have an email address.")
        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_user(self, email: str, password: str | None = None, **extra_fields: Any) -> "User":
        extra_fields.setdefault("is_staff", False)
        extra_fields.setdefault("is_superuser", False)
        if extra_fields.get("is_staff") is True:
            raise ValueError("Use create_superuser() to create staff/admin accounts.")
        if extra_fields.get("is_superuser") is True:
            raise ValueError("Use create_superuser() to create superuser accounts.")
        return self._create_user(email, password, **extra_fields)

    def create_superuser(
        self, email: str, password: str | None = None, **extra_fields: Any
    ) -> "User":
        extra_fields.setdefault("is_staff", True)
        extra_fields.setdefault("is_superuser", True)
        extra_fields.setdefault("is_active", True)

        if extra_fields.get("is_staff") is not True:
            raise ValueError("Superuser must have is_staff=True.")
        if extra_fields.get("is_superuser") is not True:
            raise ValueError("Superuser must have is_superuser=True.")

        return self._create_user(email, password, **extra_fields)

    def get_by_natural_key(self, email: str) -> "User":
        """
        Case-insensitive lookup, matching the case-insensitive
        uniqueness constraint on `User.email` below. Without this
        override, Django's default (case-sensitive) lookup could
        reject a correct login or admin sign-in whenever stored/entered
        casing differs, even though the constraint already guarantees
        there is only one matching account.
        """
        return self.get(**{f"{self.model.USERNAME_FIELD}__iexact": email})


class User(BaseModel, AbstractBaseUser, PermissionsMixin):
    """
    Ship's user account: identity, credentials, permissions, and
    current account/security state.

    Identity: UUID primary key and created_at/updated_at come from
    `core.BaseModel`. Authentication behavior (password hashing,
    `last_login`) comes from `AbstractBaseUser`. Permission flags/M2M
    (`is_superuser`, `groups`, `user_permissions`) come from
    `PermissionsMixin` — Django's own group/permission system is used
    as-is; no custom Role model is introduced (see module docstring).

    Email uniqueness is enforced twice, deliberately:
        - `unique=True` on the field gives a straightforward unique
          index for the common exact-match case.
        - The `UniqueConstraint(Lower("email"))` below additionally
          prevents two accounts differing only by case, which
          `unique=True` alone would not catch. `UserManager.
          get_by_natural_key` is overridden to match this
          case-insensitive guarantee during authentication.

    Account lifecycle fields (deliberately minimal, each independently
    meaningful):
        - `is_active`  — Django's own "may this account authenticate
          at all" flag, checked by `ModelBackend` on every login.
          Deactivation is `is_active = False`; users are never
          physically deleted as a normal lifecycle operation, and
          `SoftDeleteModel` is intentionally not composed here — that
          mixin models "this row is gone", which is the wrong semantic
          for "this account is temporarily deactivated."
        - `email_verified_at` — nullable timestamp (preferred over a
          boolean for auditability: *when* was it verified, not just
          whether). Independent of `is_active`: whether an unverified
          account may log in at all is a product decision made by the
          future service/view layer, not baked into this field.
        - `failed_login_attempts` / `locked_until` — minimal, policy-free
          primitives for login-lockout. The model does not encode a
          lockout threshold or lockout duration (that is business
          policy, decided in the future service layer); it only stores
          the current counter and, if the service layer decides to
          lock the account, the timestamp until which it is locked.
        - `password_changed_at` — updated automatically by the
          `set_password()` override below. Exists so the future API
          layer can invalidate `RefreshToken`s issued before the most
          recent password change (a standard "changing your password
          logs out every other session" security behavior) without
          needing a separate event table for something that is, at
          its core, current account state.

    No security *event history* lives on `User` (e.g. no running log of
    every failed attempt, every login IP, every password change).
    That is Core's `AuditEvent`'s job, written by the future service
    layer — this model holds current state only, not an audit trail.
    """

    email = models.EmailField(
        unique=True,
        help_text="Used as the login identifier. Must be unique (case-insensitive).",
    )
    first_name = models.CharField(max_length=150, blank=True)
    last_name = models.CharField(max_length=150, blank=True)

    is_active = models.BooleanField(
        default=True,
        help_text="Whether this account may authenticate. Unselect to deactivate instead of deleting.",
    )
    is_staff = models.BooleanField(
        default=False,
        help_text="Designates whether the user can log into the Django admin site.",
    )

    email_verified_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text="When the account's email address was confirmed via a SecurityToken. Null if unverified.",
    )

    failed_login_attempts = models.PositiveSmallIntegerField(
        default=0,
        help_text="Consecutive failed login attempts since the last successful login. Policy-free counter.",
    )
    locked_until = models.DateTimeField(
        null=True,
        blank=True,
        help_text="If set and in the future, authentication should be refused until this time.",
    )
    password_changed_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text="When the password was last changed. Used to invalidate sessions predating it.",
    )

    objects = UserManager()

    USERNAME_FIELD = "email"
    REQUIRED_FIELDS: list[str] = []

    class Meta:
        verbose_name = "user"
        verbose_name_plural = "users"
        ordering = ["-created_at"]
        constraints = [
            models.UniqueConstraint(
                Lower("email"),
                name="authentication_user_unique_lower_email",
            ),
        ]
        indexes = [
            models.Index(fields=["is_active", "is_staff"], name="user_active_staff_idx"),
        ]

    def __str__(self) -> str:
        return self.email

    def get_full_name(self) -> str:
        full_name = f"{self.first_name} {self.last_name}".strip()
        return full_name or self.email

    def get_short_name(self) -> str:
        return self.first_name or self.email

    def set_password(self, raw_password: str | None) -> None:
        """
        Extends `AbstractBaseUser.set_password` to keep
        `password_changed_at` in sync automatically, so every call
        site (registration, password change, password reset,
        `createsuperuser`, admin) gets this for free instead of each
        needing to remember to update it separately. Like the base
        method, this does not save the instance — callers still call
        `save()`.
        """
        super().set_password(raw_password)
        self.password_changed_at = timezone.now()

    @property
    def is_email_verified(self) -> bool:
        return self.email_verified_at is not None

    def mark_email_verified(self) -> None:
        self.email_verified_at = timezone.now()
        self.save(update_fields=["email_verified_at", "updated_at"])

    @property
    def is_locked(self) -> bool:
        return self.locked_until is not None and self.locked_until > timezone.now()

    def reset_failed_login_state(self) -> None:
        """Clears lockout state. Intended to be called after a successful login."""
        self.failed_login_attempts = 0
        self.locked_until = None
        self.save(update_fields=["failed_login_attempts", "locked_until", "updated_at"])


# ---------------------------------------------------------------------------
# Email verification / password reset tokens
# ---------------------------------------------------------------------------


class SecurityTokenPurpose(models.TextChoices):
    EMAIL_VERIFICATION = "email_verification", "Email verification"
    PASSWORD_RESET = "password_reset", "Password reset"


class SecurityToken(BaseModel):
    """
    A single-use, expiring, hashed token backing email verification and
    password reset.

    Why this is not two separate models: both purposes need exactly the
    same shape (a hashed token tied to a user, valid until used or
    expired) and differ only in what happens when they're consumed —
    that behavior belongs in the future service layer, not here.
    `purpose` keeps the two kinds independently queryable/indexable
    without duplicating the model.

    Why this is not fields on `User`: a verification/reset token is a
    transient, single-use secret, fundamentally different from account
    state. Storing it on `User` would mean either overwriting any
    in-flight token whenever a new one is requested for a different
    purpose, or growing a wide set of nullable `*_token`/`*_expires_at`
    column pairs on the row every other app queries constantly. A
    dedicated table keeps `User` lean and lets multiple outstanding
    tokens (e.g. a stale reset request and a fresh one) exist and be
    reasoned about explicitly.

    Only `token_hash` (a SHA-256 digest) is ever stored — never the raw
    token the user receives by email. The raw token exists only for the
    moment it's generated and emailed, by the future service layer.
    """

    user = models.ForeignKey(
        "authentication.User",
        on_delete=models.CASCADE,
        related_name="security_tokens",
    )
    purpose = models.CharField(max_length=32, choices=SecurityTokenPurpose.choices)
    token_hash = models.CharField(
        max_length=64,
        unique=True,
        help_text="SHA-256 hex digest of the raw token. The raw value is never stored.",
    )
    expires_at = models.DateTimeField()
    used_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text="When this token was consumed. Null while still usable.",
    )

    class Meta:
        verbose_name = "security token"
        verbose_name_plural = "security tokens"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["user", "purpose"], name="sectoken_by_user_purpose"),
            models.Index(fields=["expires_at"], name="sectoken_by_expiry"),
        ]

    def __str__(self) -> str:
        return f"{self.get_purpose_display()} token for {self.user_id}"

    @staticmethod
    def hash_token(raw_token: str) -> str:
        return _hash_token(raw_token)

    @property
    def is_valid(self) -> bool:
        return self.used_at is None and timezone.now() < self.expires_at

    def mark_used(self) -> None:
        self.used_at = timezone.now()
        self.save(update_fields=["used_at", "updated_at"])


# ---------------------------------------------------------------------------
# Refresh tokens / API sessions
# ---------------------------------------------------------------------------


class RefreshTokenRevocationReason(models.TextChoices):
    ROTATED = "rotated", "Rotated"
    LOGOUT = "logout", "Logout"
    PASSWORD_CHANGE = "password_change", "Password change"
    USER_REVOKED = "user_revoked", "Revoked by user"
    ADMIN_REVOKED = "admin_revoked", "Revoked by administrator"
    REUSE_DETECTED = "reuse_detected", "Token reuse detected"


class RefreshToken(BaseModel):
    """
    A persisted, hashed, revocable refresh token — the durable half of
    Ship's API authentication. Access tokens (JWTs) are intentionally
    NOT modeled or stored anywhere: they are meant to be short-lived
    and stateless, verified by signature alone by the future API layer.
    Only the long-lived refresh token needs a database row, because
    only it needs to be rotatable, revocable, and auditable.

    Rotation & reuse detection: `family_id` is generated once when a
    session begins (login) and copied forward to every token issued by
    rotating that session. This lets the future service layer, on
    reuse of an already-rotated (revoked) token, revoke the entire
    family in one query — the standard defense against a stolen
    refresh token being replayed after the legitimate client has
    already rotated past it.

    Only `token_hash` (a SHA-256 digest) is stored — never the raw
    refresh token returned to the client.

    Revocation is a first-class, reasoned state (`revoked_at` +
    `revoked_reason`) rather than a bare boolean, so "why was this
    session ended" is always answerable without a separate audit
    table for what is, at its core, this record's own lifecycle.
    """

    user = models.ForeignKey(
        "authentication.User",
        on_delete=models.CASCADE,
        related_name="refresh_tokens",
    )
    token_hash = models.CharField(
        max_length=64,
        unique=True,
        help_text="SHA-256 hex digest of the raw refresh token. The raw value is never stored.",
    )
    family_id = models.UUIDField(
        default=uuid.uuid4,
        db_index=True,
        help_text="Shared across a login's full rotation chain, enabling revoke-on-reuse-detection.",
    )
    user_agent = models.CharField(max_length=255, blank=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)

    expires_at = models.DateTimeField()
    revoked_at = models.DateTimeField(null=True, blank=True)
    revoked_reason = models.CharField(
        max_length=32,
        choices=RefreshTokenRevocationReason.choices,
        blank=True,
    )

    class Meta:
        verbose_name = "refresh token"
        verbose_name_plural = "refresh tokens"
        ordering = ["-created_at"]
        constraints = [
            models.CheckConstraint(
                condition=(
                    models.Q(revoked_at__isnull=True, revoked_reason="")
                    | (models.Q(revoked_at__isnull=False) & ~models.Q(revoked_reason=""))
                ),
                name="refreshtoken_revocation_state_valid",
            ),
        ]
        indexes = [
            models.Index(fields=["user", "revoked_at"], name="reftoken_by_user_revoked"),
            models.Index(fields=["family_id"], name="reftoken_by_family"),
            models.Index(fields=["expires_at"], name="reftoken_by_expiry"),
        ]

    def __str__(self) -> str:
        state = "revoked" if self.revoked_at else "active"
        return f"Refresh token ({state}) for {self.user_id}"

    @staticmethod
    def hash_token(raw_token: str) -> str:
        return _hash_token(raw_token)

    @property
    def is_active(self) -> bool:
        return self.revoked_at is None and timezone.now() < self.expires_at

    def revoke(self, reason: str) -> None:
        self.revoked_at = timezone.now()
        self.revoked_reason = reason
        self.save(update_fields=["revoked_at", "revoked_reason", "updated_at"])