"""
apps.authentication.tests.test_models
=======================================

Tests against the real, unmodified `apps.authentication.models` /
`UserManager`. Covers identity, password security, account/security
state, Django permissions compatibility, `SecurityToken`, and
`RefreshToken` — including database-level constraints (never relying
on serializer/service validation alone to prove uniqueness).
"""

import uuid

from django.contrib.auth.hashers import identify_hasher, is_password_usable
from django.db import IntegrityError, transaction
from django.test import TestCase
from django.utils import timezone

from apps.authentication.models import (
    RefreshToken,
    RefreshTokenRevocationReason,
    SecurityToken,
    SecurityTokenPurpose,
    User,
)


class AuthTestHelpersMixin:
    """Small, local helpers — no factory library is used elsewhere in this project."""

    def make_user(self, email="user@example.com", password="Str0ng-Pass!2024", **extra):
        return User.objects.create_user(email=email, password=password, **extra)


# ---------------------------------------------------------------------------
# User identity
# ---------------------------------------------------------------------------


class UserIdentityTests(AuthTestHelpersMixin, TestCase):
    def test_user_has_uuid_primary_key(self):
        user = self.make_user()
        self.assertIsInstance(user.id, uuid.UUID)

    def test_username_field_is_email(self):
        self.assertEqual(User.USERNAME_FIELD, "email")

    def test_required_fields_excludes_password_and_username_field(self):
        self.assertNotIn("password", User.REQUIRED_FIELDS)
        self.assertNotIn("email", User.REQUIRED_FIELDS)

    def test_email_is_normalized_by_manager(self):
        user = User.objects.create_user(email="Someone@EXAMPLE.com", password="Str0ng-Pass!2024")
        # Django's normalize_email lowercases only the domain part.
        self.assertEqual(user.email, "Someone@example.com")

    def test_case_insensitive_duplicate_email_rejected_at_db_level(self):
        self.make_user(email="dup@example.com")
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                User.objects.create_user(email="DUP@example.com", password="Str0ng-Pass!2024")

    def test_exact_duplicate_email_rejected_at_db_level(self):
        self.make_user(email="exact@example.com")
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                User.objects.create_user(email="exact@example.com", password="Str0ng-Pass!2024")

    def test_blank_email_rejected_by_manager(self):
        with self.assertRaises(ValueError):
            User.objects.create_user(email="", password="Str0ng-Pass!2024")

    def test_get_full_name_combines_first_and_last(self):
        user = self.make_user(first_name="Ada", last_name="Lovelace")
        self.assertEqual(user.get_full_name(), "Ada Lovelace")

    def test_get_full_name_falls_back_to_email_when_names_blank(self):
        user = self.make_user(email="noname@example.com")
        self.assertEqual(user.get_full_name(), "noname@example.com")

    def test_get_short_name_prefers_first_name(self):
        user = self.make_user(first_name="Grace")
        self.assertEqual(user.get_short_name(), "Grace")

    def test_get_short_name_falls_back_to_email(self):
        user = self.make_user(email="short@example.com")
        self.assertEqual(user.get_short_name(), "short@example.com")

    def test_str_returns_email(self):
        user = self.make_user(email="strtest@example.com")
        self.assertEqual(str(user), "strtest@example.com")

    def test_created_at_and_updated_at_populated(self):
        user = self.make_user()
        self.assertIsNotNone(user.created_at)
        self.assertIsNotNone(user.updated_at)
        self.assertTrue(timezone.is_aware(user.created_at))

    def test_updated_at_advances_on_save(self):
        user = self.make_user()
        original = user.updated_at
        user.first_name = "Changed"
        user.save(update_fields=["first_name", "updated_at"])
        user.refresh_from_db()
        self.assertGreaterEqual(user.updated_at, original)


# ---------------------------------------------------------------------------
# Password security
# ---------------------------------------------------------------------------


class UserPasswordSecurityTests(AuthTestHelpersMixin, TestCase):
    def test_password_is_hashed_not_plaintext(self):
        user = self.make_user(password="Str0ng-Pass!2024")
        self.assertNotEqual(user.password, "Str0ng-Pass!2024")
        self.assertTrue(is_password_usable(user.password))
        # Confirms the stored value is a real, identifiable hash produced
        # by one of the project's configured AUTH_PASSWORD hashers, without
        # assuming which specific algorithm (PBKDF2, Argon2, ...) is
        # configured — identify_hasher() raises ValueError for anything
        # that isn't a recognized hash format, including plaintext.
        identify_hasher(user.password)

    def test_check_password_succeeds_for_correct_password(self):
        user = self.make_user(password="Str0ng-Pass!2024")
        self.assertTrue(user.check_password("Str0ng-Pass!2024"))

    def test_check_password_fails_for_incorrect_password(self):
        user = self.make_user(password="Str0ng-Pass!2024")
        self.assertFalse(user.check_password("totally-wrong"))

    def test_set_password_updates_password_changed_at(self):
        user = self.make_user()
        first_ts = user.password_changed_at
        self.assertIsNotNone(first_ts)

        user.set_password("New-Str0ng-Pass!99")
        user.save()
        user.refresh_from_db()
        self.assertGreaterEqual(user.password_changed_at, first_ts)
        self.assertTrue(user.check_password("New-Str0ng-Pass!99"))

    def test_set_password_does_not_persist_without_explicit_save(self):
        user = self.make_user()
        user.set_password("Another-Str0ng-Pass!1")
        fresh = User.objects.get(pk=user.pk)
        self.assertFalse(fresh.check_password("Another-Str0ng-Pass!1"))

    def test_set_unusable_password_prevents_authentication(self):
        user = self.make_user()
        user.set_unusable_password()
        user.save()
        self.assertFalse(user.check_password("anything"))
        self.assertFalse(user.has_usable_password())


# ---------------------------------------------------------------------------
# Account state
# ---------------------------------------------------------------------------


class UserAccountStateTests(AuthTestHelpersMixin, TestCase):
    def test_is_active_defaults_to_true(self):
        user = self.make_user()
        self.assertTrue(user.is_active)

    def test_is_email_verified_false_by_default(self):
        user = self.make_user()
        self.assertFalse(user.is_email_verified)
        self.assertIsNone(user.email_verified_at)

    def test_mark_email_verified_sets_timestamp(self):
        user = self.make_user()
        user.mark_email_verified()
        user.refresh_from_db()
        self.assertTrue(user.is_email_verified)
        self.assertIsNotNone(user.email_verified_at)

    def test_is_locked_false_when_locked_until_is_none(self):
        user = self.make_user()
        self.assertFalse(user.is_locked)

    def test_is_locked_true_when_locked_until_in_future(self):
        user = self.make_user()
        user.locked_until = timezone.now() + timezone.timedelta(minutes=10)
        user.save(update_fields=["locked_until"])
        self.assertTrue(user.is_locked)

    def test_is_locked_false_when_locked_until_in_past(self):
        user = self.make_user()
        user.locked_until = timezone.now() - timezone.timedelta(minutes=10)
        user.save(update_fields=["locked_until"])
        self.assertFalse(user.is_locked)

    def test_reset_failed_login_state_clears_counters(self):
        user = self.make_user()
        user.failed_login_attempts = 7
        user.locked_until = timezone.now() + timezone.timedelta(minutes=5)
        user.save(update_fields=["failed_login_attempts", "locked_until"])

        user.reset_failed_login_state()
        user.refresh_from_db()
        self.assertEqual(user.failed_login_attempts, 0)
        self.assertIsNone(user.locked_until)


# ---------------------------------------------------------------------------
# Django permissions framework compatibility
# ---------------------------------------------------------------------------


class UserPermissionsFrameworkTests(AuthTestHelpersMixin, TestCase):
    def test_regular_user_is_not_staff_or_superuser(self):
        user = self.make_user()
        self.assertFalse(user.is_staff)
        self.assertFalse(user.is_superuser)

    def test_superuser_has_all_permissions(self):
        superuser = User.objects.create_superuser(email="root@example.com", password="Str0ng-Pass!2024")
        self.assertTrue(superuser.is_staff)
        self.assertTrue(superuser.is_superuser)
        self.assertTrue(superuser.is_active)
        self.assertTrue(superuser.has_perm("any.permission"))

    def test_user_can_be_assigned_to_groups(self):
        from django.contrib.auth.models import Group

        user = self.make_user()
        group = Group.objects.create(name="Support")
        user.groups.add(group)
        self.assertIn(group, user.groups.all())


# ---------------------------------------------------------------------------
# UserManager
# ---------------------------------------------------------------------------


class UserManagerTests(TestCase):
    def test_create_user_normalizes_email(self):
        user = User.objects.create_user(email="Foo.Bar@EXAMPLE.com", password="Str0ng-Pass!2024")
        self.assertEqual(user.email, "Foo.Bar@example.com")

    def test_create_user_hashes_password(self):
        user = User.objects.create_user(email="hash@example.com", password="Str0ng-Pass!2024")
        self.assertTrue(is_password_usable(user.password))

    def test_create_user_defaults_is_staff_false(self):
        user = User.objects.create_user(email="staffdefault@example.com", password="Str0ng-Pass!2024")
        self.assertFalse(user.is_staff)

    def test_create_user_defaults_is_superuser_false(self):
        user = User.objects.create_user(email="superdefault@example.com", password="Str0ng-Pass!2024")
        self.assertFalse(user.is_superuser)

    def test_create_user_rejects_empty_email(self):
        with self.assertRaises(ValueError):
            User.objects.create_user(email="", password="Str0ng-Pass!2024")

    def test_create_user_rejects_is_staff_true(self):
        """Privilege escalation guard: create_user must never be usable to mint staff accounts."""
        with self.assertRaises(ValueError):
            User.objects.create_user(email="escalate1@example.com", password="Str0ng-Pass!2024", is_staff=True)

    def test_create_user_rejects_is_superuser_true(self):
        """Privilege escalation guard: create_user must never be usable to mint superuser accounts."""
        with self.assertRaises(ValueError):
            User.objects.create_user(
                email="escalate2@example.com", password="Str0ng-Pass!2024", is_superuser=True
            )

    def test_create_superuser_sets_all_privilege_flags(self):
        user = User.objects.create_superuser(email="super@example.com", password="Str0ng-Pass!2024")
        self.assertTrue(user.is_staff)
        self.assertTrue(user.is_superuser)
        self.assertTrue(user.is_active)

    def test_create_superuser_rejects_is_staff_false(self):
        with self.assertRaises(ValueError):
            User.objects.create_superuser(
                email="badsuper1@example.com", password="Str0ng-Pass!2024", is_staff=False
            )

    def test_create_superuser_rejects_is_superuser_false(self):
        with self.assertRaises(ValueError):
            User.objects.create_superuser(
                email="badsuper2@example.com", password="Str0ng-Pass!2024", is_superuser=False
            )

    def test_get_by_natural_key_is_case_insensitive(self):
        User.objects.create_user(email="CaseTest@Example.com", password="Str0ng-Pass!2024")
        found = User.objects.get_by_natural_key("casetest@example.com")
        self.assertEqual(found.email, "CaseTest@example.com")

    def test_get_by_natural_key_raises_for_unknown_email(self):
        with self.assertRaises(User.DoesNotExist):
            User.objects.get_by_natural_key("nobody@example.com")


# ---------------------------------------------------------------------------
# SecurityToken
# ---------------------------------------------------------------------------


class SecurityTokenTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="sectoken@example.com", password="Str0ng-Pass!2024")

    def _make_token(self, purpose=SecurityTokenPurpose.EMAIL_VERIFICATION, raw="raw-token-value-123456"):
        return SecurityToken.objects.create(
            user=self.user,
            purpose=purpose,
            token_hash=SecurityToken.hash_token(raw),
            expires_at=timezone.now() + timezone.timedelta(hours=1),
        )

    def test_hash_token_never_equals_raw_value(self):
        raw = "some-raw-token-value"
        hashed = SecurityToken.hash_token(raw)
        self.assertNotEqual(hashed, raw)
        self.assertEqual(len(hashed), 64)  # SHA-256 hex digest

    def test_hash_token_is_deterministic(self):
        raw = "some-raw-token-value"
        self.assertEqual(SecurityToken.hash_token(raw), SecurityToken.hash_token(raw))

    def test_token_valid_before_use_and_before_expiry(self):
        token = self._make_token()
        self.assertTrue(token.is_valid)

    def test_token_invalid_after_expiry(self):
        token = self._make_token()
        token.expires_at = timezone.now() - timezone.timedelta(seconds=1)
        token.save(update_fields=["expires_at"])
        self.assertFalse(token.is_valid)

    def test_mark_used_sets_used_at_and_invalidates(self):
        token = self._make_token()
        self.assertIsNone(token.used_at)
        token.mark_used()
        token.refresh_from_db()
        self.assertIsNotNone(token.used_at)
        self.assertFalse(token.is_valid)

    def test_token_hash_uniqueness_enforced_at_db_level(self):
        self._make_token(raw="shared-raw-value")
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                SecurityToken.objects.create(
                    user=self.user,
                    purpose=SecurityTokenPurpose.PASSWORD_RESET,
                    token_hash=SecurityToken.hash_token("shared-raw-value"),
                    expires_at=timezone.now() + timezone.timedelta(hours=1),
                )

    def test_str_includes_purpose(self):
        token = self._make_token(purpose=SecurityTokenPurpose.PASSWORD_RESET)
        self.assertIn("Password reset", str(token))


# ---------------------------------------------------------------------------
# RefreshToken
# ---------------------------------------------------------------------------


class RefreshTokenTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="reftoken@example.com", password="Str0ng-Pass!2024")

    def _make_token(self, raw="raw-refresh-value-abcdef"):
        return RefreshToken.objects.create(
            user=self.user,
            token_hash=RefreshToken.hash_token(raw),
            expires_at=timezone.now() + timezone.timedelta(days=30),
        )

    def test_hash_token_never_equals_raw_value(self):
        raw = "some-raw-refresh-token"
        hashed = RefreshToken.hash_token(raw)
        self.assertNotEqual(hashed, raw)

    def test_family_id_defaults_to_a_uuid(self):
        token = self._make_token()
        self.assertIsInstance(token.family_id, uuid.UUID)

    def test_is_active_true_for_fresh_token(self):
        token = self._make_token()
        self.assertTrue(token.is_active)

    def test_is_active_false_after_expiry(self):
        token = self._make_token()
        token.expires_at = timezone.now() - timezone.timedelta(seconds=1)
        token.save(update_fields=["expires_at"])
        self.assertFalse(token.is_active)

    def test_revoke_sets_revoked_at_and_reason(self):
        token = self._make_token()
        token.revoke(RefreshTokenRevocationReason.LOGOUT)
        token.refresh_from_db()
        self.assertIsNotNone(token.revoked_at)
        self.assertEqual(token.revoked_reason, RefreshTokenRevocationReason.LOGOUT)
        self.assertFalse(token.is_active)

    def test_revocation_state_check_constraint_rejects_revoked_at_without_reason(self):
        token = self._make_token()
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                RefreshToken.objects.filter(pk=token.pk).update(
                    revoked_at=timezone.now(), revoked_reason=""
                )

    def test_token_hash_uniqueness_enforced_at_db_level(self):
        self._make_token(raw="shared-refresh-value")
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                RefreshToken.objects.create(
                    user=self.user,
                    token_hash=RefreshToken.hash_token("shared-refresh-value"),
                    expires_at=timezone.now() + timezone.timedelta(days=30),
                )

    def test_str_reflects_active_state(self):
        token = self._make_token()
        self.assertIn("active", str(token))
        token.revoke(RefreshTokenRevocationReason.LOGOUT)
        self.assertIn("revoked", str(token))