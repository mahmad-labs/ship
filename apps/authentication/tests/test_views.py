"""
apps.authentication.tests.test_views
======================================

End-to-end API tests against the real `urls.py`/`views.py`, using
`rest_framework.test.APIClient`. Every endpoint actually defined in
`urls.py` is covered:

    POST   /register/
    POST   /login/
    POST   /logout/
    POST   /refresh/
    GET    /me/
    PATCH  /me/
    POST   /password/change/
    POST   /password/reset/request/
    POST   /password/reset/confirm/
    POST   /email/verify/
    POST   /email/resend/
    POST   /sessions/revoke/
    POST   /sessions/revoke-all/

Paths are resolved via Django's `reverse()` against the `authentication`
namespace rather than hard-coded strings, so these tests stay correct
if the URL prefix changes. No response is accepted on HTTP status
alone — every test also asserts on response *content* (or its absence)
where that content is security-relevant.
"""

import uuid

from django.core.cache import cache
from django.test import override_settings
from django.urls import reverse
from django.utils import timezone
from rest_framework import status
from rest_framework.test import APITestCase

from apps.authentication.models import (
    RefreshToken,
    RefreshTokenRevocationReason,
    SecurityToken,
    SecurityTokenPurpose,
    User,
)

VALID_PASSWORD = "Str0ng-Pass!2024"


def url(name, **kwargs):
    return reverse(f"authentication:{name}", kwargs=kwargs or None)


class AuthAPITestCase(APITestCase):
    """
    Base class with shared helpers for creating users and issuing
    tokens through the real API.

    `cache.clear()` in setUp is required for correct test isolation:
    `EmailResendView`/`PasswordResetRequestView` use DRF's
    `ScopedRateThrottle`, which — with no `CACHES` configured — stores
    its request counts in Django's default `LocMemCache`. That cache is
    process-wide and, unlike the database, is NOT reset by Django's test
    machinery between test methods, so without this, throttle state
    accumulated by an earlier test can cause a later, unrelated test to
    receive an unexpected 429.
    """

    def setUp(self):
        super().setUp()
        cache.clear()

    def make_user(self, email="user@example.com", password=VALID_PASSWORD, **extra):
        return User.objects.create_user(email=email, password=password, **extra)

    def register(self, **overrides):
        payload = {
            "email": "reguser@example.com",
            "password": VALID_PASSWORD,
            "password_confirm": VALID_PASSWORD,
            "first_name": "Reg",
            "last_name": "User",
        }
        payload.update(overrides)
        return self.client.post(url("register"), payload, format="json")

    def login(self, email, password=VALID_PASSWORD):
        return self.client.post(url("login"), {"email": email, "password": password}, format="json")

    def authenticate_as(self, user):
        """Logs in via the real endpoint and attaches the resulting access token to self.client."""
        resp = self.login(user.email)
        access = resp.json()["access"]
        self.client.credentials(HTTP_AUTHORIZATION=f"Bearer {access}")
        return resp.json()


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class RegisterAPITests(AuthAPITestCase):
    def test_successful_registration_returns_201_and_safe_user(self):
        resp = self.register()
        self.assertEqual(resp.status_code, status.HTTP_201_CREATED)
        body = resp.json()
        self.assertEqual(body["email"].lower(), "reguser@example.com")
        self.assertIn("id", body)

    def test_registration_never_returns_password(self):
        resp = self.register()
        self.assertNotIn("password", resp.json())
        self.assertNotIn("password_confirm", resp.json())

    def test_registration_never_returns_password_hash(self):
        resp = self.register()
        body_str = str(resp.json())
        user = User.objects.get(email__iexact="reguser@example.com")
        self.assertNotIn(user.password, body_str)

    def test_missing_email_returns_400(self):
        resp = self.register(email="")
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_invalid_email_returns_400(self):
        resp = self.register(email="not-an-email")
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_duplicate_email_returns_400(self):
        self.make_user(email="dupe@example.com")
        resp = self.register(email="dupe@example.com")
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_case_insensitive_duplicate_email_returns_400(self):
        self.make_user(email="dupe@example.com")
        resp = self.register(email="DUPE@EXAMPLE.com")
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_weak_password_returns_400(self):
        resp = self.register(password="weak", password_confirm="weak")
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_mismatched_password_confirmation_returns_400(self):
        resp = self.register(password_confirm="Different-Pass!1")
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_privilege_escalation_attempt_is_ignored(self):
        resp = self.register(email="escalate@example.com", is_staff=True, is_superuser=True)
        self.assertEqual(resp.status_code, status.HTTP_201_CREATED)
        user = User.objects.get(email__iexact="escalate@example.com")
        self.assertFalse(user.is_staff)
        self.assertFalse(user.is_superuser)

    def test_registration_creates_unverified_account(self):
        self.register()
        user = User.objects.get(email__iexact="reguser@example.com")
        self.assertFalse(user.is_email_verified)

    @override_settings(DEBUG=True)
    def test_debug_mode_exposes_verification_token(self):
        resp = self.register()
        self.assertIn("debug_verification_token", resp.json())

    @override_settings(DEBUG=False)
    def test_production_mode_never_exposes_verification_token(self):
        resp = self.register(email="prodreg@example.com")
        self.assertNotIn("debug_verification_token", resp.json())


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------


class LoginAPITests(AuthAPITestCase):
    def setUp(self):
        super().setUp()
        self.user = self.make_user(email="login@example.com")

    def test_valid_login_returns_200_with_tokens(self):
        resp = self.login("login@example.com")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        body = resp.json()
        for field in ("access", "refresh", "access_expires_at", "refresh_expires_at", "session_id"):
            self.assertIn(field, body)

    def test_wrong_password_returns_400(self):
        resp = self.login("login@example.com", password="wrong")
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_nonexistent_account_returns_400(self):
        resp = self.login("nobody@example.com")
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_inactive_account_returns_400(self):
        self.user.is_active = False
        self.user.save(update_fields=["is_active"])
        resp = self.login("login@example.com")
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_case_insensitive_email_login_succeeds(self):
        resp = self.login("LOGIN@EXAMPLE.com")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)

    def test_response_never_contains_password_hash(self):
        resp = self.login("login@example.com")
        self.user.refresh_from_db()
        self.assertNotIn(self.user.password, str(resp.json()))

    def test_response_never_contains_internal_token_hash(self):
        resp = self.login("login@example.com")
        record = RefreshToken.objects.get(user=self.user)
        self.assertNotIn(record.token_hash, str(resp.json()))

    def test_failed_login_does_not_create_a_refresh_token(self):
        self.login("login@example.com", password="wrong")
        self.assertEqual(RefreshToken.objects.filter(user=self.user).count(), 0)

    def test_successful_login_creates_exactly_one_refresh_token(self):
        self.login("login@example.com")
        self.assertEqual(RefreshToken.objects.filter(user=self.user).count(), 1)

    def test_account_locks_after_threshold_failed_attempts(self):
        """
        Relies on the ambient `AUTH_LOGIN_LOCKOUT_THRESHOLD` configured for
        this test run (see settings) rather than `@override_settings`:
        `serializers.LOGIN_LOCKOUT_THRESHOLD` is captured as a module-level
        constant at import time (`getattr(settings, ..., default)` at
        module scope), so `override_settings` has no effect on it at
        runtime — see the accompanying response for this discovered defect.
        """
        threshold = 10  # must match AUTH_LOGIN_LOCKOUT_THRESHOLD in the test settings module
        for _ in range(threshold):
            self.login("login@example.com", password="wrong")
        resp = self.login("login@example.com")  # correct password, but now locked
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)
        self.user.refresh_from_db()
        self.assertTrue(self.user.is_locked)


# ---------------------------------------------------------------------------
# Refresh
# ---------------------------------------------------------------------------


class RefreshAPITests(AuthAPITestCase):
    def setUp(self):
        super().setUp()
        self.user = self.make_user(email="refresh@example.com")
        self.tokens = self.login("refresh@example.com").json()

    def test_valid_refresh_returns_new_token_pair(self):
        resp = self.client.post(url("refresh"), {"refresh": self.tokens["refresh"]}, format="json")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        new_tokens = resp.json()
        self.assertNotEqual(new_tokens["refresh"], self.tokens["refresh"])

    def test_old_refresh_token_invalid_after_rotation(self):
        self.client.post(url("refresh"), {"refresh": self.tokens["refresh"]}, format="json")
        resp = self.client.post(url("refresh"), {"refresh": self.tokens["refresh"]}, format="json")
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_replaying_rotated_token_revokes_the_new_one_too(self):
        """Reuse-detection: the new token issued by rotation must also stop working."""
        rotated = self.client.post(url("refresh"), {"refresh": self.tokens["refresh"]}, format="json").json()
        self.client.post(url("refresh"), {"refresh": self.tokens["refresh"]}, format="json")  # replay
        resp = self.client.post(url("refresh"), {"refresh": rotated["refresh"]}, format="json")
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_malformed_refresh_token_returns_400(self):
        resp = self.client.post(url("refresh"), {"refresh": "not-a-real-token"}, format="json")
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_missing_refresh_field_returns_400(self):
        resp = self.client.post(url("refresh"), {}, format="json")
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_expired_refresh_token_rejected(self):
        record = RefreshToken.objects.get(user=self.user)
        record.expires_at = timezone.now() - timezone.timedelta(seconds=1)
        record.save(update_fields=["expires_at"])
        resp = self.client.post(url("refresh"), {"refresh": self.tokens["refresh"]}, format="json")
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_revoked_refresh_token_rejected(self):
        record = RefreshToken.objects.get(user=self.user)
        record.revoke(RefreshTokenRevocationReason.LOGOUT)
        resp = self.client.post(url("refresh"), {"refresh": self.tokens["refresh"]}, format="json")
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_rotation_preserves_family_id(self):
        old_record = RefreshToken.objects.get(user=self.user)
        resp = self.client.post(url("refresh"), {"refresh": self.tokens["refresh"]}, format="json")
        new_hash = RefreshToken.hash_token(resp.json()["refresh"])
        new_record = RefreshToken.objects.get(token_hash=new_hash)
        self.assertEqual(new_record.family_id, old_record.family_id)

    def test_refresh_response_never_contains_token_hash(self):
        resp = self.client.post(url("refresh"), {"refresh": self.tokens["refresh"]}, format="json")
        new_record = RefreshToken.objects.get(token_hash=RefreshToken.hash_token(resp.json()["refresh"]))
        self.assertNotIn(new_record.token_hash, str(resp.json()))


# ---------------------------------------------------------------------------
# Logout
# ---------------------------------------------------------------------------


class LogoutAPITests(AuthAPITestCase):
    def setUp(self):
        super().setUp()
        self.user = self.make_user(email="logout@example.com")
        self.tokens = self.authenticate_as(self.user)

    def test_authenticated_logout_returns_204(self):
        resp = self.client.post(url("logout"), {"refresh": self.tokens["refresh"]}, format="json")
        self.assertEqual(resp.status_code, status.HTTP_204_NO_CONTENT)

    def test_unauthenticated_logout_returns_401(self):
        self.client.credentials()
        resp = self.client.post(url("logout"), {"refresh": self.tokens["refresh"]}, format="json")
        self.assertIn(resp.status_code, (status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN))

    def test_logout_revokes_the_refresh_token(self):
        self.client.post(url("logout"), {"refresh": self.tokens["refresh"]}, format="json")
        record = RefreshToken.objects.get(token_hash=RefreshToken.hash_token(self.tokens["refresh"]))
        self.assertFalse(record.is_active)

    def test_revoked_credential_cannot_be_refreshed_after_logout(self):
        self.client.post(url("logout"), {"refresh": self.tokens["refresh"]}, format="json")
        self.client.credentials()
        resp = self.client.post(url("refresh"), {"refresh": self.tokens["refresh"]}, format="json")
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_logout_is_idempotent(self):
        resp1 = self.client.post(url("logout"), {"refresh": self.tokens["refresh"]}, format="json")
        resp2 = self.client.post(url("logout"), {"refresh": self.tokens["refresh"]}, format="json")
        self.assertEqual(resp1.status_code, status.HTTP_204_NO_CONTENT)
        self.assertEqual(resp2.status_code, status.HTTP_204_NO_CONTENT)


# ---------------------------------------------------------------------------
# Current user / profile
# ---------------------------------------------------------------------------


class MeAPITests(AuthAPITestCase):
    def setUp(self):
        super().setUp()
        self.user = self.make_user(email="me@example.com", first_name="Me", last_name="User")
        self.tokens = self.authenticate_as(self.user)

    def test_get_me_authenticated_returns_200(self):
        resp = self.client.get(url("me"))
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertEqual(resp.json()["email"].lower(), "me@example.com")

    def test_get_me_unauthenticated_returns_401(self):
        self.client.credentials()
        resp = self.client.get(url("me"))
        self.assertIn(resp.status_code, (status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN))

    def test_get_me_never_returns_password(self):
        resp = self.client.get(url("me"))
        self.assertNotIn("password", resp.json())

    def test_get_me_never_returns_privileged_fields(self):
        resp = self.client.get(url("me"))
        for forbidden in ("is_staff", "is_superuser", "groups", "user_permissions"):
            self.assertNotIn(forbidden, resp.json())

    def test_patch_me_updates_first_name(self):
        resp = self.client.patch(url("me"), {"first_name": "Updated"}, format="json")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertEqual(resp.json()["first_name"], "Updated")

    def test_patch_me_unauthenticated_returns_401(self):
        self.client.credentials()
        resp = self.client.patch(url("me"), {"first_name": "X"}, format="json")
        self.assertIn(resp.status_code, (status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN))

    def test_patch_me_cannot_change_email(self):
        resp = self.client.patch(url("me"), {"email": "hijacked@example.com"}, format="json")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.user.refresh_from_db()
        self.assertEqual(self.user.email, "me@example.com")

    def test_patch_me_cannot_set_is_staff(self):
        self.client.patch(url("me"), {"is_staff": True}, format="json")
        self.user.refresh_from_db()
        self.assertFalse(self.user.is_staff)

    def test_patch_me_cannot_set_is_superuser(self):
        self.client.patch(url("me"), {"is_superuser": True}, format="json")
        self.user.refresh_from_db()
        self.assertFalse(self.user.is_superuser)

    def test_patch_me_cannot_set_is_active(self):
        self.client.patch(url("me"), {"is_active": False}, format="json")
        self.user.refresh_from_db()
        self.assertTrue(self.user.is_active)

    def test_patch_me_cannot_set_email_verified_at(self):
        self.client.patch(url("me"), {"email_verified_at": timezone.now().isoformat()}, format="json")
        self.user.refresh_from_db()
        self.assertIsNone(self.user.email_verified_at)

    def test_put_method_not_allowed(self):
        resp = self.client.put(url("me"), {"first_name": "X"}, format="json")
        self.assertEqual(resp.status_code, status.HTTP_405_METHOD_NOT_ALLOWED)

    def test_user_only_ever_sees_own_profile(self):
        other = self.make_user(email="other@example.com")
        resp = self.client.get(url("me"))
        self.assertNotEqual(resp.json()["id"], str(other.id))
        self.assertEqual(resp.json()["id"], str(self.user.id))


# ---------------------------------------------------------------------------
# Password change
# ---------------------------------------------------------------------------


class PasswordChangeAPITests(AuthAPITestCase):
    def setUp(self):
        super().setUp()
        self.user = self.make_user(email="pwchange@example.com")
        self.tokens = self.authenticate_as(self.user)

    def _change(self, **overrides):
        payload = {
            "old_password": VALID_PASSWORD,
            "new_password": "New-Str0ng-Pass!99",
            "new_password_confirm": "New-Str0ng-Pass!99",
        }
        payload.update(overrides)
        return self.client.post(url("password-change"), payload, format="json")

    def test_valid_change_returns_200(self):
        resp = self._change()
        self.assertEqual(resp.status_code, status.HTTP_200_OK)

    def test_wrong_current_password_returns_400(self):
        resp = self._change(old_password="wrong")
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_weak_new_password_returns_400(self):
        resp = self._change(new_password="weak", new_password_confirm="weak")
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_mismatched_new_password_returns_400(self):
        resp = self._change(new_password_confirm="Different-Pass!1")
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_unauthenticated_request_returns_401(self):
        self.client.credentials()
        resp = self._change()
        self.assertIn(resp.status_code, (status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN))

    def test_old_password_no_longer_works_after_change(self):
        self._change()
        resp = self.login("pwchange@example.com", password=VALID_PASSWORD)
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_new_password_works_after_change(self):
        self._change()
        resp = self.login("pwchange@example.com", password="New-Str0ng-Pass!99")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)

    def test_response_never_contains_password(self):
        resp = self._change()
        self.assertNotIn("new_password", resp.json())
        self.assertNotIn("password", resp.json())

    def test_active_sessions_revoked_after_password_change(self):
        record = RefreshToken.objects.get(token_hash=RefreshToken.hash_token(self.tokens["refresh"]))
        self._change()
        record.refresh_from_db()
        self.assertFalse(record.is_active)
        self.assertEqual(record.revoked_reason, RefreshTokenRevocationReason.PASSWORD_CHANGE)


# ---------------------------------------------------------------------------
# Password reset
# ---------------------------------------------------------------------------


class PasswordResetAPITests(AuthAPITestCase):
    def setUp(self):
        super().setUp()
        self.user = self.make_user(email="reset@example.com")

    @override_settings(DEBUG=True)
    def test_reset_request_for_known_email_returns_200_with_debug_token(self):
        resp = self.client.post(url("password-reset-request"), {"email": "reset@example.com"}, format="json")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertIn("debug_reset_token", resp.json())

    def test_reset_request_for_unknown_email_returns_identical_200(self):
        known_resp = self.client.post(
            url("password-reset-request"), {"email": "reset@example.com"}, format="json"
        )
        unknown_resp = self.client.post(
            url("password-reset-request"), {"email": "nobody@example.com"}, format="json"
        )
        self.assertEqual(known_resp.status_code, unknown_resp.status_code)
        self.assertEqual(known_resp.json()["detail"], unknown_resp.json()["detail"])

    @override_settings(DEBUG=False)
    def test_reset_request_never_exposes_token_in_production(self):
        resp = self.client.post(url("password-reset-request"), {"email": "reset@example.com"}, format="json")
        self.assertNotIn("debug_reset_token", resp.json())

    def _request_reset_token(self):
        with override_settings(DEBUG=True):
            resp = self.client.post(
                url("password-reset-request"), {"email": "reset@example.com"}, format="json"
            )
        return resp.json()["debug_reset_token"]

    def _confirm(self, token, **overrides):
        payload = {
            "token": token,
            "new_password": "New-Str0ng-Pass!99",
            "new_password_confirm": "New-Str0ng-Pass!99",
        }
        payload.update(overrides)
        return self.client.post(url("password-reset-confirm"), payload, format="json")

    def test_valid_token_confirms_reset(self):
        token = self._request_reset_token()
        resp = self._confirm(token)
        self.assertEqual(resp.status_code, status.HTTP_200_OK)

    def test_invalid_token_returns_400(self):
        resp = self._confirm("not-a-real-token")
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_expired_token_returns_400(self):
        token = self._request_reset_token()
        record = SecurityToken.objects.get(token_hash=SecurityToken.hash_token(token))
        record.expires_at = timezone.now() - timezone.timedelta(seconds=1)
        record.save(update_fields=["expires_at"])
        resp = self._confirm(token)
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_already_used_token_returns_400(self):
        token = self._request_reset_token()
        self._confirm(token)
        resp = self._confirm(token, new_password="Another-Pass!123", new_password_confirm="Another-Pass!123")
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_weak_password_returns_400(self):
        token = self._request_reset_token()
        resp = self._confirm(token, new_password="weak", new_password_confirm="weak")
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_mismatched_confirmation_returns_400(self):
        token = self._request_reset_token()
        resp = self._confirm(token, new_password_confirm="Different-Pass!1")
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_old_password_invalid_after_reset(self):
        token = self._request_reset_token()
        self._confirm(token)
        resp = self.login("reset@example.com", password=VALID_PASSWORD)
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_new_password_valid_after_reset(self):
        token = self._request_reset_token()
        self._confirm(token)
        resp = self.login("reset@example.com", password="New-Str0ng-Pass!99")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)

    def test_sessions_revoked_after_successful_reset(self):
        tokens = self.login("reset@example.com").json()
        record = RefreshToken.objects.get(token_hash=RefreshToken.hash_token(tokens["refresh"]))
        reset_token = self._request_reset_token()
        self._confirm(reset_token)
        record.refresh_from_db()
        self.assertFalse(record.is_active)


# ---------------------------------------------------------------------------
# Email verification
# ---------------------------------------------------------------------------


class EmailVerificationAPITests(AuthAPITestCase):
    def setUp(self):
        super().setUp()
        with override_settings(DEBUG=True):
            resp = self.register(email="verify@example.com")
        self.verification_token = resp.json()["debug_verification_token"]
        self.user = User.objects.get(email__iexact="verify@example.com")

    def test_valid_token_verifies_email(self):
        resp = self.client.post(url("email-verify"), {"token": self.verification_token}, format="json")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.user.refresh_from_db()
        self.assertTrue(self.user.is_email_verified)

    def test_invalid_token_returns_400(self):
        resp = self.client.post(url("email-verify"), {"token": "bogus-token"}, format="json")
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_already_used_token_returns_400(self):
        self.client.post(url("email-verify"), {"token": self.verification_token}, format="json")
        resp = self.client.post(url("email-verify"), {"token": self.verification_token}, format="json")
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_expired_token_returns_400(self):
        record = SecurityToken.objects.get(token_hash=SecurityToken.hash_token(self.verification_token))
        record.expires_at = timezone.now() - timezone.timedelta(seconds=1)
        record.save(update_fields=["expires_at"])
        resp = self.client.post(url("email-verify"), {"token": self.verification_token}, format="json")
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    @override_settings(DEBUG=True)
    def test_resend_for_unverified_account_issues_new_token(self):
        resp = self.client.post(url("email-resend"), {"email": "verify@example.com"}, format="json")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertIn("debug_verification_token", resp.json())

    @override_settings(DEBUG=True)
    def test_resend_for_already_verified_account_issues_no_token(self):
        self.client.post(url("email-verify"), {"token": self.verification_token}, format="json")
        resp = self.client.post(url("email-resend"), {"email": "verify@example.com"}, format="json")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertNotIn("debug_verification_token", resp.json())

    def test_resend_for_unknown_email_returns_identical_response(self):
        known = self.client.post(url("email-resend"), {"email": "verify@example.com"}, format="json")
        unknown = self.client.post(url("email-resend"), {"email": "ghost@example.com"}, format="json")
        self.assertEqual(known.json()["detail"], unknown.json()["detail"])

    def test_old_verification_token_invalidated_by_resend(self):
        with override_settings(DEBUG=True):
            self.client.post(url("email-resend"), {"email": "verify@example.com"}, format="json")
        resp = self.client.post(url("email-verify"), {"token": self.verification_token}, format="json")
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------


class SessionManagementAPITests(AuthAPITestCase):
    def setUp(self):
        super().setUp()
        self.user = self.make_user(email="sessions@example.com")
        self.other_user = self.make_user(email="other-sessions@example.com")
        self.tokens = self.authenticate_as(self.user)
        self.session_id = self.tokens["session_id"]

    def test_revoke_own_session_returns_204(self):
        resp = self.client.post(url("session-revoke"), {"session_id": self.session_id}, format="json")
        self.assertEqual(resp.status_code, status.HTTP_204_NO_CONTENT)
        record = RefreshToken.objects.get(pk=self.session_id)
        self.assertFalse(record.is_active)

    def test_revoke_nonexistent_session_returns_204_without_error(self):
        resp = self.client.post(
            url("session-revoke"), {"session_id": str(uuid.uuid4())}, format="json"
        )
        self.assertEqual(resp.status_code, status.HTTP_204_NO_CONTENT)

    def test_cannot_revoke_another_users_session(self):
        self.client.credentials()
        other_tokens = self.authenticate_as(self.other_user)
        # Switch back to the original user's credentials and try to revoke the other user's session.
        self.client.credentials(HTTP_AUTHORIZATION=f"Bearer {self.tokens['access']}")
        resp = self.client.post(
            url("session-revoke"), {"session_id": other_tokens["session_id"]}, format="json"
        )
        self.assertEqual(resp.status_code, status.HTTP_204_NO_CONTENT)  # no error, but no-op
        other_record = RefreshToken.objects.get(pk=other_tokens["session_id"])
        self.assertTrue(other_record.is_active)

    def test_revoke_requires_authentication(self):
        self.client.credentials()
        resp = self.client.post(url("session-revoke"), {"session_id": self.session_id}, format="json")
        self.assertIn(resp.status_code, (status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN))

    def test_revoked_session_cannot_be_refreshed(self):
        self.client.post(url("session-revoke"), {"session_id": self.session_id}, format="json")
        self.client.credentials()
        resp = self.client.post(url("refresh"), {"refresh": self.tokens["refresh"]}, format="json")
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_revoke_all_returns_204(self):
        resp = self.client.post(url("session-revoke-all"))
        self.assertEqual(resp.status_code, status.HTTP_204_NO_CONTENT)

    def test_revoke_all_revokes_every_active_session_for_caller(self):
        self.client.post(url("login"), {"email": "sessions@example.com", "password": VALID_PASSWORD}, format="json")
        self.assertEqual(RefreshToken.objects.filter(user=self.user, revoked_at__isnull=True).count(), 2)
        self.client.post(url("session-revoke-all"))
        self.assertEqual(RefreshToken.objects.filter(user=self.user, revoked_at__isnull=True).count(), 0)

    def test_revoke_all_does_not_touch_other_users_sessions(self):
        self.client.credentials()
        other_tokens = self.authenticate_as(self.other_user)
        self.client.credentials(HTTP_AUTHORIZATION=f"Bearer {self.tokens['access']}")
        self.client.post(url("session-revoke-all"))
        other_record = RefreshToken.objects.get(pk=other_tokens["session_id"])
        self.assertTrue(other_record.is_active)

    def test_revoke_all_requires_authentication(self):
        self.client.credentials()
        resp = self.client.post(url("session-revoke-all"))
        self.assertIn(resp.status_code, (status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN))


# ---------------------------------------------------------------------------
# permissions.py — direct unit tests
# ---------------------------------------------------------------------------
# None of these three classes are currently imported by any view in
# views.py (confirmed by reading views.py/permissions.py: every existing
# view uses DRF's own AllowAny/IsAuthenticated directly). They are,
# however, real, reusable, non-trivial authorization logic — exercised
# here directly against a bare `rest_framework.request.Request` /
# `APIView` rather than through an HTTP round trip, since there is no
# endpoint wiring them in to test through yet.


class PermissionClassesTests(APITestCase):
    """Direct unit tests for apps.authentication.permissions.*."""

    def setUp(self):
        super().setUp()
        self.user = User.objects.create_user(email="owner@example.com", password=VALID_PASSWORD)
        self.other_user = User.objects.create_user(email="other2@example.com", password=VALID_PASSWORD)

    def _drf_request(self, django_request, user):
        from rest_framework.request import Request

        request = Request(django_request)
        request.user = user
        return request

    def test_is_account_owner_allows_access_to_own_user_object(self):
        from django.test import RequestFactory

        from apps.authentication.permissions import IsAccountOwner

        request = self._drf_request(RequestFactory().get("/"), self.user)
        self.assertTrue(IsAccountOwner().has_object_permission(request, None, self.user))

    def test_is_account_owner_denies_access_to_another_users_object(self):
        from django.test import RequestFactory

        from apps.authentication.permissions import IsAccountOwner

        request = self._drf_request(RequestFactory().get("/"), self.user)
        self.assertFalse(IsAccountOwner().has_object_permission(request, None, self.other_user))

    def test_is_account_owner_checks_user_fk_on_related_objects(self):
        from django.test import RequestFactory

        from apps.authentication.permissions import IsAccountOwner

        token_record = RefreshToken.objects.create(
            user=self.user,
            token_hash="a" * 64,
            expires_at=timezone.now() + timezone.timedelta(days=1),
        )
        owner_request = self._drf_request(RequestFactory().get("/"), self.user)
        other_request = self._drf_request(RequestFactory().get("/"), self.other_user)

        self.assertTrue(IsAccountOwner().has_object_permission(owner_request, None, token_record))
        self.assertFalse(IsAccountOwner().has_object_permission(other_request, None, token_record))

    def test_is_verified_user_denies_unverified_account(self):
        from django.test import RequestFactory

        from apps.authentication.permissions import IsVerifiedUser

        request = self._drf_request(RequestFactory().get("/"), self.user)
        self.assertFalse(IsVerifiedUser().has_permission(request, None))

    def test_is_verified_user_allows_verified_account(self):
        from django.test import RequestFactory

        from apps.authentication.permissions import IsVerifiedUser

        self.user.mark_email_verified()
        request = self._drf_request(RequestFactory().get("/"), self.user)
        self.assertTrue(IsVerifiedUser().has_permission(request, None))

    def test_is_staff_or_superuser_denies_regular_user(self):
        from django.test import RequestFactory

        from apps.authentication.permissions import IsStaffOrSuperuser

        request = self._drf_request(RequestFactory().get("/"), self.user)
        self.assertFalse(IsStaffOrSuperuser().has_permission(request, None))

    def test_is_staff_or_superuser_allows_staff_user(self):
        from django.test import RequestFactory

        from apps.authentication.permissions import IsStaffOrSuperuser

        self.user.is_staff = True
        self.user.save(update_fields=["is_staff"])
        request = self._drf_request(RequestFactory().get("/"), self.user)
        self.assertTrue(IsStaffOrSuperuser().has_permission(request, None))

    def test_is_staff_or_superuser_denies_anonymous_user(self):
        from django.contrib.auth.models import AnonymousUser
        from django.test import RequestFactory

        from apps.authentication.permissions import IsStaffOrSuperuser

        request = self._drf_request(RequestFactory().get("/"), AnonymousUser())
        self.assertFalse(IsStaffOrSuperuser().has_permission(request, None))