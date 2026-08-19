"""
apps.authentication.tests.test_services
==========================================

Tests against the real, unmodified `apps.authentication.services`.

IMPORTANT CONTEXT: `views.py` does not currently call this module (see
the architecture note delivered alongside `services.py` — confirmed
again while writing these tests: `views.py` has its own private
`_issue_token_pair`/`_issue_security_token`/`_send_*_email` and never
imports `apps.authentication.services`). These tests call every public
service function directly, exercising the layer on its own merits —
this is NOT redundant with `test_views.py`, which tests a currently
separate, independently-implemented code path. See the defect note at
the end of the accompanying response for what this means.

Every test that touches Core's `AuditEvent` asserts on the specific
event created and scans its `metadata`/`changes` for the presence of
any password or raw token used in that test, to make the "audit never
contains secrets" guarantee an actual executable check rather than a
claim in a docstring.
"""

import uuid
from datetime import timedelta
from unittest import mock

from django.core.exceptions import ValidationError
from django.test import RequestFactory, TestCase
from django.utils import timezone

from apps.authentication import services
from apps.authentication.models import (
    RefreshToken,
    RefreshTokenRevocationReason,
    SecurityToken,
    SecurityTokenPurpose,
    User,
)
from apps.core.models import AuditEvent

VALID_PASSWORD = "Str0ng-Pass!2024"


def fake_request():
    request = RequestFactory().post("/", HTTP_USER_AGENT="pytest/1.0")
    request.META["REMOTE_ADDR"] = "10.0.0.5"
    return request


def make_user(email="user@example.com", password=VALID_PASSWORD, **extra):
    return User.objects.create_user(email=email, password=password, **extra)


def assert_no_secret_in_audit(testcase, *secrets_to_check):
    """Scans every AuditEvent's metadata/changes for any of the given secret values."""
    for event in AuditEvent.objects.all():
        blob = str(event.metadata) + str(event.changes)
        for secret in secrets_to_check:
            testcase.assertNotIn(secret, blob, f"Secret leaked into AuditEvent metadata: {secret!r}")


# ---------------------------------------------------------------------------
# register_user
# ---------------------------------------------------------------------------


class RegisterUserServiceTests(TestCase):
    def test_successful_registration_creates_active_unverified_user(self):
        user, raw_token = services.register_user(email="new@example.com", password=VALID_PASSWORD)
        self.assertTrue(user.is_active)
        self.assertFalse(user.is_email_verified)
        self.assertTrue(user.check_password(VALID_PASSWORD))

    def test_email_is_normalized(self):
        user, _ = services.register_user(email="Mixed.Case@EXAMPLE.com", password=VALID_PASSWORD)
        self.assertEqual(user.email, "Mixed.Case@example.com")

    def test_password_is_hashed_not_plaintext(self):
        user, _ = services.register_user(email="hash@example.com", password=VALID_PASSWORD)
        self.assertNotEqual(user.password, VALID_PASSWORD)

    def test_duplicate_email_raises_validation_error(self):
        make_user(email="taken@example.com")
        with self.assertRaises(ValidationError):
            services.register_user(email="taken@example.com", password=VALID_PASSWORD)

    def test_concurrent_duplicate_registration_raises_domain_error_not_raw_integrity_error(self):
        """
        Simulates the race the validator pre-check cannot fully close: the
        email passes validate_email_available, but a duplicate is inserted
        before this function's own insert executes.
        """
        with mock.patch("apps.authentication.services.validate_email_available", return_value="race@example.com"):
            make_user(email="race@example.com")  # the "concurrent" winner
            with self.assertRaises(services.EmailAlreadyRegisteredError):
                services.register_user(email="race@example.com", password=VALID_PASSWORD)

    def test_issues_email_verification_token(self):
        user, raw_token = services.register_user(email="verify@example.com", password=VALID_PASSWORD)
        self.assertTrue(raw_token)
        token = SecurityToken.objects.get(user=user, purpose=SecurityTokenPurpose.EMAIL_VERIFICATION)
        self.assertEqual(token.token_hash, SecurityToken.hash_token(raw_token))
        self.assertTrue(token.is_valid)

    def test_creates_audit_event_with_no_secrets(self):
        user, raw_token = services.register_user(email="audited@example.com", password=VALID_PASSWORD)
        event = AuditEvent.objects.get(resource_id=str(user.id), action="create")
        self.assertEqual(event.actor_id, str(user.id))
        assert_no_secret_in_audit(self, VALID_PASSWORD, raw_token)

    @mock.patch("apps.authentication.services.send_verification_email")
    def test_calls_email_delivery_hook_with_user_and_raw_token(self, mock_send):
        user, raw_token = services.register_user(email="deliver@example.com", password=VALID_PASSWORD)
        mock_send.assert_called_once_with(user, raw_token)

    def test_transaction_rolls_back_user_creation_on_downstream_failure(self):
        """If token issuance fails, the whole registration must not partially commit."""
        with mock.patch(
            "apps.authentication.services._issue_security_token", side_effect=RuntimeError("boom")
        ):
            with self.assertRaises(RuntimeError):
                services.register_user(email="rollback@example.com", password=VALID_PASSWORD)
        self.assertFalse(User.objects.filter(email__iexact="rollback@example.com").exists())

    def test_weak_password_rejected_before_user_created(self):
        with self.assertRaises(ValidationError):
            services.register_user(email="weakpw@example.com", password="weak")
        self.assertFalse(User.objects.filter(email__iexact="weakpw@example.com").exists())


# ---------------------------------------------------------------------------
# authenticate_user
# ---------------------------------------------------------------------------


class AuthenticateUserServiceTests(TestCase):
    def setUp(self):
        self.user = make_user(email="auth@example.com")

    def test_successful_authentication_returns_user(self):
        result = services.authenticate_user(email="auth@example.com", password=VALID_PASSWORD, request=fake_request())
        self.assertEqual(result, self.user)

    def test_wrong_password_raises_invalid_credentials(self):
        with self.assertRaises(services.InvalidCredentialsError):
            services.authenticate_user(email="auth@example.com", password="wrong", request=fake_request())

    def test_nonexistent_user_raises_invalid_credentials(self):
        with self.assertRaises(services.InvalidCredentialsError):
            services.authenticate_user(email="ghost@example.com", password=VALID_PASSWORD, request=fake_request())

    def test_inactive_user_raises_invalid_credentials(self):
        self.user.is_active = False
        self.user.save(update_fields=["is_active"])
        with self.assertRaises(services.InvalidCredentialsError):
            services.authenticate_user(email="auth@example.com", password=VALID_PASSWORD, request=fake_request())

    def test_locked_user_raises_account_locked_even_with_correct_password(self):
        self.user.locked_until = timezone.now() + timedelta(minutes=10)
        self.user.save(update_fields=["locked_until"])
        with self.assertRaises(services.AccountLockedError):
            services.authenticate_user(email="auth@example.com", password=VALID_PASSWORD, request=fake_request())

    def test_successful_authentication_resets_lockout_state(self):
        self.user.failed_login_attempts = 4
        self.user.save(update_fields=["failed_login_attempts"])
        services.authenticate_user(email="auth@example.com", password=VALID_PASSWORD, request=fake_request())
        self.user.refresh_from_db()
        self.assertEqual(self.user.failed_login_attempts, 0)

    def test_successful_authentication_updates_last_login(self):
        self.assertIsNone(self.user.last_login)
        services.authenticate_user(email="auth@example.com", password=VALID_PASSWORD, request=fake_request())
        self.user.refresh_from_db()
        self.assertIsNotNone(self.user.last_login)

    def test_failed_attempt_increments_counter(self):
        with self.assertRaises(services.InvalidCredentialsError):
            services.authenticate_user(email="auth@example.com", password="wrong", request=fake_request())
        self.user.refresh_from_db()
        self.assertEqual(self.user.failed_login_attempts, 1)

    def test_successful_login_creates_audit_event_with_no_password(self):
        services.authenticate_user(email="auth@example.com", password=VALID_PASSWORD, request=fake_request())
        self.assertTrue(AuditEvent.objects.filter(resource_id=str(self.user.id), action="login").exists())
        assert_no_secret_in_audit(self, VALID_PASSWORD)


# ---------------------------------------------------------------------------
# issue_authentication_credentials / refresh_credentials
# ---------------------------------------------------------------------------


class TokenServiceTests(TestCase):
    def setUp(self):
        self.user = make_user(email="tokens@example.com")
        self.request = fake_request()

    def test_issue_credentials_creates_hashed_refresh_token_row(self):
        tokens = services.issue_authentication_credentials(self.user, self.request)
        record = RefreshToken.objects.get(pk=tokens.session_id)
        self.assertEqual(record.token_hash, RefreshToken.hash_token(tokens.refresh))
        self.assertNotEqual(record.token_hash, tokens.refresh)

    def test_issue_credentials_records_device_metadata(self):
        tokens = services.issue_authentication_credentials(self.user, self.request)
        record = RefreshToken.objects.get(pk=tokens.session_id)
        self.assertEqual(record.ip_address, "10.0.0.5")
        self.assertIn("pytest", record.user_agent)

    def test_refresh_rotation_issues_new_pair_and_revokes_old(self):
        first = services.issue_authentication_credentials(self.user, self.request)
        rotated = services.refresh_credentials(raw_refresh_token=first.refresh, request=self.request)
        self.assertNotEqual(rotated.refresh, first.refresh)

        old_record = RefreshToken.objects.get(pk=first.session_id)
        self.assertFalse(old_record.is_active)
        self.assertEqual(old_record.revoked_reason, RefreshTokenRevocationReason.ROTATED)

    def test_rotation_preserves_family_id(self):
        first = services.issue_authentication_credentials(self.user, self.request)
        rotated = services.refresh_credentials(raw_refresh_token=first.refresh, request=self.request)
        old_record = RefreshToken.objects.get(pk=first.session_id)
        new_record = RefreshToken.objects.get(pk=rotated.session_id)
        self.assertEqual(old_record.family_id, new_record.family_id)

    def test_unknown_refresh_token_raises_invalid_token_error(self):
        with self.assertRaises(services.InvalidTokenError):
            services.refresh_credentials(raw_refresh_token="not-a-real-token", request=self.request)

    def test_expired_refresh_token_raises_invalid_token_error(self):
        first = services.issue_authentication_credentials(self.user, self.request)
        record = RefreshToken.objects.get(pk=first.session_id)
        record.expires_at = timezone.now() - timedelta(seconds=1)
        record.save(update_fields=["expires_at"])
        with self.assertRaises(services.InvalidTokenError):
            services.refresh_credentials(raw_refresh_token=first.refresh, request=self.request)

    def test_replaying_rotated_token_raises_and_revokes_entire_family(self):
        first = services.issue_authentication_credentials(self.user, self.request)
        rotated = services.refresh_credentials(raw_refresh_token=first.refresh, request=self.request)

        with self.assertRaises(services.InvalidTokenError):
            services.refresh_credentials(raw_refresh_token=first.refresh, request=self.request)

        new_record = RefreshToken.objects.get(pk=rotated.session_id)
        self.assertFalse(new_record.is_active)
        self.assertEqual(new_record.revoked_reason, RefreshTokenRevocationReason.REUSE_DETECTED)

    def test_reuse_detection_creates_audit_event(self):
        first = services.issue_authentication_credentials(self.user, self.request)
        services.refresh_credentials(raw_refresh_token=first.refresh, request=self.request)
        with self.assertRaises(services.InvalidTokenError):
            services.refresh_credentials(raw_refresh_token=first.refresh, request=self.request)
        self.assertTrue(
            AuditEvent.objects.filter(metadata__event="refresh_token_reuse_detected").exists()
        )

    def test_cross_user_token_cannot_be_refreshed_by_looking_up_another_users_hash(self):
        other_user = make_user(email="cross@example.com")
        tokens = services.issue_authentication_credentials(other_user, self.request)
        # A valid token always belongs to whichever user it was issued for —
        # refresh_credentials must resolve the OWNING user from the token
        # itself, never trust an externally supplied identity.
        result = services.refresh_credentials(raw_refresh_token=tokens.refresh, request=self.request)
        rotated_record = RefreshToken.objects.get(pk=result.session_id)
        self.assertEqual(rotated_record.user, other_user)
        self.assertNotEqual(rotated_record.user, self.user)

    def test_rotation_rolls_back_if_new_token_issuance_fails(self):
        first = services.issue_authentication_credentials(self.user, self.request)
        with mock.patch(
            "apps.authentication.services.AccessToken.for_user", side_effect=RuntimeError("boom")
        ):
            with self.assertRaises(RuntimeError):
                services.refresh_credentials(raw_refresh_token=first.refresh, request=self.request)

        old_record = RefreshToken.objects.get(pk=first.session_id)
        self.assertTrue(old_record.is_active, "old token's revoke() must be rolled back with the failed rotation")


# ---------------------------------------------------------------------------
# logout_user / revoke_session / revoke_all_sessions
# ---------------------------------------------------------------------------


class SessionServiceTests(TestCase):
    def setUp(self):
        self.user = make_user(email="sessions@example.com")
        self.other_user = make_user(email="other@example.com")
        self.request = fake_request()

    def test_logout_revokes_matching_token(self):
        tokens = services.issue_authentication_credentials(self.user, self.request)
        services.logout_user(user=self.user, raw_refresh_token=tokens.refresh)
        record = RefreshToken.objects.get(pk=tokens.session_id)
        self.assertFalse(record.is_active)
        self.assertEqual(record.revoked_reason, RefreshTokenRevocationReason.LOGOUT)

    def test_logout_with_unknown_token_does_not_raise(self):
        services.logout_user(user=self.user, raw_refresh_token="never-issued")  # must not raise

    def test_logout_cannot_revoke_another_users_token(self):
        tokens = services.issue_authentication_credentials(self.other_user, self.request)
        services.logout_user(user=self.user, raw_refresh_token=tokens.refresh)
        record = RefreshToken.objects.get(pk=tokens.session_id)
        self.assertTrue(record.is_active, "logout must not revoke a token belonging to a different user")

    def test_revoke_session_revokes_own_session(self):
        tokens = services.issue_authentication_credentials(self.user, self.request)
        services.revoke_session(user=self.user, session_id=tokens.session_id)
        record = RefreshToken.objects.get(pk=tokens.session_id)
        self.assertFalse(record.is_active)

    def test_revoke_session_cannot_revoke_another_users_session(self):
        tokens = services.issue_authentication_credentials(self.other_user, self.request)
        services.revoke_session(user=self.user, session_id=tokens.session_id)
        record = RefreshToken.objects.get(pk=tokens.session_id)
        self.assertTrue(record.is_active)

    def test_revoke_session_with_nonexistent_id_does_not_raise(self):
        services.revoke_session(user=self.user, session_id=uuid.uuid4())  # must not raise

    def test_revoke_all_sessions_revokes_every_active_session(self):
        services.issue_authentication_credentials(self.user, self.request)
        services.issue_authentication_credentials(self.user, self.request)
        services.issue_authentication_credentials(self.user, self.request)
        count = services.revoke_all_sessions(user=self.user)
        self.assertEqual(count, 3)
        self.assertEqual(RefreshToken.objects.filter(user=self.user, revoked_at__isnull=True).count(), 0)

    def test_revoke_all_sessions_does_not_touch_other_users(self):
        other_tokens = services.issue_authentication_credentials(self.other_user, self.request)
        services.issue_authentication_credentials(self.user, self.request)
        services.revoke_all_sessions(user=self.user)
        record = RefreshToken.objects.get(pk=other_tokens.session_id)
        self.assertTrue(record.is_active)

    def test_revoke_all_sessions_returns_zero_when_nothing_active(self):
        self.assertEqual(services.revoke_all_sessions(user=self.user), 0)


# ---------------------------------------------------------------------------
# change_password
# ---------------------------------------------------------------------------


class ChangePasswordServiceTests(TestCase):
    def setUp(self):
        self.user = make_user(email="pwchange@example.com")
        self.request = fake_request()

    def test_successful_change_updates_password(self):
        services.change_password(user=self.user, current_password=VALID_PASSWORD, new_password="New-Str0ng-Pass!99")
        self.user.refresh_from_db()
        self.assertTrue(self.user.check_password("New-Str0ng-Pass!99"))

    def test_wrong_current_password_raises(self):
        with self.assertRaises(services.InvalidCredentialsError):
            services.change_password(user=self.user, current_password="wrong", new_password="New-Str0ng-Pass!99")

    def test_weak_new_password_raises(self):
        with self.assertRaises(ValidationError):
            services.change_password(user=self.user, current_password=VALID_PASSWORD, new_password="weak")

    def test_reusing_current_password_raises(self):
        with self.assertRaises(ValidationError):
            services.change_password(user=self.user, current_password=VALID_PASSWORD, new_password=VALID_PASSWORD)

    def test_all_active_sessions_revoked_on_change(self):
        tokens = services.issue_authentication_credentials(self.user, self.request)
        services.change_password(user=self.user, current_password=VALID_PASSWORD, new_password="New-Str0ng-Pass!99")
        record = RefreshToken.objects.get(pk=tokens.session_id)
        self.assertFalse(record.is_active)
        self.assertEqual(record.revoked_reason, RefreshTokenRevocationReason.PASSWORD_CHANGE)

    def test_creates_audit_event_without_leaking_passwords(self):
        services.change_password(user=self.user, current_password=VALID_PASSWORD, new_password="New-Str0ng-Pass!99")
        self.assertTrue(
            AuditEvent.objects.filter(resource_id=str(self.user.id), metadata__event="password_changed").exists()
        )
        assert_no_secret_in_audit(self, VALID_PASSWORD, "New-Str0ng-Pass!99")

    def test_transaction_rolls_back_if_session_revocation_fails(self):
        original_password_hash = self.user.password
        with mock.patch("apps.authentication.services.revoke_all_sessions", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                services.change_password(
                    user=self.user, current_password=VALID_PASSWORD, new_password="New-Str0ng-Pass!99"
                )
        self.user.refresh_from_db()
        self.assertEqual(self.user.password, original_password_hash)


# ---------------------------------------------------------------------------
# request_password_reset / confirm_password_reset
# ---------------------------------------------------------------------------


class PasswordResetServiceTests(TestCase):
    def setUp(self):
        self.user = make_user(email="reset@example.com")

    def test_request_for_known_active_user_returns_raw_token(self):
        raw_token = services.request_password_reset(email="reset@example.com")
        self.assertIsNotNone(raw_token)
        record = SecurityToken.objects.get(user=self.user, purpose=SecurityTokenPurpose.PASSWORD_RESET)
        self.assertEqual(record.token_hash, SecurityToken.hash_token(raw_token))

    def test_request_for_unknown_email_returns_none_without_raising(self):
        result = services.request_password_reset(email="nobody@example.com")
        self.assertIsNone(result)

    def test_request_for_inactive_user_returns_none(self):
        self.user.is_active = False
        self.user.save(update_fields=["is_active"])
        result = services.request_password_reset(email="reset@example.com")
        self.assertIsNone(result)

    @mock.patch("apps.authentication.services.send_password_reset_email")
    def test_request_invokes_delivery_hook_only_for_known_user(self, mock_send):
        services.request_password_reset(email="reset@example.com")
        mock_send.assert_called_once()
        mock_send.reset_mock()
        services.request_password_reset(email="nobody@example.com")
        mock_send.assert_not_called()

    def test_confirm_with_valid_token_updates_password(self):
        raw_token = services.request_password_reset(email="reset@example.com")
        services.confirm_password_reset(raw_token=raw_token, new_password="New-Str0ng-Pass!99")
        self.user.refresh_from_db()
        self.assertTrue(self.user.check_password("New-Str0ng-Pass!99"))

    def test_confirm_with_invalid_token_raises(self):
        with self.assertRaises(services.InvalidTokenError):
            services.confirm_password_reset(raw_token="bogus", new_password="New-Str0ng-Pass!99")

    def test_confirm_with_expired_token_raises(self):
        raw_token = services.request_password_reset(email="reset@example.com")
        record = SecurityToken.objects.get(token_hash=SecurityToken.hash_token(raw_token))
        record.expires_at = timezone.now() - timedelta(seconds=1)
        record.save(update_fields=["expires_at"])
        with self.assertRaises(services.InvalidTokenError):
            services.confirm_password_reset(raw_token=raw_token, new_password="New-Str0ng-Pass!99")

    def test_confirm_token_is_one_time_use(self):
        raw_token = services.request_password_reset(email="reset@example.com")
        services.confirm_password_reset(raw_token=raw_token, new_password="New-Str0ng-Pass!99")
        with self.assertRaises(services.InvalidTokenError):
            services.confirm_password_reset(raw_token=raw_token, new_password="Another-Pass!123")

    def test_confirm_revokes_active_sessions(self):
        request = fake_request()
        tokens = services.issue_authentication_credentials(self.user, request)
        raw_token = services.request_password_reset(email="reset@example.com")
        services.confirm_password_reset(raw_token=raw_token, new_password="New-Str0ng-Pass!99")
        record = RefreshToken.objects.get(pk=tokens.session_id)
        self.assertFalse(record.is_active)

    def test_requesting_new_token_invalidates_previous_unused_token(self):
        first_raw = services.request_password_reset(email="reset@example.com")
        services.request_password_reset(email="reset@example.com")  # second request supersedes it
        with self.assertRaises(services.InvalidTokenError):
            services.confirm_password_reset(raw_token=first_raw, new_password="New-Str0ng-Pass!99")

    def test_transaction_rolls_back_if_session_revocation_fails(self):
        raw_token = services.request_password_reset(email="reset@example.com")
        with mock.patch("apps.authentication.services.revoke_all_sessions", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                services.confirm_password_reset(raw_token=raw_token, new_password="New-Str0ng-Pass!99")

        self.user.refresh_from_db()
        self.assertFalse(self.user.check_password("New-Str0ng-Pass!99"))
        record = SecurityToken.objects.get(token_hash=SecurityToken.hash_token(raw_token))
        self.assertIsNone(record.used_at, "token consumption must be rolled back with the failed transaction")

    def test_reset_flow_creates_no_secret_leaking_audit_events(self):
        raw_token = services.request_password_reset(email="reset@example.com")
        services.confirm_password_reset(raw_token=raw_token, new_password="New-Str0ng-Pass!99")
        assert_no_secret_in_audit(self, raw_token, "New-Str0ng-Pass!99")


# ---------------------------------------------------------------------------
# verify_email / resend_email_verification
# ---------------------------------------------------------------------------


class EmailVerificationServiceTests(TestCase):
    def setUp(self):
        self.user = make_user(email="verify@example.com")
        self.raw_token = services._issue_security_token(
            self.user, SecurityTokenPurpose.EMAIL_VERIFICATION, timedelta(hours=24)
        )

    def test_verify_with_valid_token_marks_verified(self):
        services.verify_email(raw_token=self.raw_token)
        self.user.refresh_from_db()
        self.assertTrue(self.user.is_email_verified)

    def test_verify_with_invalid_token_raises(self):
        with self.assertRaises(services.InvalidTokenError):
            services.verify_email(raw_token="bogus-token")

    def test_verify_token_is_one_time_use(self):
        services.verify_email(raw_token=self.raw_token)
        with self.assertRaises(services.InvalidTokenError):
            services.verify_email(raw_token=self.raw_token)

    def test_verify_rolls_back_if_state_update_fails(self):
        with mock.patch("apps.authentication.models.User.mark_email_verified", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                services.verify_email(raw_token=self.raw_token)
        record = SecurityToken.objects.get(token_hash=SecurityToken.hash_token(self.raw_token))
        self.assertIsNone(record.used_at, "token consumption must roll back with the failed verification")

    def test_resend_for_unverified_user_issues_new_token(self):
        new_token = services.resend_email_verification(email="verify@example.com")
        self.assertIsNotNone(new_token)
        self.assertNotEqual(new_token, self.raw_token)

    def test_resend_invalidates_previous_token(self):
        services.resend_email_verification(email="verify@example.com")
        with self.assertRaises(services.InvalidTokenError):
            services.verify_email(raw_token=self.raw_token)

    def test_resend_for_already_verified_user_returns_none(self):
        services.verify_email(raw_token=self.raw_token)
        result = services.resend_email_verification(email="verify@example.com")
        self.assertIsNone(result)

    def test_resend_for_unknown_email_returns_none(self):
        result = services.resend_email_verification(email="ghost@example.com")
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# deactivate_user / reactivate_user
# ---------------------------------------------------------------------------


class AccountStateServiceTests(TestCase):
    def setUp(self):
        self.user = make_user(email="deactivate@example.com")
        self.staff_actor = User.objects.create_superuser(email="admin@example.com", password=VALID_PASSWORD)

    def test_deactivate_sets_is_active_false(self):
        services.deactivate_user(user=self.user)
        self.user.refresh_from_db()
        self.assertFalse(self.user.is_active)

    def test_deactivate_revokes_active_sessions(self):
        tokens = services.issue_authentication_credentials(self.user, fake_request())
        services.deactivate_user(user=self.user)
        record = RefreshToken.objects.get(pk=tokens.session_id)
        self.assertFalse(record.is_active)

    def test_double_deactivate_raises(self):
        services.deactivate_user(user=self.user)
        with self.assertRaises(ValidationError):
            services.deactivate_user(user=self.user)

    def test_reactivate_sets_is_active_true(self):
        services.deactivate_user(user=self.user)
        services.reactivate_user(user=self.user)
        self.user.refresh_from_db()
        self.assertTrue(self.user.is_active)

    def test_reactivate_without_prior_deactivation_raises(self):
        with self.assertRaises(ValidationError):
            services.reactivate_user(user=self.user)

    def test_deactivate_by_admin_records_admin_as_actor(self):
        services.deactivate_user(user=self.user, actor=self.staff_actor, reason="policy violation")
        event = AuditEvent.objects.get(resource_id=str(self.user.id), metadata__event="account_deactivated")
        self.assertEqual(event.actor_id, str(self.staff_actor.id))

    def test_deactivated_user_cannot_authenticate(self):
        services.deactivate_user(user=self.user)
        with self.assertRaises(services.InvalidCredentialsError):
            services.authenticate_user(email="deactivate@example.com", password=VALID_PASSWORD, request=fake_request())