"""
apps.authentication.tests.test_serializers
=============================================

Tests against the real, unmodified `apps.authentication.serializers`.
Every serializer defined there is covered. Uses `django.test.TestCase`
with `rest_framework.test.APIRequestFactory` where a `request` context
is required (LoginSerializer, PasswordChangeSerializer).
"""

from django.contrib.auth.hashers import is_password_usable
from django.test import TestCase
from django.utils import timezone
from rest_framework.test import APIRequestFactory

from apps.authentication.models import (
    RefreshToken,
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

VALID_PASSWORD = "Str0ng-Pass!2024"


def make_user(email="user@example.com", password=VALID_PASSWORD, **extra):
    return User.objects.create_user(email=email, password=password, **extra)


def make_request(user=None):
    request = APIRequestFactory().post("/")
    request.user = user
    return request


# ---------------------------------------------------------------------------
# RegistrationSerializer
# ---------------------------------------------------------------------------


class RegistrationSerializerTests(TestCase):
    def _payload(self, **overrides):
        payload = {
            "email": "newuser@example.com",
            "password": VALID_PASSWORD,
            "password_confirm": VALID_PASSWORD,
            "first_name": "New",
            "last_name": "User",
        }
        payload.update(overrides)
        return payload

    def test_valid_registration_creates_user(self):
        serializer = RegistrationSerializer(data=self._payload())
        self.assertTrue(serializer.is_valid(), serializer.errors)
        user = serializer.save()
        self.assertEqual(user.email, "newuser@example.com")
        self.assertTrue(user.check_password(VALID_PASSWORD))

    def test_missing_email_is_invalid(self):
        payload = self._payload()
        del payload["email"]
        serializer = RegistrationSerializer(data=payload)
        self.assertFalse(serializer.is_valid())
        self.assertIn("email", serializer.errors)

    def test_missing_password_is_invalid(self):
        payload = self._payload()
        del payload["password"]
        serializer = RegistrationSerializer(data=payload)
        self.assertFalse(serializer.is_valid())
        self.assertIn("password", serializer.errors)

    def test_invalid_email_format_rejected(self):
        serializer = RegistrationSerializer(data=self._payload(email="not-an-email"))
        self.assertFalse(serializer.is_valid())
        self.assertIn("email", serializer.errors)

    def test_duplicate_email_rejected(self):
        make_user(email="taken@example.com")
        serializer = RegistrationSerializer(data=self._payload(email="taken@example.com"))
        self.assertFalse(serializer.is_valid())
        self.assertIn("email", serializer.errors)

    def test_case_insensitive_duplicate_email_rejected(self):
        make_user(email="taken@example.com")
        serializer = RegistrationSerializer(data=self._payload(email="TAKEN@EXAMPLE.com"))
        self.assertFalse(serializer.is_valid())
        self.assertIn("email", serializer.errors)

    def test_password_confirmation_mismatch_rejected(self):
        serializer = RegistrationSerializer(
            data=self._payload(password_confirm="Different-Pass!99")
        )
        self.assertFalse(serializer.is_valid())
        self.assertIn("password_confirm", serializer.errors)

    def test_weak_password_rejected_by_django_validators(self):
        serializer = RegistrationSerializer(
            data=self._payload(password="weak", password_confirm="weak")
        )
        self.assertFalse(serializer.is_valid())
        self.assertIn("password", serializer.errors)

    def test_password_fields_are_write_only(self):
        serializer = RegistrationSerializer(data=self._payload())
        serializer.is_valid()
        user = serializer.save()
        output = RegistrationSerializer(user).data
        self.assertNotIn("password", output)
        self.assertNotIn("password_confirm", output)

    def test_is_staff_cannot_be_injected(self):
        serializer = RegistrationSerializer(data=self._payload(is_staff=True))
        serializer.is_valid()
        user = serializer.save()
        self.assertFalse(user.is_staff)

    def test_is_superuser_cannot_be_injected(self):
        serializer = RegistrationSerializer(data=self._payload(is_superuser=True))
        serializer.is_valid()
        user = serializer.save()
        self.assertFalse(user.is_superuser)

    def test_groups_field_not_accepted(self):
        """`groups` isn't in `Meta.fields` at all, so it's silently ignored, not applied."""
        serializer = RegistrationSerializer(data=self._payload(groups=[1, 2, 3]))
        self.assertTrue(serializer.is_valid(), serializer.errors)
        user = serializer.save()
        self.assertEqual(user.groups.count(), 0)

    def test_email_verified_at_cannot_be_injected(self):
        serializer = RegistrationSerializer(
            data=self._payload(email_verified_at=timezone.now().isoformat())
        )
        serializer.is_valid()
        user = serializer.save()
        self.assertIsNone(user.email_verified_at)

    def test_created_user_has_hashed_password_not_plaintext(self):
        serializer = RegistrationSerializer(data=self._payload())
        serializer.is_valid()
        user = serializer.save()
        self.assertTrue(is_password_usable(user.password))
        self.assertNotEqual(user.password, VALID_PASSWORD)


# ---------------------------------------------------------------------------
# LoginSerializer
# ---------------------------------------------------------------------------


class LoginSerializerTests(TestCase):
    def setUp(self):
        self.user = make_user(email="login@example.com")

    def test_valid_credentials_authenticate(self):
        serializer = LoginSerializer(
            data={"email": "login@example.com", "password": VALID_PASSWORD},
            context={"request": make_request()},
        )
        self.assertTrue(serializer.is_valid(), serializer.errors)
        self.assertEqual(serializer.validated_data["user"], self.user)

    def test_case_insensitive_email_authenticates(self):
        serializer = LoginSerializer(
            data={"email": "LOGIN@EXAMPLE.com", "password": VALID_PASSWORD},
            context={"request": make_request()},
        )
        self.assertTrue(serializer.is_valid(), serializer.errors)

    def test_incorrect_password_rejected(self):
        serializer = LoginSerializer(
            data={"email": "login@example.com", "password": "wrong-password"},
            context={"request": make_request()},
        )
        self.assertFalse(serializer.is_valid())

    def test_nonexistent_account_rejected(self):
        serializer = LoginSerializer(
            data={"email": "nobody@example.com", "password": VALID_PASSWORD},
            context={"request": make_request()},
        )
        self.assertFalse(serializer.is_valid())

    def test_nonexistent_account_and_wrong_password_give_identical_error(self):
        """Enumeration-safety: the two failure modes must not be distinguishable."""
        s1 = LoginSerializer(
            data={"email": "nobody@example.com", "password": VALID_PASSWORD},
            context={"request": make_request()},
        )
        s2 = LoginSerializer(
            data={"email": "login@example.com", "password": "wrong-password"},
            context={"request": make_request()},
        )
        s1.is_valid()
        s2.is_valid()
        self.assertEqual(s1.errors["non_field_errors"], s2.errors["non_field_errors"])

    def test_inactive_account_rejected(self):
        self.user.is_active = False
        self.user.save(update_fields=["is_active"])
        serializer = LoginSerializer(
            data={"email": "login@example.com", "password": VALID_PASSWORD},
            context={"request": make_request()},
        )
        self.assertFalse(serializer.is_valid())

    def test_locked_account_rejected_even_with_correct_password(self):
        self.user.locked_until = timezone.now() + timezone.timedelta(minutes=15)
        self.user.save(update_fields=["locked_until"])
        serializer = LoginSerializer(
            data={"email": "login@example.com", "password": VALID_PASSWORD},
            context={"request": make_request()},
        )
        self.assertFalse(serializer.is_valid())

    def test_failed_attempt_increments_counter_on_matching_account(self):
        LoginSerializer(
            data={"email": "login@example.com", "password": "wrong"},
            context={"request": make_request()},
        ).is_valid()
        self.user.refresh_from_db()
        self.assertEqual(self.user.failed_login_attempts, 1)

    def test_failed_attempt_against_unknown_email_creates_no_user(self):
        LoginSerializer(
            data={"email": "ghost@example.com", "password": "wrong"},
            context={"request": make_request()},
        ).is_valid()
        self.assertFalse(User.objects.filter(email__iexact="ghost@example.com").exists())

    def test_password_never_appears_in_validated_data_output(self):
        serializer = LoginSerializer(
            data={"email": "login@example.com", "password": VALID_PASSWORD},
            context={"request": make_request()},
        )
        serializer.is_valid()
        # `password` is write_only, so it must not surface via to_representation.
        self.assertNotIn("password", serializer.data)


# ---------------------------------------------------------------------------
# TokenResponseSerializer
# ---------------------------------------------------------------------------


class TokenResponseSerializerTests(TestCase):
    def test_serializes_expected_safe_fields_only(self):
        import uuid

        data = {
            "access": "fake.jwt.token",
            "access_expires_at": timezone.now(),
            "refresh": "raw-refresh-value",
            "refresh_expires_at": timezone.now(),
            "session_id": uuid.uuid4(),
        }
        serializer = TokenResponseSerializer(data)
        output = serializer.data
        self.assertEqual(set(output.keys()), {
            "access", "access_expires_at", "refresh", "refresh_expires_at", "session_id"
        })
        self.assertNotIn("token_hash", output)


# ---------------------------------------------------------------------------
# RefreshSerializer
# ---------------------------------------------------------------------------


class RefreshSerializerTests(TestCase):
    def setUp(self):
        self.user = make_user(email="refresh@example.com")

    def _make_token(self, raw="raw-refresh-token-value", **overrides):
        defaults = dict(
            user=self.user,
            token_hash=RefreshToken.hash_token(raw),
            expires_at=timezone.now() + timezone.timedelta(days=30),
        )
        defaults.update(overrides)
        return RefreshToken.objects.create(**defaults)

    def test_valid_active_refresh_token_accepted(self):
        self._make_token(raw="valid-token")
        serializer = RefreshSerializer(data={"refresh": "valid-token"})
        self.assertTrue(serializer.is_valid(), serializer.errors)
        self.assertEqual(serializer.validated_data["refresh_record"].user, self.user)

    def test_unknown_refresh_token_rejected(self):
        serializer = RefreshSerializer(data={"refresh": "does-not-exist"})
        self.assertFalse(serializer.is_valid())

    def test_expired_refresh_token_rejected(self):
        self._make_token(raw="expired-token", expires_at=timezone.now() - timezone.timedelta(seconds=1))
        serializer = RefreshSerializer(data={"refresh": "expired-token"})
        self.assertFalse(serializer.is_valid())

    def test_revoked_refresh_token_rejected(self):
        token = self._make_token(raw="revoked-token")
        from apps.authentication.models import RefreshTokenRevocationReason

        token.revoke(RefreshTokenRevocationReason.LOGOUT)
        serializer = RefreshSerializer(data={"refresh": "revoked-token"})
        self.assertFalse(serializer.is_valid())

    def test_reuse_of_revoked_token_revokes_entire_family(self):
        """Security-critical: presenting an already-rotated token nukes the whole session family."""
        from apps.authentication.models import RefreshTokenRevocationReason

        token = self._make_token(raw="rotated-away-token")
        sibling = RefreshToken.objects.create(
            user=self.user,
            token_hash=RefreshToken.hash_token("sibling-token"),
            family_id=token.family_id,
            expires_at=timezone.now() + timezone.timedelta(days=30),
        )
        token.revoke(RefreshTokenRevocationReason.ROTATED)

        serializer = RefreshSerializer(data={"refresh": "rotated-away-token"})
        self.assertFalse(serializer.is_valid())

        sibling.refresh_from_db()
        self.assertFalse(sibling.is_active)
        self.assertEqual(sibling.revoked_reason, RefreshTokenRevocationReason.REUSE_DETECTED)

    def test_refresh_token_for_inactive_user_rejected(self):
        self._make_token(raw="inactive-user-token")
        self.user.is_active = False
        self.user.save(update_fields=["is_active"])
        serializer = RefreshSerializer(data={"refresh": "inactive-user-token"})
        self.assertFalse(serializer.is_valid())

    def test_no_internal_token_hash_leaked_in_errors_or_validated_data(self):
        token = self._make_token(raw="leak-check-token")
        serializer = RefreshSerializer(data={"refresh": "leak-check-token"})
        serializer.is_valid()
        self.assertNotIn(token.token_hash, str(serializer.validated_data))


# ---------------------------------------------------------------------------
# LogoutSerializer / SessionRevokeSerializer (input shape only — behavior lives in views)
# ---------------------------------------------------------------------------


class LogoutAndSessionSerializerTests(TestCase):
    def test_logout_serializer_requires_refresh_field(self):
        serializer = LogoutSerializer(data={})
        self.assertFalse(serializer.is_valid())
        self.assertIn("refresh", serializer.errors)

    def test_session_revoke_serializer_requires_valid_uuid(self):
        serializer = SessionRevokeSerializer(data={"session_id": "not-a-uuid"})
        self.assertFalse(serializer.is_valid())

    def test_session_revoke_serializer_accepts_valid_uuid(self):
        import uuid

        serializer = SessionRevokeSerializer(data={"session_id": str(uuid.uuid4())})
        self.assertTrue(serializer.is_valid(), serializer.errors)


# ---------------------------------------------------------------------------
# UserSerializer (profile)
# ---------------------------------------------------------------------------


class UserSerializerTests(TestCase):
    def setUp(self):
        self.user = make_user(email="profile@example.com", first_name="Old", last_name="Name")

    def test_output_excludes_password(self):
        data = UserSerializer(self.user).data
        self.assertNotIn("password", data)

    def test_output_excludes_privilege_and_security_fields(self):
        data = UserSerializer(self.user).data
        for forbidden in ("is_staff", "is_superuser", "groups", "user_permissions",
                          "failed_login_attempts", "locked_until", "password_changed_at"):
            self.assertNotIn(forbidden, data)

    def test_output_includes_expected_safe_fields(self):
        data = UserSerializer(self.user).data
        for expected in ("id", "email", "first_name", "last_name", "is_active",
                          "email_verified_at", "created_at", "updated_at"):
            self.assertIn(expected, data)

    def test_first_name_and_last_name_are_writable(self):
        serializer = UserSerializer(
            self.user, data={"first_name": "New", "last_name": "Name2"}, partial=True
        )
        self.assertTrue(serializer.is_valid(), serializer.errors)
        updated = serializer.save()
        self.assertEqual(updated.first_name, "New")

    def test_email_is_read_only(self):
        serializer = UserSerializer(self.user, data={"email": "hijack@example.com"}, partial=True)
        serializer.is_valid()
        updated = serializer.save()
        self.assertEqual(updated.email, "profile@example.com")

    def test_is_active_is_read_only(self):
        serializer = UserSerializer(self.user, data={"is_active": False}, partial=True)
        serializer.is_valid()
        updated = serializer.save()
        self.assertTrue(updated.is_active)

    def test_is_staff_cannot_be_set_via_update(self):
        serializer = UserSerializer(self.user, data={"is_staff": True}, partial=True)
        serializer.is_valid()
        updated = serializer.save()
        self.assertFalse(updated.is_staff)

    def test_email_verified_at_is_read_only(self):
        serializer = UserSerializer(
            self.user, data={"email_verified_at": timezone.now().isoformat()}, partial=True
        )
        serializer.is_valid()
        updated = serializer.save()
        self.assertIsNone(updated.email_verified_at)

    def test_invalid_update_rejects_overlong_first_name(self):
        serializer = UserSerializer(self.user, data={"first_name": "x" * 200}, partial=True)
        self.assertFalse(serializer.is_valid())


# ---------------------------------------------------------------------------
# PasswordChangeSerializer
# ---------------------------------------------------------------------------


class PasswordChangeSerializerTests(TestCase):
    def setUp(self):
        self.user = make_user(email="pwchange@example.com")

    def _serializer(self, **data):
        payload = {
            "old_password": VALID_PASSWORD,
            "new_password": "New-Str0ng-Pass!99",
            "new_password_confirm": "New-Str0ng-Pass!99",
        }
        payload.update(data)
        return PasswordChangeSerializer(data=payload, context={"request": make_request(self.user)})

    def test_correct_current_password_accepted(self):
        serializer = self._serializer()
        self.assertTrue(serializer.is_valid(), serializer.errors)

    def test_incorrect_current_password_rejected(self):
        serializer = self._serializer(old_password="wrong-current")
        self.assertFalse(serializer.is_valid())
        self.assertIn("old_password", serializer.errors)

    def test_weak_new_password_rejected(self):
        serializer = self._serializer(new_password="weak", new_password_confirm="weak")
        self.assertFalse(serializer.is_valid())
        self.assertIn("new_password", serializer.errors)

    def test_new_password_mismatch_rejected(self):
        serializer = self._serializer(new_password_confirm="Different-Pass!1")
        self.assertFalse(serializer.is_valid())
        self.assertIn("new_password_confirm", serializer.errors)

    def test_password_never_returned_in_output(self):
        serializer = self._serializer()
        serializer.is_valid()
        self.assertNotIn("new_password", serializer.data)
        self.assertNotIn("old_password", serializer.data)


# ---------------------------------------------------------------------------
# PasswordResetRequestSerializer / PasswordResetConfirmSerializer
# ---------------------------------------------------------------------------


class PasswordResetSerializerTests(TestCase):
    def setUp(self):
        self.user = make_user(email="reset@example.com")

    def test_reset_request_accepts_known_email(self):
        serializer = PasswordResetRequestSerializer(data={"email": "reset@example.com"})
        self.assertTrue(serializer.is_valid())

    def test_reset_request_accepts_unknown_email(self):
        """Only email FORMAT is validated — existence is deliberately not checked here (enumeration-safety)."""
        serializer = PasswordResetRequestSerializer(data={"email": "unknown@example.com"})
        self.assertTrue(serializer.is_valid())

    def test_reset_request_rejects_malformed_email(self):
        serializer = PasswordResetRequestSerializer(data={"email": "not-an-email"})
        self.assertFalse(serializer.is_valid())

    def _make_reset_token(self, raw="reset-token-raw-value", **overrides):
        defaults = dict(
            user=self.user,
            purpose=SecurityTokenPurpose.PASSWORD_RESET,
            token_hash=SecurityToken.hash_token(raw),
            expires_at=timezone.now() + timezone.timedelta(hours=1),
        )
        defaults.update(overrides)
        return SecurityToken.objects.create(**defaults)

    def test_confirm_with_valid_token_succeeds(self):
        self._make_reset_token(raw="good-reset-token")
        serializer = PasswordResetConfirmSerializer(data={
            "token": "good-reset-token",
            "new_password": "New-Str0ng-Pass!99",
            "new_password_confirm": "New-Str0ng-Pass!99",
        })
        self.assertTrue(serializer.is_valid(), serializer.errors)

    def test_confirm_with_invalid_token_rejected(self):
        serializer = PasswordResetConfirmSerializer(data={
            "token": "no-such-token",
            "new_password": "New-Str0ng-Pass!99",
            "new_password_confirm": "New-Str0ng-Pass!99",
        })
        self.assertFalse(serializer.is_valid())
        self.assertIn("token", serializer.errors)

    def test_confirm_with_expired_token_rejected(self):
        self._make_reset_token(raw="expired-reset", expires_at=timezone.now() - timezone.timedelta(seconds=1))
        serializer = PasswordResetConfirmSerializer(data={
            "token": "expired-reset",
            "new_password": "New-Str0ng-Pass!99",
            "new_password_confirm": "New-Str0ng-Pass!99",
        })
        self.assertFalse(serializer.is_valid())

    def test_confirm_with_already_used_token_rejected(self):
        token = self._make_reset_token(raw="used-reset")
        token.mark_used()
        serializer = PasswordResetConfirmSerializer(data={
            "token": "used-reset",
            "new_password": "New-Str0ng-Pass!99",
            "new_password_confirm": "New-Str0ng-Pass!99",
        })
        self.assertFalse(serializer.is_valid())

    def test_confirm_password_mismatch_rejected(self):
        self._make_reset_token(raw="mismatch-reset")
        serializer = PasswordResetConfirmSerializer(data={
            "token": "mismatch-reset",
            "new_password": "New-Str0ng-Pass!99",
            "new_password_confirm": "Different-Pass!1",
        })
        self.assertFalse(serializer.is_valid())
        self.assertIn("new_password_confirm", serializer.errors)

    def test_confirm_weak_password_rejected(self):
        self._make_reset_token(raw="weak-reset")
        serializer = PasswordResetConfirmSerializer(data={
            "token": "weak-reset",
            "new_password": "weak",
            "new_password_confirm": "weak",
        })
        self.assertFalse(serializer.is_valid())
        self.assertIn("new_password", serializer.errors)

    def test_confirm_output_never_exposes_password_or_token(self):
        self._make_reset_token(raw="output-check-reset")
        serializer = PasswordResetConfirmSerializer(data={
            "token": "output-check-reset",
            "new_password": "New-Str0ng-Pass!99",
            "new_password_confirm": "New-Str0ng-Pass!99",
        })
        serializer.is_valid()
        self.assertNotIn("new_password", serializer.data)
        self.assertNotIn("token", serializer.data)


# ---------------------------------------------------------------------------
# EmailVerifySerializer / EmailResendSerializer
# ---------------------------------------------------------------------------


class EmailVerifySerializerTests(TestCase):
    def setUp(self):
        self.user = make_user(email="verify@example.com")

    def _make_token(self, raw="verify-token-raw", **overrides):
        defaults = dict(
            user=self.user,
            purpose=SecurityTokenPurpose.EMAIL_VERIFICATION,
            token_hash=SecurityToken.hash_token(raw),
            expires_at=timezone.now() + timezone.timedelta(hours=24),
        )
        defaults.update(overrides)
        return SecurityToken.objects.create(**defaults)

    def test_valid_token_accepted(self):
        self._make_token(raw="good-verify")
        serializer = EmailVerifySerializer(data={"token": "good-verify"})
        self.assertTrue(serializer.is_valid(), serializer.errors)
        self.assertEqual(serializer.validated_data["security_token"].user, self.user)

    def test_invalid_token_rejected(self):
        serializer = EmailVerifySerializer(data={"token": "no-such-token"})
        self.assertFalse(serializer.is_valid())

    def test_expired_token_rejected(self):
        self._make_token(raw="expired-verify", expires_at=timezone.now() - timezone.timedelta(seconds=1))
        serializer = EmailVerifySerializer(data={"token": "expired-verify"})
        self.assertFalse(serializer.is_valid())

    def test_already_used_token_rejected(self):
        token = self._make_token(raw="used-verify")
        token.mark_used()
        serializer = EmailVerifySerializer(data={"token": "used-verify"})
        self.assertFalse(serializer.is_valid())

    def test_password_reset_purpose_token_rejected_for_email_verification(self):
        """A token issued for a different purpose must not validate here."""
        self._make_token(raw="wrong-purpose", purpose=SecurityTokenPurpose.PASSWORD_RESET)
        serializer = EmailVerifySerializer(data={"token": "wrong-purpose"})
        self.assertFalse(serializer.is_valid())


class EmailResendSerializerTests(TestCase):
    def test_accepts_well_formed_email_regardless_of_existence(self):
        serializer = EmailResendSerializer(data={"email": "whoever@example.com"})
        self.assertTrue(serializer.is_valid())

    def test_rejects_malformed_email(self):
        serializer = EmailResendSerializer(data={"email": "not-an-email"})
        self.assertFalse(serializer.is_valid())