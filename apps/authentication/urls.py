"""
apps.authentication.urls
==========================

Self-contained authentication routes. Include from the project root as:

    path("api/v1/auth/", include("apps.authentication.urls"))
"""

from django.urls import path

from apps.authentication.views import (
    EmailResendView,
    EmailVerifyView,
    LoginView,
    LogoutView,
    MeView,
    PasswordChangeView,
    PasswordResetConfirmView,
    PasswordResetRequestView,
    RefreshView,
    RegisterView,
    SessionRevokeAllView,
    SessionRevokeView,
)

app_name = "authentication"

urlpatterns = [
    path("register/", RegisterView.as_view(), name="register"),
    path("login/", LoginView.as_view(), name="login"),
    path("logout/", LogoutView.as_view(), name="logout"),
    path("refresh/", RefreshView.as_view(), name="refresh"),
    path("me/", MeView.as_view(), name="me"),
    path("password/change/", PasswordChangeView.as_view(), name="password-change"),
    path("password/reset/request/", PasswordResetRequestView.as_view(), name="password-reset-request"),
    path("password/reset/confirm/", PasswordResetConfirmView.as_view(), name="password-reset-confirm"),
    path("email/verify/", EmailVerifyView.as_view(), name="email-verify"),
    path("email/resend/", EmailResendView.as_view(), name="email-resend"),
    path("sessions/revoke/", SessionRevokeView.as_view(), name="session-revoke"),
    path("sessions/revoke-all/", SessionRevokeAllView.as_view(), name="session-revoke-all"),
]