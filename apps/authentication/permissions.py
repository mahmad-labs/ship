"""
apps.authentication.permissions
=================================

DRF permission classes for the authentication API.

Scope discipline: permissions answer "is this already-authenticated
requester allowed to do this?" — never "how do we authenticate them?"
(that's `authenticate()`/`JWTAuthentication`/services.py) and never
"perform the operation" (that's services.py). Every class below is
small, deterministic, and side-effect free: no database writes, no
token issuance, no email, no session revocation, no business logic.

Most endpoints need nothing beyond DRF's own `AllowAny`/`IsAuthenticated`
— `views.py` already uses those directly and correctly:
    AllowAny:       register, login, refresh, password reset
                     request/confirm, email verify/resend
    IsAuthenticated: me (GET/PATCH), password change, logout,
                     session revoke/revoke-all

Ownership on session endpoints (`sessions/revoke/`) is currently
enforced in `views.py` via queryset scoping
(`RefreshToken.objects.filter(user=request.user, ...)`), which is a
correct and standard DRF idiom — not a gap. `IsAccountOwner` below
formalizes that same rule as a reusable, testable object-level
permission for anywhere a view fetches an object via `get_object()`
instead of pre-filtering a queryset (not currently done by any
existing view, since none of them use `get_object()` on a
`RefreshToken`/`SecurityToken` — provided so a future view built that
way doesn't have to reinvent the check, and so the rule is documented
in exactly one place).

`IsVerifiedUser` and `IsStaffOrSuperuser` are provided for the same
reason: real rules the architecture will need (a future write endpoint
gated on verified email; a future administrative account-management
endpoint using `services.deactivate_user`/`reactivate_user`, which
currently have no URL at all) — not yet imported by any existing view,
since none currently need them. Not wiring them in preemptively avoids
guessing at how a not-yet-built endpoint should be shaped.
"""

from __future__ import annotations

from typing import Any

from rest_framework.permissions import BasePermission
from rest_framework.request import Request
from rest_framework.views import APIView


class IsAccountOwner(BasePermission):
    """
    Object-level check: the requester must own the object being
    accessed. Works for objects with a `user` foreign key
    (`RefreshToken`, `SecurityToken`) and for the `User` object itself
    (`obj == request.user`).

    Never trusts a client-supplied user id — ownership is always
    determined by comparing against `request.user`, which DRF has
    already resolved from the authenticated request.
    """

    def has_object_permission(self, request: Request, view: APIView, obj: Any) -> bool:
        owner = getattr(obj, "user", obj)
        return bool(request.user and request.user.is_authenticated and owner == request.user)


class IsVerifiedUser(BasePermission):
    """Requires an authenticated request from a user whose email is verified."""

    message = "This action requires a verified email address."

    def has_permission(self, request: Request, view: APIView) -> bool:
        user = request.user
        return bool(user and user.is_authenticated and user.is_email_verified)


class IsStaffOrSuperuser(BasePermission):
    """Requires an authenticated staff or superuser account. For future administrative endpoints."""

    def has_permission(self, request: Request, view: APIView) -> bool:
        user = request.user
        return bool(user and user.is_authenticated and (user.is_staff or user.is_superuser))