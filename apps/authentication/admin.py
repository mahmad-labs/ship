"""
apps.authentication.admin
==========================

Admin configuration for all concrete authentication models: User,
SecurityToken, RefreshToken.

Security records (SecurityToken, RefreshToken) are never manually
creatable or directly field-editable here — they are issued and
consumed by the future service layer. The admin only allows narrow,
explicit operational actions (revoke / invalidate) rather than free
field editing or casual deletion.
"""

from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.contrib.auth.forms import UserChangeForm as BaseUserChangeForm
from django.contrib.auth.forms import UserCreationForm as BaseUserCreationForm
from django.utils import timezone
from django.utils.translation import gettext_lazy as _

from apps.authentication.models import (
    RefreshToken,
    RefreshTokenRevocationReason,
    SecurityToken,
    User,
)


# ---------------------------------------------------------------------------
# User
# ---------------------------------------------------------------------------


class UserCreationForm(BaseUserCreationForm):
    """Admin "add user" form for the email-only custom User model."""

    class Meta(BaseUserCreationForm.Meta):
        model = User
        fields = ("email",)


class UserChangeForm(BaseUserChangeForm):
    """Admin "edit user" form for the email-only custom User model."""

    class Meta(BaseUserChangeForm.Meta):
        model = User
        fields = "__all__"


@admin.register(User)
class UserAdmin(BaseUserAdmin):
    add_form = UserCreationForm
    form = UserChangeForm
    model = User

    ordering = ("email",)
    list_display = (
        "email",
        "first_name",
        "last_name",
        "is_staff",
        "is_active",
        "is_superuser",
        "is_email_verified",
        "is_locked",
        "created_at",
    )
    list_filter = ("is_staff", "is_superuser", "is_active")
    search_fields = ("email", "first_name", "last_name")
    filter_horizontal = ("groups", "user_permissions")

    readonly_fields = (
        "id",
        "last_login",
        "created_at",
        "updated_at",
        "password_changed_at",
        "email_verified_at",
        "failed_login_attempts",
        "locked_until",
    )

    fieldsets = (
        (None, {"fields": ("email", "password")}),
        (_("Personal info"), {"fields": ("first_name", "last_name")}),
        (
            _("Permissions"),
            {
                "fields": (
                    "is_active",
                    "is_staff",
                    "is_superuser",
                    "groups",
                    "user_permissions",
                )
            },
        ),
        (
            _("Account security"),
            {
                "fields": (
                    "email_verified_at",
                    "failed_login_attempts",
                    "locked_until",
                    "password_changed_at",
                )
            },
        ),
        (_("Important dates"), {"fields": ("last_login", "created_at", "updated_at")}),
        (_("Identity"), {"fields": ("id",)}),
    )

    add_fieldsets = (
        (
            None,
            {
                "classes": ("wide",),
                "fields": ("email", "password1", "password2", "is_staff", "is_active"),
            },
        ),
    )

    @admin.display(boolean=True, description="Email verified")
    def is_email_verified(self, obj: User) -> bool:
        return obj.is_email_verified

    @admin.display(boolean=True, description="Locked")
    def is_locked(self, obj: User) -> bool:
        return obj.is_locked


# ---------------------------------------------------------------------------
# SecurityToken
# ---------------------------------------------------------------------------


@admin.register(SecurityToken)
class SecurityTokenAdmin(admin.ModelAdmin):
    """
    Operational visibility into outstanding verification/reset tokens.
    `token_hash` is intentionally never displayed — even though it is a
    one-way digest and not the raw secret, there is no legitimate admin
    need to see it, so it is simply excluded.
    """

    date_hierarchy = "created_at"
    ordering = ("-created_at",)

    list_display = ("user", "purpose", "created_at", "expires_at", "used_at", "is_valid")
    list_filter = ("purpose",)
    search_fields = ("user__email",)
    autocomplete_fields = ("user",)

    exclude = ("token_hash",)
    readonly_fields = (
        "id",
        "user",
        "purpose",
        "created_at",
        "updated_at",
        "expires_at",
        "used_at",
    )

    actions = ["invalidate_selected"]

    @admin.display(boolean=True, description="Valid")
    def is_valid(self, obj: SecurityToken) -> bool:
        return obj.is_valid

    def has_add_permission(self, request) -> bool:
        return False

    def has_delete_permission(self, request, obj=None) -> bool:
        return False

    @admin.action(description="Invalidate selected tokens")
    def invalidate_selected(self, request, queryset):
        updated = queryset.filter(used_at__isnull=True).update(used_at=timezone.now())
        self.message_user(request, f"Invalidated {updated} token(s).")


# ---------------------------------------------------------------------------
# RefreshToken
# ---------------------------------------------------------------------------


@admin.register(RefreshToken)
class RefreshTokenAdmin(admin.ModelAdmin):
    """
    Operational visibility into active/revoked API sessions, with a
    single safe mutation path: the "Revoke selected" action. Direct
    field editing is disabled — `token_hash` is excluded entirely, and
    revocation always goes through `revoke()`-equivalent logic below
    rather than letting an operator hand-edit `revoked_at`.
    """

    date_hierarchy = "created_at"
    ordering = ("-created_at",)

    list_display = (
        "user",
        "family_id",
        "ip_address",
        "created_at",
        "expires_at",
        "revoked_at",
        "revoked_reason",
        "is_active",
    )
    list_filter = ("revoked_reason",)
    search_fields = ("user__email", "ip_address")
    autocomplete_fields = ("user",)

    exclude = ("token_hash",)
    readonly_fields = (
        "id",
        "user",
        "family_id",
        "user_agent",
        "ip_address",
        "created_at",
        "updated_at",
        "expires_at",
        "revoked_at",
        "revoked_reason",
    )

    actions = ["revoke_selected"]

    @admin.display(boolean=True, description="Active")
    def is_active(self, obj: RefreshToken) -> bool:
        return obj.is_active

    def has_add_permission(self, request) -> bool:
        return False

    def has_delete_permission(self, request, obj=None) -> bool:
        return False

    @admin.action(description="Revoke selected refresh tokens")
    def revoke_selected(self, request, queryset):
        updated = queryset.filter(revoked_at__isnull=True).update(
            revoked_at=timezone.now(),
            revoked_reason=RefreshTokenRevocationReason.ADMIN_REVOKED,
        )
        self.message_user(request, f"Revoked {updated} token(s).")