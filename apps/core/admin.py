"""
apps.core.admin
================

Admin registration for Core's concrete, platform-level models.

Only concrete database tables are registered here:
    - AuditEvent
    - IdempotencyRecord
    - ProcessedEvent

Abstract infrastructure (UUIDModel, TimeStampedModel, BaseModel,
SoftDeleteModel and its managers/querysets) has no database table and
is intentionally not registered.

AuditEvent is append-only/immutable at the model layer (see
apps.core.models.AuditEvent.save/delete). The admin configuration below
reinforces that at the UI layer: no add, no delete, and every field is
read-only, so AuditEvent is effectively a read-only audit viewer.
"""

from django.contrib import admin

from apps.core.models import AuditEvent, IdempotencyRecord, ProcessedEvent


@admin.register(AuditEvent)
class AuditEventAdmin(admin.ModelAdmin):
    """
    Read-only viewer for the immutable audit trail.

    No add/change/delete is offered — AuditEvent rows are created only
    by application code, never through the admin, and must never be
    edited or removed here.
    """

    date_hierarchy = "created_at"
    ordering = ("-created_at",)

    list_display = (
        "created_at",
        "action",
        "actor_type",
        "actor_id",
        "resource_type",
        "resource_id",
        "correlation_id",
    )
    list_filter = ("action", "actor_type")
    search_fields = (
        "actor_id",
        "resource_type",
        "resource_id",
        "correlation_id",
    )

    readonly_fields = (
        "id",
        "actor_type",
        "actor_id",
        "action",
        "resource_type",
        "resource_id",
        "changes",
        "metadata",
        "correlation_id",
        "created_at",
    )
    fields = readonly_fields  # explicit detail-view field order

    def has_add_permission(self, request) -> bool:
        return False

    def has_change_permission(self, request, obj=None) -> bool:
        # False disables saving; view access is still governed by
        # Django's default has_view_permission, so records remain
        # inspectable without being editable.
        return False

    def has_delete_permission(self, request, obj=None) -> bool:
        return False


@admin.register(IdempotencyRecord)
class IdempotencyRecordAdmin(admin.ModelAdmin):
    """Operational visibility into idempotent-request tracking."""

    date_hierarchy = "created_at"
    ordering = ("-created_at",)

    list_display = (
        "scope",
        "key",
        "status",
        "response_status",
        "expires_at",
        "created_at",
    )
    list_filter = ("status", "scope")
    search_fields = ("scope", "key")

    readonly_fields = ("id", "created_at", "updated_at")


@admin.register(ProcessedEvent)
class ProcessedEventAdmin(admin.ModelAdmin):
    """Operational visibility into inbound webhook/event de-duplication."""

    date_hierarchy = "created_at"
    ordering = ("-created_at",)

    list_display = (
        "source",
        "external_event_id",
        "event_type",
        "status",
        "processed_at",
        "created_at",
    )
    list_filter = ("status", "source")
    search_fields = ("source", "external_event_id", "event_type")

    readonly_fields = ("id", "created_at", "updated_at")