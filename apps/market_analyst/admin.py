"""
apps.market_analyst.admin
============================

Django admin registrations for Ship's Market Analyst models.

Mirrors the data-integrity line drawn in `models.py`: opportunity
scores, analysis results, and lifecycle `status` are *derived*
intelligence produced by the analysis pipeline, not user input. The
admin therefore exposes them as read-only wherever a human could
otherwise hand-edit a value that's supposed to come only from
`agents.market_analyst.MarketAnalystAgent` / the Celery tasks that
call it. Operators can still act on opportunities (approve, reject,
archive) through the read/write fields that were always meant to be
human-controlled -- just not by typing a fake score into a form.
"""

from __future__ import annotations

from django.contrib import admin

from apps.market_analyst.models import (
    Market,
    MarketAnalysis,
    ProductOpportunity,
)

# ---------------------------------------------------------------------------
# Market
# ---------------------------------------------------------------------------


@admin.register(Market)
class MarketAdmin(admin.ModelAdmin):
    list_display = ("name", "country_code", "region", "currency", "is_active", "opportunity_count", "created_at")
    list_filter = ("is_active", "country_code")
    search_fields = ("name", "country_code", "region")
    ordering = ("name",)
    readonly_fields = ("id", "created_at", "updated_at")

    fieldsets = (
        (None, {"fields": ("name", "country_code", "region", "currency", "is_active")}),
        ("Metadata", {"fields": ("id", "created_at", "updated_at"), "classes": ("collapse",)}),
    )

    @admin.display(description="Opportunities")
    def opportunity_count(self, obj: Market) -> int:
        return obj.opportunities.count()

    def get_queryset(self, request):
        return super().get_queryset(request).prefetch_related("opportunities")


# ---------------------------------------------------------------------------
# Product opportunity
# ---------------------------------------------------------------------------


class MarketAnalysisInline(admin.TabularInline):
    """
    Read-only inline showing an opportunity's analysis history without
    leaving the ProductOpportunity change page. Every field here is a
    pipeline output (see MarketAnalysisAdmin below for why), so the
    inline is display-only -- `has_add_permission` is disabled to make
    that explicit rather than relying on `can_delete=False` alone.
    """

    model = MarketAnalysis
    fields = ("created_at", "recommendation", "confidence_score", "analysis_version", "source", "summary")
    readonly_fields = fields
    extra = 0
    can_delete = False
    show_change_link = True
    ordering = ("-created_at",)

    def has_add_permission(self, request, obj=None) -> bool:
        return False


# Fields the future analysis pipeline owns; never hand-editable here.
_DERIVED_SCORE_FIELDS = (
    "demand_score",
    "trend_score",
    "margin_score",
    "competition_score",
    "saturation_score",
    "overall_score",
)


@admin.register(ProductOpportunity)
class ProductOpportunityAdmin(admin.ModelAdmin):
    list_display = (
        "name",
        "market",
        "category",
        "status",
        "overall_score",
        "is_scored",
        "submitted_by",
        "created_at",
    )
    list_filter = ("status", "market", "category")
    search_fields = ("name", "category", "description", "market__name")
    autocomplete_fields = ("market", "submitted_by")
    ordering = ("-created_at",)
    date_hierarchy = "created_at"
    inlines = (MarketAnalysisInline,)

    readonly_fields = ("id", "created_at", "updated_at", "is_scored", *_DERIVED_SCORE_FIELDS)

    fieldsets = (
        (None, {"fields": ("market", "submitted_by", "name", "category", "description", "status")}),
        (
            "Derived scores (pipeline-only, read-only)",
            {
                "fields": (*_DERIVED_SCORE_FIELDS, "is_scored"),
                "description": (
                    "Populated by the analysis pipeline (MarketAnalystAgent via the "
                    "Celery tasks). Not editable here -- see the module docstring on "
                    "models.py for why user input never writes to these fields."
                ),
            },
        ),
        ("Metadata", {"fields": ("id", "created_at", "updated_at"), "classes": ("collapse",)}),
    )

    @admin.display(boolean=True, description="Scored")
    def is_scored(self, obj: ProductOpportunity) -> bool:
        return obj.is_scored

    def get_queryset(self, request):
        return super().get_queryset(request).select_related("market", "submitted_by")


# ---------------------------------------------------------------------------
# Market analysis
# ---------------------------------------------------------------------------


@admin.register(MarketAnalysis)
class MarketAnalysisAdmin(admin.ModelAdmin):
    """
    Every `MarketAnalysis` row is an immutable pipeline output (see the
    model's own docstring: an opportunity accumulates a history of
    these rather than any one row being edited over time). The admin
    is intentionally view-only -- no add, no change, no delete -- so
    operators can inspect analysis history without anyone hand-editing
    a recorded result after the fact.
    """

    list_display = (
        "opportunity",
        "recommendation",
        "confidence_score",
        "analysis_version",
        "source",
        "created_at",
    )
    list_filter = ("recommendation", "analysis_version", "source")
    search_fields = ("opportunity__name", "summary", "source")
    autocomplete_fields = ("opportunity",)
    ordering = ("-created_at",)
    date_hierarchy = "created_at"

    readonly_fields = (
        "id",
        "opportunity",
        "summary",
        "recommendation",
        "confidence_score",
        "structured_data",
        "analysis_version",
        "source",
        "created_at",
        "updated_at",
    )

    def get_queryset(self, request):
        return super().get_queryset(request).select_related("opportunity")

    def has_add_permission(self, request) -> bool:
        return False

    def has_change_permission(self, request, obj=None) -> bool:
        return False

    def has_delete_permission(self, request, obj=None) -> bool:
        return False