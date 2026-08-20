"""
apps.market_analyst.models
============================

Domain models for Ship's Market Analyst: the first intelligence layer
in the pipeline, responsible for representing which markets exist,
which product opportunities have been discovered inside them, and what
analytical conclusions have been reached about each opportunity.

Scope discipline: this module stores *state* (markets, opportunities,
analysis results) produced or consumed by the analysis pipeline. It
does not compute anything -- trend scraping, competitor research, LLM
calls, and scoring algorithms belong in ``services/`` and ``agents/``
(future work), which read/write these models. Nothing here talks to an
external API or a specific AI/LLM vendor.

Models:
    Market              A country/region market being analyzed.
    ProductOpportunity  A candidate product Ship is evaluating inside
                         a market, carrying derived opportunity scores.
    MarketAnalysis      A single analytical result (summary,
                         recommendation, confidence, structured
                         evidence) produced for a ProductOpportunity.

All three build on ``apps.core.models.BaseModel`` for UUID identity and
``created_at``/``updated_at``, matching the rest of Ship rather than
introducing a second identity scheme.

Data-integrity philosophy (see module-level docstrings below for
detail): opportunity scores and analysis confidence are *derived*
intelligence. They are nullable until actually calculated, bounded at
both the model-validation layer (``MinValueValidator``/
``MaxValueValidator``) and the database layer (``CheckConstraint``),
and are never accepted directly from arbitrary client input -- see
``forms.py``/``views.py`` for the write surface, which only ever
accepts raw submission fields (name, category, description, market),
never scores or recommendations.
"""

from __future__ import annotations

from decimal import Decimal

from django.conf import settings
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models
from django.db.models.functions import Lower

from apps.core.models import BaseModel

# ---------------------------------------------------------------------------
# Shared score field configuration
# ---------------------------------------------------------------------------

# All opportunity/analysis scores share one 0-100 scale, stored with two
# decimal places so downstream averaging/weighting (e.g. overall_score
# from its components) never loses precision to floating-point error.
SCORE_MAX_DIGITS = 5
SCORE_DECIMAL_PLACES = 2
SCORE_MIN = Decimal("0")
SCORE_MAX = Decimal("100")


def _score_field(help_text: str) -> models.DecimalField:
    """
    Shared factory for a nullable 0-100 opportunity/confidence score.

    Nullable because a score is *derived* intelligence: a freshly
    discovered opportunity has no demand/trend/margin score yet, and
    pretending otherwise (e.g. defaulting to 0) would make "not yet
    analyzed" indistinguishable from "analyzed and scored zero".
    Range enforcement happens twice, deliberately: `MinValueValidator`/
    `MaxValueValidator` give a clean Django/DRF-style validation error;
    the model's `CheckConstraint` (see each model's `Meta`) makes the
    invariant hold at the database level regardless of write path.
    """
    return models.DecimalField(
        max_digits=SCORE_MAX_DIGITS,
        decimal_places=SCORE_DECIMAL_PLACES,
        null=True,
        blank=True,
        validators=[MinValueValidator(SCORE_MIN), MaxValueValidator(SCORE_MAX)],
        help_text=help_text,
    )


def _score_range_constraint(field_name: str, app_label: str = "market_analyst") -> models.CheckConstraint:
    """Database-level twin of `_score_field`'s validators: NULL or within [0, 100]."""
    return models.CheckConstraint(
        condition=(
            models.Q(**{f"{field_name}__isnull": True})
            | (
                models.Q(**{f"{field_name}__gte": SCORE_MIN})
                & models.Q(**{f"{field_name}__lte": SCORE_MAX})
            )
        ),
        name=f"{app_label}_%(class)s_{field_name}_in_range",
    )


# ---------------------------------------------------------------------------
# Market
# ---------------------------------------------------------------------------


class Market(BaseModel):
    """
    A country/region market that Ship evaluates product opportunities
    within (e.g. "United States", or a narrower region inside it).

    Deliberately minimal: this is a reference/dimension table for
    `ProductOpportunity`, not a place to accumulate market-wide
    analytics (those belong on `MarketAnalysis`/future market-level
    intelligence models, keyed off this one).
    """

    name = models.CharField(
        max_length=150,
        help_text='Market name, e.g. "United States" or "United States - West Coast".',
    )
    country_code = models.CharField(
        max_length=2,
        help_text='ISO 3166-1 alpha-2 country code, e.g. "US".',
    )
    region = models.CharField(
        max_length=150,
        blank=True,
        help_text='Optional sub-national/regional qualifier, e.g. "West Coast".',
    )
    currency = models.CharField(
        max_length=3,
        help_text='ISO 4217 currency code, e.g. "USD".',
    )
    is_active = models.BooleanField(
        default=True,
        help_text="Whether Ship is currently sourcing/evaluating opportunities in this market.",
    )

    class Meta:
        verbose_name = "Market"
        verbose_name_plural = "Markets"
        ordering = ["name"]
        constraints = [
            models.UniqueConstraint(
                Lower("name"),
                "country_code",
                "region",
                name="market_analyst_market_unique_name_country_region",
            ),
        ]
        indexes = [
            models.Index(fields=["country_code"], name="market_by_country_code"),
            models.Index(fields=["is_active"], name="market_by_is_active"),
        ]

    def __str__(self) -> str:
        label = f"{self.name} ({self.country_code})"
        return f"{label} - {self.region}" if self.region else label


# ---------------------------------------------------------------------------
# Product opportunity
# ---------------------------------------------------------------------------


class ProductOpportunityStatus(models.TextChoices):
    """
    Lifecycle of a product opportunity, from initial submission through
    to a final human/agent decision.

    Deliberately a small, closed vocabulary (matches the checklist this
    app was scoped against) rather than free-form status text, so
    downstream agents/dashboards can rely on a fixed set of states.
    """

    DISCOVERED = "discovered", "Discovered"
    ANALYZING = "analyzing", "Analyzing"
    INVESTIGATE = "investigate", "Investigate"
    MONITOR = "monitor", "Monitor"
    APPROVED = "approved", "Approved"
    REJECTED = "rejected", "Rejected"
    ARCHIVED = "archived", "Archived"


class ProductOpportunity(BaseModel):
    """
    A candidate product Ship is evaluating as a potential ecommerce
    opportunity inside a `Market`.

    Score fields (`demand_score` ... `overall_score`) are *derived*
    intelligence, not user input: `forms.ProductOpportunityCreateForm`
    only ever accepts `market`, `name`, `category`, `description`.
    Scores are populated later, out-of-band, by the future analysis
    pipeline (see `tasks.py`), which is the only code path expected to
    write them. This keeps a hard line between "what a user submitted"
    and "what Ship calculated" -- see module docstring on models.py and
    the project's data-integrity requirements.

    `created_at` (from `BaseModel`) doubles as the discovery timestamp;
    a separate `discovered_at` field would be redundant since an
    opportunity is "discovered" at the moment its row is created.
    """

    market = models.ForeignKey(
        Market,
        on_delete=models.PROTECT,
        related_name="opportunities",
        help_text="Market this opportunity is being evaluated within.",
    )
    submitted_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="submitted_opportunities",
        help_text="User who submitted this opportunity for analysis, if any.",
    )

    name = models.CharField(max_length=255, help_text="Product name or working title.")
    category = models.CharField(max_length=150, help_text='Product category, e.g. "Jewelry".')
    description = models.TextField(blank=True, help_text="Optional free-text description of the product idea.")

    status = models.CharField(
        max_length=20,
        choices=ProductOpportunityStatus.choices,
        default=ProductOpportunityStatus.DISCOVERED,
        help_text="Current lifecycle state of this opportunity.",
    )

    demand_score = _score_field("How strong current demand appears to be (0-100).")
    trend_score = _score_field("How quickly demand is growing (0-100).")
    margin_score = _score_field("How attractive potential margins appear to be (0-100).")
    competition_score = _score_field("How competitive the market is; higher = less competition (0-100).")
    saturation_score = _score_field("How saturated the product/category appears; higher = less saturated (0-100).")
    overall_score = _score_field(
        "Composite opportunity score derived from the component scores above (0-100)."
    )

    class Meta:
        verbose_name = "Product opportunity"
        verbose_name_plural = "Product opportunities"
        ordering = ["-created_at"]
        constraints = [
            _score_range_constraint("demand_score"),
            _score_range_constraint("trend_score"),
            _score_range_constraint("margin_score"),
            _score_range_constraint("competition_score"),
            _score_range_constraint("saturation_score"),
            _score_range_constraint("overall_score"),
        ]
        indexes = [
            models.Index(fields=["market", "status"], name="opportunity_by_market_status"),
            models.Index(fields=["status"], name="opportunity_by_status"),
            models.Index(fields=["overall_score"], name="opportunity_by_overall_score"),
            models.Index(fields=["category"], name="opportunity_by_category"),
            models.Index(fields=["created_at"], name="opportunity_by_created_at"),
        ]

    def __str__(self) -> str:
        return f"{self.name} ({self.market.country_code}) [{self.status}]"

    @property
    def is_scored(self) -> bool:
        """True once an overall opportunity score has actually been calculated."""
        return self.overall_score is not None


# ---------------------------------------------------------------------------
# Market analysis
# ---------------------------------------------------------------------------


class MarketAnalysisRecommendation(models.TextChoices):
    """Controlled vocabulary for an analysis's bottom-line recommendation."""

    STRONG_OPPORTUNITY = "strong_opportunity", "Strong opportunity"
    INVESTIGATE = "investigate", "Investigate"
    MONITOR = "monitor", "Monitor"
    AVOID = "avoid", "Avoid"


class MarketAnalysis(BaseModel):
    """
    A single analytical result produced for a `ProductOpportunity`.

    An opportunity may accumulate more than one `MarketAnalysis` over
    time (e.g. re-analyzed as new data comes in); the most recent one
    (by `created_at`, the default ordering) is the current view.

    `structured_data` holds the analysis engine's supporting evidence
    (e.g. per-signal breakdowns, source references) in a schema that
    is expected to evolve as new data providers/agents are added.
    Because that shape isn't fixed, this model does not attempt to
    normalize it into columns; `analysis_version` records which schema
    a given row used, so future readers can branch on it instead of
    guessing. Callers must treat `structured_data` as untrusted,
    schema-less JSON -- never rendered as HTML, never used to build a
    query, and never assumed to contain any particular key.
    """

    opportunity = models.ForeignKey(
        ProductOpportunity,
        on_delete=models.CASCADE,
        related_name="analyses",
        help_text="Product opportunity this analysis result belongs to.",
    )

    summary = models.TextField(help_text="Human-readable summary of the analysis findings.")
    recommendation = models.CharField(
        max_length=20,
        choices=MarketAnalysisRecommendation.choices,
        help_text="Bottom-line recommendation produced by this analysis.",
    )
    confidence_score = models.DecimalField(
        max_digits=SCORE_MAX_DIGITS,
        decimal_places=SCORE_DECIMAL_PLACES,
        validators=[MinValueValidator(SCORE_MIN), MaxValueValidator(SCORE_MAX)],
        help_text="How confident the analysis engine is in this result (0-100).",
    )
    structured_data = models.JSONField(
        default=dict,
        blank=True,
        help_text="Structured, engine-specific evidence/breakdown backing this result. Schema-less; never trusted as-is.",
    )
    analysis_version = models.CharField(
        max_length=50,
        blank=True,
        help_text='Identifier of the analysis engine/schema version that produced this row, e.g. "heuristic-v1".',
    )
    source = models.CharField(
        max_length=100,
        blank=True,
        help_text='Who/what produced this analysis, e.g. "market_analyst_agent" or an operator\'s identifier.',
    )

    class Meta:
        verbose_name = "Market analysis"
        verbose_name_plural = "Market analyses"
        ordering = ["-created_at"]
        constraints = [
            models.CheckConstraint(
                condition=(
                    models.Q(confidence_score__gte=SCORE_MIN) & models.Q(confidence_score__lte=SCORE_MAX)
                ),
                name="market_analyst_marketanalysis_confidence_score_in_range",
            ),
        ]
        indexes = [
            models.Index(fields=["opportunity", "-created_at"], name="analysis_by_opp_created"),
            models.Index(fields=["recommendation"], name="analysis_by_recommendation"),
        ]

    def __str__(self) -> str:
        return f"Analysis for {self.opportunity.name}: {self.recommendation} ({self.confidence_score})"