"""
apps.market_analyst.services.customer_analyzer
==================================================

Customer intelligence analysis: what do customers want, what
frustrates them, and what unsolved problem/differentiation opportunity
exists.

Consumes
--------
A sequence of ``CustomerFeedbackItem`` -- already-collected reviews,
survey responses, or social discussion snippets, supplied by the
caller. Ship has no AI/LLM provider abstraction anywhere in the
codebase yet (checked across `apps/core`, `apps/authentication`, and
this app), so this module does not call an LLM, does not import any
AI SDK, and does not attempt its own natural-language sentiment
inference from raw text. Instead it works the way `pricing_analyzer`
and `trend_analyzer` do: it aggregates already-structured signals
(a numeric rating, an already-assigned sentiment label, already-tagged
themes) that an upstream collection/labeling step is responsible for
producing. When that upstream AI step exists, wiring its output into
`CustomerFeedbackItem.sentiment`/`theme_tags` is a data-provider
concern, not a change needed here -- see `AI INTEGRATION` in this
app's build spec: "the services must remain AI-provider agnostic."

Produces
--------
``CustomerAnalysisResult``. Customer intelligence has no dedicated
slot among the five persisted `ProductOpportunity` dimensions
(demand/trend/margin/competition/saturation) -- it is qualitative,
corroborating evidence rather than a sixth score. The higher-level
`Analyzer` uses it to enrich narrative strengths/weaknesses/risks and
as an optional confidence signal for the `demand` dimension, not as a
`DimensionEvidence` of its own.

Privacy discipline
--------------------
Only aggregated, de-identified themes are ever returned -- never raw
review/feedback text, and never anything from `CustomerFeedbackItem`
that could identify an individual customer. `raw_text` on each item
is used only to compute `sample_size`/length-based signals here; it is
never copied into the result.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from enum import Enum

from apps.market_analyst.agents.market_analyst import AnalysisStatus, EvidenceQuality

logger = logging.getLogger(__name__)

__all__ = [
    "CustomerSentiment",
    "ThemeCategory",
    "CustomerFeedbackItem",
    "ThemeCount",
    "CustomerAnalysisResult",
    "CustomerAnalyzer",
]

_QUANTIZE = Decimal("0.01")
_SCORE_MIN = Decimal("0")
_SCORE_MAX = Decimal("100")


class CustomerSentiment(str, Enum):
    """Sentiment label. Assigned upstream (by a human or an AI-backed data provider); never inferred here."""

    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class ThemeCategory(str, Enum):
    """How an item's `theme_tags` should be bucketed. Assigned upstream, same as sentiment."""

    PAIN_POINT = "pain_point"
    FEATURE_REQUEST = "feature_request"
    PURCHASE_DRIVER = "purchase_driver"
    OBJECTION = "objection"
    GENERAL = "general"


@dataclass(frozen=True)
class CustomerFeedbackItem:
    """
    One already-collected piece of customer feedback.

    ``rating`` is expected on a 1-5 scale if present (the common
    ecommerce-review convention); ``sentiment`` and ``theme_tags`` are
    expected to already be labeled by whatever upstream process
    collected this item (see module docstring) -- this analyzer only
    aggregates them, never assigns them.
    """

    rating: Decimal | None = None
    sentiment: CustomerSentiment | None = None
    theme_tags: Sequence[tuple[str, ThemeCategory]] = field(default_factory=tuple)
    source: str = ""


@dataclass(frozen=True)
class ThemeCount:
    """A theme tag and how many feedback items mentioned it."""

    theme: str
    count: int


@dataclass(frozen=True)
class CustomerAnalysisResult:
    """Structured output of `CustomerAnalyzer.analyze()`."""

    status: AnalysisStatus

    sentiment_score: Decimal | None  # 0-100, higher = more positive
    sentiment_quality: EvidenceQuality

    positive_themes: list[ThemeCount] = field(default_factory=list)
    negative_themes: list[ThemeCount] = field(default_factory=list)
    pain_points: list[str] = field(default_factory=list)
    desired_features: list[str] = field(default_factory=list)
    purchase_drivers: list[str] = field(default_factory=list)
    objections: list[str] = field(default_factory=list)

    sample_size: int = 0
    confidence: Decimal = Decimal("0.00")
    warnings: list[str] = field(default_factory=list)
    missing_data: list[str] = field(default_factory=list)


class CustomerAnalyzer:
    """Stateless, deterministic aggregator over a sequence of `CustomerFeedbackItem`."""

    RATING_SCALE_MAX = Decimal("5")
    MAX_THEMES_RETURNED = 10
    # A sentiment claim below this many items is reported with reduced
    # confidence rather than withheld outright -- unlike a trend claim,
    # even a handful of reviews is genuine (if thin) evidence.
    LOW_SAMPLE_THRESHOLD = 5

    def analyze(self, feedback: Sequence[CustomerFeedbackItem] | None) -> CustomerAnalysisResult:
        if not feedback:
            return self._insufficient_data_result(["No customer feedback was supplied."])

        warnings: list[str] = []
        valid_ratings = self._valid_ratings(feedback, warnings)
        sentiment_labels = [item.sentiment for item in feedback if item.sentiment is not None]

        sentiment_score, sentiment_quality = self._sentiment(valid_ratings, sentiment_labels)

        positive_themes, negative_themes, pain_points, desired_features, purchase_drivers, objections = (
            self._themes(feedback)
        )

        missing_data: list[str] = []
        if sentiment_quality == EvidenceQuality.UNAVAILABLE:
            missing_data.append("sentiment")

        confidence = self._confidence(feedback, valid_ratings, sentiment_labels)
        status = AnalysisStatus.SUCCESS if not missing_data else AnalysisStatus.PARTIAL
        if sentiment_quality == EvidenceQuality.UNAVAILABLE and not (positive_themes or negative_themes):
            status = AnalysisStatus.INSUFFICIENT_DATA

        return CustomerAnalysisResult(
            status=status,
            sentiment_score=sentiment_score,
            sentiment_quality=sentiment_quality,
            positive_themes=positive_themes,
            negative_themes=negative_themes,
            pain_points=pain_points,
            desired_features=desired_features,
            purchase_drivers=purchase_drivers,
            objections=objections,
            sample_size=len(feedback),
            confidence=confidence,
            warnings=warnings,
            missing_data=missing_data,
        )

    # -- Validation ------------------------------------------------------

    def _valid_ratings(self, feedback: Sequence[CustomerFeedbackItem], warnings: list[str]) -> list[Decimal]:
        ratings: list[Decimal] = []
        for item in feedback:
            if item.rating is None:
                continue
            try:
                value = Decimal(item.rating)
            except (InvalidOperation, TypeError, ValueError):
                warnings.append(f"Discarded non-numeric rating: {item.rating!r}.")
                continue
            if not value.is_finite() or value < 0 or value > self.RATING_SCALE_MAX:
                warnings.append(f"Discarded out-of-range rating (expected 0-{self.RATING_SCALE_MAX}): {value!r}.")
                continue
            ratings.append(value)
        return ratings

    # -- Sentiment -----------------------------------------------------

    def _sentiment(
        self, ratings: list[Decimal], sentiment_labels: list[CustomerSentiment]
    ) -> tuple[Decimal | None, EvidenceQuality]:
        components: list[Decimal] = []

        if ratings:
            mean_rating = sum(ratings) / Decimal(len(ratings))
            components.append((mean_rating / self.RATING_SCALE_MAX) * _SCORE_MAX)

        if sentiment_labels:
            total = Decimal(len(sentiment_labels))
            positive = Decimal(sum(1 for s in sentiment_labels if s == CustomerSentiment.POSITIVE))
            negative = Decimal(sum(1 for s in sentiment_labels if s == CustomerSentiment.NEGATIVE))
            components.append((((positive - negative) / total) * Decimal("50")) + Decimal("50"))

        if not components:
            return None, EvidenceQuality.UNAVAILABLE

        score = (sum(components) / Decimal(len(components))).quantize(_QUANTIZE, rounding=ROUND_HALF_UP)
        score = max(_SCORE_MIN, min(_SCORE_MAX, score))
        quality = EvidenceQuality.CALCULATED if len(components) > 1 or len(ratings) + len(sentiment_labels) > 1 else EvidenceQuality.OBSERVED
        return score, quality

    # -- Themes -----------------------------------------------------

    def _themes(
        self, feedback: Sequence[CustomerFeedbackItem]
    ) -> tuple[list[ThemeCount], list[ThemeCount], list[str], list[str], list[str], list[str]]:
        positive_counter: Counter[str] = Counter()
        negative_counter: Counter[str] = Counter()
        pain_point_counter: Counter[str] = Counter()
        feature_request_counter: Counter[str] = Counter()
        purchase_driver_counter: Counter[str] = Counter()
        objection_counter: Counter[str] = Counter()

        for item in feedback:
            for theme, category in item.theme_tags:
                theme = theme.strip()
                if not theme:
                    continue
                if category == ThemeCategory.PAIN_POINT:
                    pain_point_counter[theme] += 1
                    negative_counter[theme] += 1
                elif category == ThemeCategory.FEATURE_REQUEST:
                    feature_request_counter[theme] += 1
                elif category == ThemeCategory.PURCHASE_DRIVER:
                    purchase_driver_counter[theme] += 1
                    positive_counter[theme] += 1
                elif category == ThemeCategory.OBJECTION:
                    objection_counter[theme] += 1
                    negative_counter[theme] += 1
                elif item.sentiment == CustomerSentiment.POSITIVE:
                    positive_counter[theme] += 1
                elif item.sentiment == CustomerSentiment.NEGATIVE:
                    negative_counter[theme] += 1

        def top(counter: Counter[str]) -> list[ThemeCount]:
            return [ThemeCount(theme=theme, count=count) for theme, count in counter.most_common(self.MAX_THEMES_RETURNED)]

        def top_names(counter: Counter[str]) -> list[str]:
            return [theme for theme, _ in counter.most_common(self.MAX_THEMES_RETURNED)]

        return (
            top(positive_counter),
            top(negative_counter),
            top_names(pain_point_counter),
            top_names(feature_request_counter),
            top_names(purchase_driver_counter),
            top_names(objection_counter),
        )

    # -- Confidence -----------------------------------------------------

    def _confidence(
        self,
        feedback: Sequence[CustomerFeedbackItem],
        ratings: list[Decimal],
        sentiment_labels: list[CustomerSentiment],
    ) -> Decimal:
        usable = len(ratings) + len(sentiment_labels)
        if usable == 0:
            return Decimal("0.00")
        sample_adequacy = min(Decimal("1"), Decimal(usable) / Decimal(self.LOW_SAMPLE_THRESHOLD * 4))
        confidence = (sample_adequacy * _SCORE_MAX).quantize(_QUANTIZE, rounding=ROUND_HALF_UP)
        return max(_SCORE_MIN, min(_SCORE_MAX, confidence))

    # -- Fallbacks -----------------------------------------------------

    def _insufficient_data_result(self, warnings: list[str]) -> CustomerAnalysisResult:
        return CustomerAnalysisResult(
            status=AnalysisStatus.INSUFFICIENT_DATA,
            sentiment_score=None,
            sentiment_quality=EvidenceQuality.UNAVAILABLE,
            sample_size=0,
            confidence=Decimal("0.00"),
            warnings=warnings,
            missing_data=["sentiment"],
        )