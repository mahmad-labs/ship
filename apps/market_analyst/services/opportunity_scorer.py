"""
apps.market_analyst.services.opportunity_scorer
==================================================

Deterministic opportunity scoring for the Market Analyst pipeline.

Role
----
This module owns exactly one concern: turning a set of already-derived,
already-validated 0-100 dimension scores (demand, trend, margin,
competition, saturation) into a single weighted overall opportunity
score. It never:

    * calls an external API, scrapes a website, or touches a browser
    * invokes an LLM/AI provider
    * queries a data source (Reddit, Google Trends, Shopify, suppliers)
    * reads or writes the database
    * fabricates a dimension score it wasn't given

Existing integration point
---------------------------
``apps.market_analyst.agents.market_analyst.MarketAnalystAgent`` already
looks up ``calculate_overall_score`` on this module at import time (see
``_optional_service_callable`` there) and, if present, delegates to it
instead of its own internal fallback mean. That lookup is a plain
``getattr`` against this module -- it imposes the contract this module
must satisfy:

    calculate_overall_score(scores: Mapping[str, Decimal]) -> Decimal

``scores`` only contains dimensions the agent actually has usable
evidence for (never ``None`` values, never the ``EvidenceQuality``
wrapper) and the return value must be ``Decimal(...)``-constructible.
This module implements exactly that function, plus a small
``OpportunityScorer`` class that wraps it with an explainable
breakdown (per-dimension weighted contribution, missing dimensions,
narrative strengths/weaknesses) for callers -- chiefly
``services.analyzer`` -- that want more than a bare number.

Score direction
----------------
All five dimensions already share one convention, defined on
``ProductOpportunity`` in ``models.py``: 0-100, higher is always
better. In particular ``competition_score`` and ``saturation_score``
are *pre-inverted* by whichever service produces them (see
``competitor_analyzer.py``) so that "higher = less competition" /
"higher = less saturated". This module does not re-interpret or invert
any dimension -- it trusts the 0-100/higher-is-better contract the
model layer already enforces (``MinValueValidator``/``MaxValueValidator``
and a matching ``CheckConstraint`` on every score field).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any

from apps.market_analyst.models import SCORE_MAX, SCORE_MIN

logger = logging.getLogger(__name__)

__all__ = [
    "DIMENSION_WEIGHTS",
    "DIMENSION_LABELS",
    "InvalidScoreError",
    "DimensionContribution",
    "OpportunityScoreBreakdown",
    "calculate_overall_score",
    "OpportunityScorer",
]

_QUANTIZE = Decimal("0.01")

# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

# Explicit, documented, and deliberately the only place these numbers
# live. Demand and margin are weighted heaviest because they most
# directly answer "will this sell, and will it be profitable"; trend
# is a secondary confirming signal; competition/saturation matter but
# a merely-crowded category is not automatically disqualifying the
# way "nobody wants this" or "there is no margin" is. These are Ship's
# initial weights, not a claim of empirical optimality -- change them
# here, in one place, if the business decides otherwise.
DIMENSION_WEIGHTS: dict[str, Decimal] = {
    "demand": Decimal("0.30"),
    "trend": Decimal("0.20"),
    "margin": Decimal("0.25"),
    "competition": Decimal("0.15"),
    "saturation": Decimal("0.10"),
}

# Human-readable labels, including a reminder of score direction for
# the two dimensions whose direction is not obvious from the name
# alone. Used only for explanatory text (see `OpportunityScorer`).
DIMENSION_LABELS: dict[str, str] = {
    "demand": "demand",
    "trend": "trend",
    "margin": "margin",
    "competition": "competition (higher score = less competition)",
    "saturation": "saturation (higher score = less saturated)",
}


class InvalidScoreError(ValueError):
    """
    Raised when a supplied dimension score is not a valid 0-100 value.

    Distinct from "dimension missing" (which is not an error -- a
    caller simply omits it, per the no-fabrication policy). This is
    raised only for a dimension that *was* supplied but is malformed:
    not numeric, NaN, infinite, or outside [0, 100].
    """


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_score(dimension: str, raw: Any) -> Decimal:
    if raw is None:
        raise InvalidScoreError(f"score for dimension {dimension!r} is None; omit missing dimensions instead.")
    try:
        value = Decimal(str(raw))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise InvalidScoreError(f"score for dimension {dimension!r} is not a valid number: {raw!r}") from exc
    if not value.is_finite():
        raise InvalidScoreError(f"score for dimension {dimension!r} is not finite: {raw!r}")
    if value < SCORE_MIN or value > SCORE_MAX:
        raise InvalidScoreError(
            f"score for dimension {dimension!r} is out of range [{SCORE_MIN}, {SCORE_MAX}]: {value}"
        )
    return value


# ---------------------------------------------------------------------------
# Required contract: calculate_overall_score
# ---------------------------------------------------------------------------


def calculate_overall_score(scores: Mapping[str, Any]) -> Decimal:
    """
    Deterministic weighted-mean opportunity score over whichever
    dimensions are supplied.

    Only dimensions present in ``scores`` are used; weights are
    renormalized over the dimensions actually supplied so a partial
    analysis is never silently treated as if the missing dimensions
    scored zero (see module docstring on the no-fabrication policy --
    "unavailable" must never collapse into "0"). This is the same
    principle ``MarketAnalystAgent._fallback_overall_score`` already
    uses; this function is the real (weighted, validated, documented)
    implementation it defers to when available.

    Raises:
        InvalidScoreError: if ``scores`` is empty, if any supplied
            value is not a finite number in [0, 100], or if none of
            the supplied dimensions carry a configured weight.
    """
    if not scores:
        raise InvalidScoreError("calculate_overall_score() requires at least one dimension score.")

    weighted_total = Decimal("0")
    weight_sum = Decimal("0")
    for dimension, raw_value in scores.items():
        value = _validate_score(str(dimension), raw_value)
        weight = DIMENSION_WEIGHTS.get(str(dimension))
        if weight is None:
            logger.warning(
                "opportunity_scorer: dimension %r has no configured weight; excluded from scoring.", dimension
            )
            continue
        weighted_total += value * weight
        weight_sum += weight

    if weight_sum == 0:
        raise InvalidScoreError(
            "calculate_overall_score(): none of the supplied dimensions have a configured weight."
        )

    overall = (weighted_total / weight_sum).quantize(_QUANTIZE, rounding=ROUND_HALF_UP)
    return max(SCORE_MIN, min(SCORE_MAX, overall))


# ---------------------------------------------------------------------------
# Explainable wrapper
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DimensionContribution:
    """How much a single dimension contributed to the overall score."""

    dimension: str
    value: Decimal
    weight: Decimal
    weighted_value: Decimal
    contribution_pct: Decimal  # this dimension's share of the weighted total, 0-100


@dataclass(frozen=True)
class OpportunityScoreBreakdown:
    """
    Explainable result of ``OpportunityScorer.score()``: the overall
    score plus enough detail to answer "why did this score X" without
    the caller needing to re-derive the weighting scheme itself.
    """

    overall_score: Decimal
    contributions: tuple[DimensionContribution, ...] = field(default_factory=tuple)
    missing_dimensions: tuple[str, ...] = field(default_factory=tuple)


class OpportunityScorer:
    """
    Explainable, standalone-usable wrapper around
    ``calculate_overall_score``.

    Usable independently of the rest of the pipeline
    (``OpportunityScorer().score({...})``) for testing, dashboards, or
    future API usage, per this app's reusability requirement. Holds no
    state beyond an optional weight override, performs no I/O, and
    never invents a value for a dimension it wasn't given.
    """

    def __init__(self, *, dimension_weights: Mapping[str, Decimal] | None = None) -> None:
        self._weights: dict[str, Decimal] = dict(dimension_weights) if dimension_weights else dict(DIMENSION_WEIGHTS)

    def score(self, scores: Mapping[str, Any]) -> OpportunityScoreBreakdown:
        """
        Compute the overall score and a per-dimension explanation.

        Uses this instance's own weights for both the overall number
        and the breakdown in a single pass, so the two can never
        silently diverge from each other. A default-constructed
        ``OpportunityScorer()`` uses the same weights as (and always
        agrees with) the module-level ``calculate_overall_score``.
        """
        if not scores:
            raise InvalidScoreError("score() requires at least one dimension score.")

        validated: dict[str, Decimal] = {
            str(dimension): _validate_score(str(dimension), raw) for dimension, raw in scores.items()
        }

        weighted_total = Decimal("0")
        weight_sum = Decimal("0")
        usable: dict[str, tuple[Decimal, Decimal]] = {}
        for dimension, value in validated.items():
            weight = self._weights.get(dimension)
            if weight is None:
                logger.warning(
                    "opportunity_scorer: dimension %r has no configured weight; excluded from scoring.", dimension
                )
                continue
            usable[dimension] = (value, weight)
            weighted_total += value * weight
            weight_sum += weight

        if weight_sum == 0:
            raise InvalidScoreError("score(): none of the supplied dimensions have a configured weight.")

        overall = max(
            SCORE_MIN, min(SCORE_MAX, (weighted_total / weight_sum).quantize(_QUANTIZE, rounding=ROUND_HALF_UP))
        )

        contributions = tuple(
            DimensionContribution(
                dimension=dimension,
                value=value,
                weight=weight,
                weighted_value=(value * weight).quantize(_QUANTIZE, rounding=ROUND_HALF_UP),
                contribution_pct=(value * weight / weight_sum).quantize(_QUANTIZE, rounding=ROUND_HALF_UP),
            )
            for dimension, (value, weight) in usable.items()
        )
        missing = tuple(dim for dim in self._weights if dim not in usable)

        return OpportunityScoreBreakdown(
            overall_score=overall,
            contributions=contributions,
            missing_dimensions=missing,
        )

    def strengths_and_weaknesses(
        self,
        breakdown: OpportunityScoreBreakdown,
        *,
        strength_threshold: Decimal = Decimal("70"),
        weakness_threshold: Decimal = Decimal("40"),
    ) -> tuple[list[str], list[str]]:
        """
        Turn a breakdown into short, human-readable strength/weakness
        statements, one per dimension that clears either threshold.

        Centralized here (rather than re-implemented in ``analyzer.py``
        or any specialized service) so "what counts as a strong demand
        score" has exactly one definition project-wide.
        """
        strengths: list[str] = []
        weaknesses: list[str] = []
        for contribution in breakdown.contributions:
            label = DIMENSION_LABELS.get(contribution.dimension, contribution.dimension)
            if contribution.value >= strength_threshold:
                strengths.append(f"Strong {label}: {contribution.value}/100.")
            elif contribution.value <= weakness_threshold:
                weaknesses.append(f"Weak {label}: {contribution.value}/100.")
        return strengths, weaknesses