"""
apps.market_analyst.agents.market_analyst
=============================================

Orchestration layer for Ship's Market Analyst intelligence pipeline.

Role
----
``MarketAnalystAgent`` is a coordinator, not a data source. It turns
already-collected evidence about a `ProductOpportunity` into a
structured analytical result (score, recommendation, confidence,
evidence breakdown) that a caller (a Celery task, management command,
or view) can persist as a `MarketAnalysis`. It never:

    * scrapes or fetches external data (no HTTP, no browser automation)
    * calls an LLM/AI provider directly
    * writes to the database
    * fabricates a metric it wasn't given evidence for

Where this fits today
----------------------
As of this change, every module under `apps.market_analyst.services`
(`analyzer.py`, `trend_analyzer.py`, `pricing_analyzer.py`,
`competitor_analyzer.py`, `customer_analyzer.py`,
`opportunity_scorer.py`) is an empty placeholder -- none of them
expose a scoring function yet. `apps.market_analyst.tasks` already
documents that its `_run_placeholder_analysis` is a stand-in for
"the future `services`/`agents` layer". This module *is* that future
layer's orchestration half. Concretely:

    apps.market_analyst.tasks.analyze_product_opportunity
        -> loads a ProductOpportunity (unchanged, out of scope here)
        -> would construct evidence (out of scope here; a future
           evidence-collection service's job) and call
           MarketAnalystAgent().analyze(opportunity, evidence)
        -> persists the MarketAnalysis using
           MarketAnalystResult.as_market_analysis_kwargs()

This file does not modify `tasks.py`; the above is documentation of
the intended integration point, not a change made here.

Scoring delegation
-------------------
Ship's opportunity scoring implementation (`services.opportunity_scorer`)
does not exist yet -- the module is currently empty. Per this app's
build constraints, this agent does not duplicate a scoring algorithm
inside itself. Instead it attempts, at import time, to bind an
optional `calculate_overall_score(scores: Mapping[str, Decimal]) ->
Decimal` callable from `services.opportunity_scorer`. The moment that
function is implemented there, this agent starts using it automatically
with no further changes here. Until then, `_fallback_overall_score`
below is the smallest possible compatibility calculation (an
evidence-weighted mean of the dimensions that were actually supplied),
clearly labelled as a fallback in `analysis_metadata.scoring_source`
on every result.

No-hallucination policy
------------------------
A `ProductOpportunity` persists exactly five derived scores: demand,
trend, margin, competition, saturation (see `models.py`). This agent
mirrors that set exactly via `AnalysisDimension` rather than inventing
extra dimensions (seasonality, risk, ...) with nowhere to be stored.
For any dimension the caller has no real evidence for, this agent
records it in `missing_data` and excludes it from scoring -- it never
substitutes an estimate for an observed fact. `EvidenceQuality`
preserves the observed/calculated/estimated/inferred/unavailable
distinction the project's data-integrity requirements call for.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any

from apps.market_analyst.models import (
    SCORE_MAX,
    SCORE_MIN,
    MarketAnalysisRecommendation,
    ProductOpportunity,
)

logger = logging.getLogger(__name__)

__all__ = [
    "AnalysisDimension",
    "AnalysisStatus",
    "EvidenceQuality",
    "DimensionEvidence",
    "MarketAnalystResult",
    "MarketAnalystInputError",
    "MarketAnalystAgent",
]


# ---------------------------------------------------------------------------
# Optional delegation to services.opportunity_scorer, if/when implemented.
# ---------------------------------------------------------------------------


def _optional_service_callable(module_path: str, attr: str) -> Callable[..., Any] | None:
    """
    Best-effort lookup of ``attr`` on ``module_path``, returning None
    rather than raising if the module has no such attribute (which is
    the current state of every services/*.py placeholder). The module
    itself is expected to exist (it does -- it's part of this app's
    committed structure) and a genuine ImportError is allowed to
    propagate, since that would indicate a real packaging problem
    rather than "not implemented yet".
    """
    module = importlib.import_module(module_path)
    return getattr(module, attr, None)


_calculate_overall_score: Callable[[Mapping[str, Decimal]], Decimal] | None = _optional_service_callable(
    "apps.market_analyst.services.opportunity_scorer", "calculate_overall_score"
)


# ---------------------------------------------------------------------------
# Controlled vocabularies
# ---------------------------------------------------------------------------


class AnalysisDimension(str, Enum):
    """
    The scored dimensions Ship currently persists on `ProductOpportunity`.

    Deliberately a 1:1 mirror of that model's score fields
    (`demand_score`, `trend_score`, `margin_score`,
    `competition_score`, `saturation_score`) rather than an
    independently invented list -- there is nowhere to store a score
    for a dimension the model doesn't have yet.
    """

    DEMAND = "demand"
    TREND = "trend"
    MARGIN = "margin"
    COMPETITION = "competition"
    SATURATION = "saturation"


class EvidenceQuality(str, Enum):
    """How a dimension's value was derived, not just what it is."""

    OBSERVED = "observed"
    CALCULATED = "calculated"
    ESTIMATED = "estimated"
    INFERRED = "inferred"
    UNAVAILABLE = "unavailable"


class AnalysisStatus(str, Enum):
    """Bottom-line outcome of a single `analyze()` call."""

    SUCCESS = "success"
    PARTIAL = "partial"
    INSUFFICIENT_DATA = "insufficient_data"


# ---------------------------------------------------------------------------
# Input / output data shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DimensionEvidence:
    """
    Evidence for a single `AnalysisDimension`, supplied by the caller.

    This agent does not collect evidence itself (see module
    docstring); a future evidence-collection service/task is expected
    to build these from real, sourced signals. Leave ``value`` as
    ``None`` (the default) when no real signal exists yet -- the agent
    treats that as a missing dimension rather than fabricating a
    placeholder number.
    """

    value: Decimal | None = None
    quality: EvidenceQuality = EvidenceQuality.UNAVAILABLE
    source: str = ""


@dataclass(frozen=True)
class MarketAnalystResult:
    """
    Structured output of `MarketAnalystAgent.analyze()`.

    This is a plain, immutable data object -- it is never itself
    persisted by the agent (see "Idempotency"/"Transaction safety" in
    the module docstring). `as_market_analysis_kwargs()` maps it onto
    `MarketAnalysis`'s persisted fields for a caller to do:

        MarketAnalysis.objects.create(
            opportunity=opportunity,
            **result.as_market_analysis_kwargs(),
        )
    """

    status: AnalysisStatus
    opportunity_id: str
    overall_score: Decimal | None
    recommendation: MarketAnalysisRecommendation
    confidence_score: Decimal
    summary: str
    metrics: dict[str, Any] = field(default_factory=dict)
    missing_data: list[str] = field(default_factory=list)
    analysis_metadata: dict[str, Any] = field(default_factory=dict)

    def as_market_analysis_kwargs(self) -> dict[str, Any]:
        """Field mapping for persisting this result as a `MarketAnalysis`."""
        return {
            "summary": self.summary,
            "recommendation": self.recommendation,
            "confidence_score": self.confidence_score,
            "structured_data": {
                "status": self.status.value,
                "overall_score": str(self.overall_score) if self.overall_score is not None else None,
                "metrics": self.metrics,
                "missing_data": self.missing_data,
                "analysis_metadata": self.analysis_metadata,
            },
            "analysis_version": self.analysis_metadata.get("engine_version", ""),
            "source": "market_analyst_agent",
        }


class MarketAnalystInputError(ValueError):
    """
    Raised when `analyze()` receives structurally invalid input --
    e.g. the wrong type, or evidence that isn't a `DimensionEvidence`.
    This is a programmer error, distinct from an opportunity that
    simply lacks evidence (a normal, expected outcome represented by
    `AnalysisStatus.INSUFFICIENT_DATA` in the returned result, not an
    exception).
    """


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class MarketAnalystAgent:
    """
    Coordinates market-intelligence analysis for a single
    `ProductOpportunity`.

    Usage::

        agent = MarketAnalystAgent()
        result = agent.analyze(opportunity, evidence)

    ``evidence`` maps `AnalysisDimension.value` strings (e.g.
    ``"demand"``) to `DimensionEvidence`. Dimensions absent from the
    mapping, or present with ``value=None``/``quality=UNAVAILABLE``,
    are treated as missing -- never inferred or estimated by this
    agent.

    The constructor performs no I/O: no database queries, no external
    calls, no LLM invocation. It only records configuration (dimension
    weights used by the internal fallback scorer), so agents can be
    constructed cheaply and repeatedly, including in tests.
    """

    ENGINE_VERSION = "market-analyst-agent-v1"

    DIMENSIONS: tuple[AnalysisDimension, ...] = tuple(AnalysisDimension)

    #: Relative trust given to each evidence quality tier when deriving
    #: confidence. Not a probability -- a simple, documented weighting.
    QUALITY_WEIGHTS: dict[EvidenceQuality, Decimal] = {
        EvidenceQuality.OBSERVED: Decimal("1.00"),
        EvidenceQuality.CALCULATED: Decimal("0.85"),
        EvidenceQuality.ESTIMATED: Decimal("0.60"),
        EvidenceQuality.INFERRED: Decimal("0.40"),
        EvidenceQuality.UNAVAILABLE: Decimal("0.00"),
    }

    # Recommendation thresholds. A high score alone is never enough for
    # the strongest/weakest calls -- confidence must also clear its own
    # bar, preserving the score/confidence distinction the project
    # requires (e.g. score=91 but confidence=42 must not read as a
    # strong opportunity).
    STRONG_OPPORTUNITY_SCORE = Decimal("75")
    STRONG_OPPORTUNITY_CONFIDENCE = Decimal("60")
    INVESTIGATE_SCORE = Decimal("55")
    AVOID_SCORE = Decimal("35")
    AVOID_CONFIDENCE = Decimal("40")

    def __init__(self, *, dimension_weights: Mapping[AnalysisDimension, Decimal] | None = None) -> None:
        self._dimension_weights: dict[AnalysisDimension, Decimal] = (
            dict(dimension_weights) if dimension_weights else {dim: Decimal("1") for dim in self.DIMENSIONS}
        )

    # -- Public API ---------------------------------------------------

    def analyze(
        self,
        opportunity: ProductOpportunity,
        evidence: Mapping[str, DimensionEvidence] | None = None,
    ) -> MarketAnalystResult:
        """
        Run one analysis pass for ``opportunity`` using ``evidence``.

        Deterministic with respect to its inputs: the same opportunity
        and evidence always produce the same result. Performs no
        database writes and raises only for structurally invalid input
        (`MarketAnalystInputError`) or genuinely unexpected failures,
        which are logged and re-raised rather than swallowed.
        """
        if not isinstance(opportunity, ProductOpportunity):
            raise MarketAnalystInputError(
                f"analyze() requires a ProductOpportunity instance, got {type(opportunity)!r}."
            )

        evidence = evidence or {}
        for key, value in evidence.items():
            if not isinstance(value, DimensionEvidence):
                raise MarketAnalystInputError(
                    f"evidence[{key!r}] must be a DimensionEvidence, got {type(value)!r}."
                )

        logger.info("market_analyst_agent: analysis started opportunity_id=%s", opportunity.id)

        try:
            usable, missing = self._partition_evidence(evidence)

            if not usable:
                result = self._insufficient_data_result(opportunity, missing)
                logger.info(
                    "market_analyst_agent: analysis insufficient_data opportunity_id=%s", opportunity.id
                )
                return result

            overall_score = self._score(usable)
            confidence = self._confidence(usable)
            recommendation = self._recommend(overall_score, confidence)
            status = AnalysisStatus.SUCCESS if not missing else AnalysisStatus.PARTIAL

            result = MarketAnalystResult(
                status=status,
                opportunity_id=str(opportunity.id),
                overall_score=overall_score,
                recommendation=recommendation,
                confidence_score=confidence,
                summary=self._summarize(opportunity, overall_score, recommendation, missing),
                metrics=self._metrics(usable),
                missing_data=[dim.value for dim in missing],
                analysis_metadata={
                    "engine_version": self.ENGINE_VERSION,
                    "scoring_source": (
                        "services.opportunity_scorer" if _calculate_overall_score else "agent_internal_fallback"
                    ),
                    "dimensions_evaluated": [dim.value for dim in usable],
                },
            )

            logger.info(
                "market_analyst_agent: analysis %s opportunity_id=%s recommendation=%s confidence=%s",
                status.value,
                opportunity.id,
                recommendation,
                confidence,
            )
            return result
        except MarketAnalystInputError:
            raise
        except Exception:
            logger.exception(
                "market_analyst_agent: unexpected failure analyzing opportunity_id=%s", opportunity.id
            )
            raise

    # -- Evidence handling ---------------------------------------------

    def _partition_evidence(
        self, evidence: Mapping[str, DimensionEvidence]
    ) -> tuple[dict[AnalysisDimension, DimensionEvidence], list[AnalysisDimension]]:
        usable: dict[AnalysisDimension, DimensionEvidence] = {}
        missing: list[AnalysisDimension] = []
        for dim in self.DIMENSIONS:
            data = evidence.get(dim.value)
            if data is not None and data.value is not None and data.quality != EvidenceQuality.UNAVAILABLE:
                usable[dim] = data
            else:
                missing.append(dim)
        return usable, missing

    def _metrics(self, usable: Mapping[AnalysisDimension, DimensionEvidence]) -> dict[str, Any]:
        return {
            dim.value: {
                "value": str(data.value),
                "quality": data.quality.value,
                "source": data.source or None,
            }
            for dim, data in usable.items()
        }

    # -- Scoring ---------------------------------------------------------

    def _score(self, usable: Mapping[AnalysisDimension, DimensionEvidence]) -> Decimal | None:
        if _calculate_overall_score is not None:
            scores = {dim.value: data.value for dim, data in usable.items()}
            return self._clamp(Decimal(_calculate_overall_score(scores)))
        return self._fallback_overall_score(usable)

    def _fallback_overall_score(
        self, usable: Mapping[AnalysisDimension, DimensionEvidence]
    ) -> Decimal | None:
        """
        Minimal compatibility scorer used only while
        `services.opportunity_scorer.calculate_overall_score` doesn't
        exist yet. A weighted mean of the dimensions actually
        evidenced -- nothing more sophisticated, so it can be cleanly
        superseded once the real scoring engine lands.
        """
        weighted_total = Decimal("0")
        weight_sum = Decimal("0")
        for dim, data in usable.items():
            weight = self._dimension_weights.get(dim, Decimal("1"))
            weighted_total += data.value * weight
            weight_sum += weight
        if weight_sum == 0:
            return None
        try:
            return self._clamp((weighted_total / weight_sum).quantize(Decimal("0.01")))
        except InvalidOperation:
            return None

    # -- Confidence --------------------------------------------------------

    def _confidence(self, usable: Mapping[AnalysisDimension, DimensionEvidence]) -> Decimal:
        """
        Confidence reflects both how much of the picture is filled in
        (completeness across the five dimensions) and how trustworthy
        the evidence that *is* present is (its average quality tier).
        It is intentionally independent of the resulting score: a
        small amount of high-quality evidence can outscore a large
        amount of low-quality evidence, and vice versa.
        """
        completeness = Decimal(len(usable)) / Decimal(len(self.DIMENSIONS))
        avg_quality = sum((self.QUALITY_WEIGHTS[data.quality] for data in usable.values()), Decimal("0")) / Decimal(
            len(usable)
        )
        confidence = (completeness * avg_quality * Decimal("100")).quantize(Decimal("0.01"))
        return self._clamp(confidence)

    # -- Recommendation ----------------------------------------------------

    def _recommend(
        self, overall_score: Decimal | None, confidence: Decimal
    ) -> MarketAnalysisRecommendation:
        if overall_score is None:
            return MarketAnalysisRecommendation.MONITOR
        if overall_score >= self.STRONG_OPPORTUNITY_SCORE and confidence >= self.STRONG_OPPORTUNITY_CONFIDENCE:
            return MarketAnalysisRecommendation.STRONG_OPPORTUNITY
        if overall_score < self.AVOID_SCORE and confidence >= self.AVOID_CONFIDENCE:
            return MarketAnalysisRecommendation.AVOID
        if overall_score >= self.INVESTIGATE_SCORE:
            return MarketAnalysisRecommendation.INVESTIGATE
        return MarketAnalysisRecommendation.MONITOR

    # -- Summaries / fallbacks ----------------------------------------------

    def _summarize(
        self,
        opportunity: ProductOpportunity,
        overall_score: Decimal | None,
        recommendation: MarketAnalysisRecommendation,
        missing: list[AnalysisDimension],
    ) -> str:
        parts = [f"Analysis for '{opportunity.name}'."]
        if overall_score is not None:
            parts.append(f"Overall opportunity score: {overall_score}/100.")
        parts.append(f"Recommendation: {MarketAnalysisRecommendation(recommendation).label}.")
        if missing:
            parts.append("Missing evidence for: " + ", ".join(dim.value for dim in missing) + ".")
        return " ".join(parts)

    def _insufficient_data_result(
        self, opportunity: ProductOpportunity, missing: list[AnalysisDimension]
    ) -> MarketAnalystResult:
        return MarketAnalystResult(
            status=AnalysisStatus.INSUFFICIENT_DATA,
            opportunity_id=str(opportunity.id),
            overall_score=None,
            recommendation=MarketAnalysisRecommendation.MONITOR,
            confidence_score=Decimal("0.00"),
            summary=(
                f"No usable evidence is currently available for '{opportunity.name}'. "
                "Awaiting evidence collection before a recommendation can be made."
            ),
            metrics={},
            missing_data=[dim.value for dim in missing],
            analysis_metadata={
                "engine_version": self.ENGINE_VERSION,
                "scoring_source": None,
                "dimensions_evaluated": [],
            },
        )

    @staticmethod
    def _clamp(value: Decimal) -> Decimal:
        return max(SCORE_MIN, min(SCORE_MAX, value))