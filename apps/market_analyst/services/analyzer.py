"""
apps.market_analyst.services.analyzer
========================================

High-level analytical composition for the Market Analyst pipeline.

Role
----
``Analyzer`` is the single entry point that turns already-collected,
raw evidence (trend observations, a competitor snapshot, customer
feedback, pricing figures) into the same ``MarketAnalystResult`` that
``MarketAnalystAgent.analyze()`` produces, by:

    1. Running each specialized service (`TrendAnalyzer`,
       `CompetitorAnalyzer`, `CustomerAnalyzer`, `PricingAnalyzer`) on
       the relevant slice of evidence.
    2. Converting the specialized results into the
       ``AnalysisDimension -> DimensionEvidence`` mapping
       ``MarketAnalystAgent.analyze()`` already expects.
    3. Calling the existing agent so scoring/confidence/recommendation
       logic is reused, not duplicated (see "Why this depends on the
       agent" below).
    4. Enriching the agent's result with an explainable narrative
       (strengths/weaknesses/risks) and a per-service breakdown, added
       to ``analysis_metadata`` without altering any field the agent
       already computed.

Analyzer never talks to an external API, an LLM, or the database, and
never recomputes a score or recommendation the agent already produced
-- it is a composition layer, not a second scoring engine.

Why this depends on the agent (not the other way around)
-----------------------------------------------------------
This build's file scope is limited to ``services/*.py`` --
``agents/market_analyst.py`` cannot be modified. That existing agent
already implements evidence partitioning, confidence calculation, and
recommendation thresholds; duplicating that logic here would violate
this app's "no duplicate scoring/analysis systems" requirement. So
rather than reimplementing (or bypassing) the agent, ``Analyzer``
depends on it directly: ``Analyzer.analyze()`` is the composition step
a management command/task is expected to call, and it calls
``MarketAnalystAgent.analyze()`` internally as its last, canonical
step. This keeps the dependency graph one-directional and acyclic --
the agent has no import-time dependency on this module or any other
service except its existing optional binding to
``services.opportunity_scorer.calculate_overall_score`` -- while still
giving a caller (management command, task, or future API) one
function to call for a complete analysis built from raw evidence.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from decimal import Decimal
from typing import Any

from apps.market_analyst.agents.market_analyst import (
    AnalysisDimension,
    DimensionEvidence,
    EvidenceQuality,
    MarketAnalystAgent,
    MarketAnalystResult,
)
from apps.market_analyst.models import ProductOpportunity
from apps.market_analyst.services.competitor_analyzer import (
    CompetitorAnalysisResult,
    CompetitorAnalyzer,
    CompetitorEvidence,
)
from apps.market_analyst.services.customer_analyzer import (
    CustomerAnalysisResult,
    CustomerAnalyzer,
    CustomerFeedbackItem,
)
from apps.market_analyst.services.opportunity_scorer import InvalidScoreError, OpportunityScorer
from apps.market_analyst.services.pricing_analyzer import PricingAnalysisResult, PricingAnalyzer, PricingEvidence
from apps.market_analyst.services.trend_analyzer import TrendAnalysisResult, TrendAnalyzer, TrendObservation

logger = logging.getLogger(__name__)

__all__ = ["AnalyzerInput", "Analyzer"]

_TREND_SOURCE = "services.trend_analyzer"
_COMPETITOR_SOURCE = "services.competitor_analyzer"
_PRICING_SOURCE = "services.pricing_analyzer"


@dataclass(frozen=True)
class AnalyzerInput:
    """
    Raw, already-collected evidence bundles for one `ProductOpportunity`.

    Every field is optional: `Analyzer` degrades gracefully (missing
    dimensions, reduced confidence, an ``insufficient_data`` status
    from the agent) rather than requiring a caller to supply evidence
    it doesn't yet have.
    """

    trend_observations: Sequence[TrendObservation] = field(default_factory=tuple)
    competitor_evidence: CompetitorEvidence | None = None
    customer_feedback: Sequence[CustomerFeedbackItem] = field(default_factory=tuple)
    pricing_evidence: PricingEvidence | None = None


class Analyzer:
    """
    Composes the specialized services into a single `MarketAnalystResult`.

    Usable independently of any particular caller
    (``Analyzer().analyze(opportunity, evidence)``); every collaborator
    is injectable for testing, defaulting to a fresh instance of each
    when not supplied. Holds no state beyond those collaborators and
    performs no I/O of its own beyond what the injected agent does
    (nothing -- `MarketAnalystAgent` performs no I/O either).
    """

    def __init__(
        self,
        *,
        trend_analyzer: TrendAnalyzer | None = None,
        competitor_analyzer: CompetitorAnalyzer | None = None,
        customer_analyzer: CustomerAnalyzer | None = None,
        pricing_analyzer: PricingAnalyzer | None = None,
        opportunity_scorer: OpportunityScorer | None = None,
        agent: MarketAnalystAgent | None = None,
    ) -> None:
        self._trend_analyzer = trend_analyzer or TrendAnalyzer()
        self._competitor_analyzer = competitor_analyzer or CompetitorAnalyzer()
        self._customer_analyzer = customer_analyzer or CustomerAnalyzer()
        self._pricing_analyzer = pricing_analyzer or PricingAnalyzer()
        self._opportunity_scorer = opportunity_scorer or OpportunityScorer()
        self._agent = agent or MarketAnalystAgent()

    def analyze(
        self,
        opportunity: ProductOpportunity,
        evidence: AnalyzerInput | None = None,
    ) -> MarketAnalystResult:
        """
        Run a full analysis pass for `opportunity` from raw evidence.

        Deterministic with respect to its inputs (aside from
        `TrendAnalyzer`'s freshness calculation, which is explicitly
        time-relative unless a fixed `as_of` is threaded through by the
        caller). Never persists anything -- the caller is responsible
        for turning the returned `MarketAnalystResult` into a
        `MarketAnalysis` row via `result.as_market_analysis_kwargs()`,
        exactly as `MarketAnalystAgent.analyze()` already documents.
        """
        evidence = evidence or AnalyzerInput()

        logger.info("analyzer: composition started opportunity_id=%s", opportunity.id)

        trend_result = self._trend_analyzer.analyze(evidence.trend_observations)
        competitor_result = self._competitor_analyzer.analyze(evidence.competitor_evidence)
        customer_result = self._customer_analyzer.analyze(evidence.customer_feedback)
        pricing_result = self._pricing_analyzer.analyze(evidence.pricing_evidence)

        dimension_evidence = self._build_dimension_evidence(trend_result, competitor_result, pricing_result)

        result = self._agent.analyze(opportunity, dimension_evidence)

        enriched = self._enrich(result, trend_result, competitor_result, customer_result, pricing_result)

        logger.info(
            "analyzer: composition finished opportunity_id=%s status=%s recommendation=%s",
            opportunity.id,
            enriched.status.value,
            enriched.recommendation,
        )
        return enriched

    # -- Evidence composition -----------------------------------------------

    def _build_dimension_evidence(
        self,
        trend_result: TrendAnalysisResult,
        competitor_result: CompetitorAnalysisResult,
        pricing_result: PricingAnalysisResult,
    ) -> dict[str, DimensionEvidence]:
        """
        Map each specialized service's output onto the exact
        `AnalysisDimension -> DimensionEvidence` mapping
        `MarketAnalystAgent.analyze()` expects. A dimension is omitted
        entirely (never included with a `None` value) whenever its
        source quality is `UNAVAILABLE`, so the agent's own
        missing-evidence handling applies unchanged.
        """
        evidence: dict[str, DimensionEvidence] = {}

        if trend_result.demand_quality != EvidenceQuality.UNAVAILABLE and trend_result.demand_score is not None:
            evidence[AnalysisDimension.DEMAND.value] = DimensionEvidence(
                value=trend_result.demand_score, quality=trend_result.demand_quality, source=_TREND_SOURCE
            )
        if trend_result.trend_quality != EvidenceQuality.UNAVAILABLE and trend_result.trend_score is not None:
            evidence[AnalysisDimension.TREND.value] = DimensionEvidence(
                value=trend_result.trend_score, quality=trend_result.trend_quality, source=_TREND_SOURCE
            )
        if pricing_result.margin_quality != EvidenceQuality.UNAVAILABLE and pricing_result.margin_score is not None:
            evidence[AnalysisDimension.MARGIN.value] = DimensionEvidence(
                value=pricing_result.margin_score, quality=pricing_result.margin_quality, source=_PRICING_SOURCE
            )
        if (
            competitor_result.competition_quality != EvidenceQuality.UNAVAILABLE
            and competitor_result.competition_score is not None
        ):
            evidence[AnalysisDimension.COMPETITION.value] = DimensionEvidence(
                value=competitor_result.competition_score,
                quality=competitor_result.competition_quality,
                source=_COMPETITOR_SOURCE,
            )
        if (
            competitor_result.saturation_quality != EvidenceQuality.UNAVAILABLE
            and competitor_result.saturation_score is not None
        ):
            evidence[AnalysisDimension.SATURATION.value] = DimensionEvidence(
                value=competitor_result.saturation_score,
                quality=competitor_result.saturation_quality,
                source=_COMPETITOR_SOURCE,
            )

        return evidence

    # -- Enrichment (additive only; never touches the agent's own fields) ---

    def _enrich(
        self,
        result: MarketAnalystResult,
        trend_result: TrendAnalysisResult,
        competitor_result: CompetitorAnalysisResult,
        customer_result: CustomerAnalysisResult,
        pricing_result: PricingAnalysisResult,
    ) -> MarketAnalystResult:
        strengths, weaknesses = self._strengths_and_weaknesses(result)
        risks = self._risks(competitor_result, pricing_result, customer_result)

        metadata: dict[str, Any] = dict(result.analysis_metadata)
        metadata["explainability"] = {
            "strengths": strengths,
            "weaknesses": weaknesses,
            "risks": risks,
        }
        metadata["specialized_analysis"] = {
            "trend": self._trend_metadata(trend_result),
            "competitor": self._competitor_metadata(competitor_result),
            "customer": self._customer_metadata(customer_result),
            "pricing": self._pricing_metadata(pricing_result),
        }

        return replace(result, analysis_metadata=metadata)

    def _strengths_and_weaknesses(self, result: MarketAnalystResult) -> tuple[list[str], list[str]]:
        """
        Delegates to `OpportunityScorer`'s centralized thresholding
        (see its module docstring) over the exact dimension values the
        agent scored, so "what counts as strong/weak" has one
        definition project-wide rather than a second copy here.
        """
        scores = {
            dimension: Decimal(metric["value"])
            for dimension, metric in result.metrics.items()
            if metric.get("value") is not None
        }
        if not scores:
            return [], []
        try:
            breakdown = self._opportunity_scorer.score(scores)
        except InvalidScoreError:
            logger.warning("analyzer: could not build a score breakdown for explainability; skipping.")
            return [], []
        return self._opportunity_scorer.strengths_and_weaknesses(breakdown)

    def _risks(
        self,
        competitor_result: CompetitorAnalysisResult,
        pricing_result: PricingAnalysisResult,
        customer_result: CustomerAnalysisResult,
    ) -> list[str]:
        risks: list[str] = []

        if competitor_result.competition_score is not None and competitor_result.competition_score < Decimal("40"):
            risks.append(
                f"High competitive intensity (competition score {competitor_result.competition_score}/100)."
            )
        if competitor_result.saturation_score is not None and competitor_result.saturation_score < Decimal("40"):
            risks.append(f"Category may be saturated (saturation score {competitor_result.saturation_score}/100).")
        if pricing_result.gross_margin_pct is not None and pricing_result.gross_margin_pct < Decimal("15"):
            risks.append(f"Thin gross margin ({pricing_result.gross_margin_pct}%).")
        if pricing_result.competitor_price_position.value == "above_market":
            risks.append("Candidate price sits above the observed competitor market.")
        for objection in customer_result.objections[:5]:
            risks.append(f"Customer objection: {objection}.")
        for pain_point in customer_result.pain_points[:5]:
            risks.append(f"Recurring customer pain point: {pain_point}.")

        return risks

    # -- Metadata serialization -----------------------------------------------

    @staticmethod
    def _trend_metadata(trend_result: TrendAnalysisResult) -> dict[str, Any]:
        return {
            "status": trend_result.status.value,
            "direction": trend_result.direction.value,
            "growth_rate_pct": str(trend_result.growth_rate_pct) if trend_result.growth_rate_pct is not None else None,
            "seasonality": trend_result.seasonality.value,
            "freshness_score": str(trend_result.freshness_score) if trend_result.freshness_score is not None else None,
            "observations_used": trend_result.observations_used,
            "confidence": str(trend_result.confidence),
            "warnings": trend_result.warnings,
        }

    @staticmethod
    def _competitor_metadata(competitor_result: CompetitorAnalysisResult) -> dict[str, Any]:
        return {
            "status": competitor_result.status.value,
            "competitor_count": competitor_result.competitor_count,
            "average_competitor_strength": (
                str(competitor_result.average_competitor_strength)
                if competitor_result.average_competitor_strength is not None
                else None
            ),
            "pricing_pressure": str(competitor_result.pricing_pressure) if competitor_result.pricing_pressure is not None else None,
            "differentiation_opportunity": (
                str(competitor_result.differentiation_opportunity)
                if competitor_result.differentiation_opportunity is not None
                else None
            ),
            "confidence": str(competitor_result.confidence),
            "warnings": competitor_result.warnings,
        }

    @staticmethod
    def _customer_metadata(customer_result: CustomerAnalysisResult) -> dict[str, Any]:
        return {
            "status": customer_result.status.value,
            "sentiment_score": str(customer_result.sentiment_score) if customer_result.sentiment_score is not None else None,
            "sample_size": customer_result.sample_size,
            "positive_themes": [t.theme for t in customer_result.positive_themes],
            "negative_themes": [t.theme for t in customer_result.negative_themes],
            "desired_features": customer_result.desired_features,
            "confidence": str(customer_result.confidence),
            "warnings": customer_result.warnings,
        }

    @staticmethod
    def _pricing_metadata(pricing_result: PricingAnalysisResult) -> dict[str, Any]:
        return {
            "status": pricing_result.status.value,
            "landed_cost": str(pricing_result.landed_cost) if pricing_result.landed_cost is not None else None,
            "gross_profit": str(pricing_result.gross_profit) if pricing_result.gross_profit is not None else None,
            "gross_margin_pct": str(pricing_result.gross_margin_pct) if pricing_result.gross_margin_pct is not None else None,
            "competitor_price_position": pricing_result.competitor_price_position.value,
            "recommended_price": str(pricing_result.recommended_price) if pricing_result.recommended_price is not None else None,
            "confidence": str(pricing_result.confidence),
            "warnings": pricing_result.warnings,
        }