"""
apps.market_analyst.services.competitor_analyzer
====================================================

Competitive-intelligence analysis for a single candidate
product/category.

Consumes
--------
``CompetitorEvidence`` -- an already-collected snapshot of who else is
selling into this space, supplied by the caller. This module performs
no HTTP requests, no scraping, no browser automation, and no supplier/
marketplace API calls; acquiring competitor data is a data provider's
job (out of scope here).

Produces
--------
``CompetitorAnalysisResult``, covering two of the five persisted
dimensions -- ``competition`` and ``saturation`` -- plus supplementary,
non-persisted signals (`pricing_pressure`, `differentiation_opportunity`)
useful for the higher-level `Analyzer`'s narrative explanation.

Score direction (important)
-----------------------------
Per ``ProductOpportunity`` in ``models.py``, both persisted dimensions
here are *inverted* relative to the raw signal: "higher = less
competition" / "higher = less saturated" (higher is always better,
matching every other dimension). This module produces its scores
already in that convention -- callers never need to flip the sign
themselves, and `opportunity_scorer` never has to special-case these
two dimensions.

No-fabrication discipline
---------------------------
* Without a competitor count, `competition_score` is `unavailable` --
  it is never assumed to be 0 (maximum competition) or 100 (none).
* `saturation_score` uses a caller-supplied, dedicated saturation
  signal when available (`EvidenceQuality.OBSERVED`/`CALCULATED`,
  depending on how the caller labels its source); only when no
  dedicated signal exists does it fall back to a documented,
  lower-confidence estimate derived from competitor density
  (`EvidenceQuality.ESTIMATED`), with an explicit warning that it is
  a proxy, not a direct measurement.
"""

from __future__ import annotations

import logging
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

from apps.market_analyst.agents.market_analyst import AnalysisStatus, EvidenceQuality

logger = logging.getLogger(__name__)

__all__ = [
    "CompetitorObservation",
    "CompetitorEvidence",
    "CompetitorAnalysisResult",
    "CompetitorAnalyzer",
]

_QUANTIZE = Decimal("0.01")
_SCORE_MIN = Decimal("0")
_SCORE_MAX = Decimal("100")


@dataclass(frozen=True)
class CompetitorObservation:
    """A single known competitor, with whatever attributes are actually known."""

    name: str
    price: Decimal | None = None
    estimated_strength: Decimal | None = None  # 0-100, e.g. from review volume/rating/market-share signals
    source: str = ""


@dataclass(frozen=True)
class CompetitorEvidence:
    """
    Everything the caller has collected about the competitive landscape
    for one candidate product/category.

    ``competitor_count`` is the count of *meaningful* competitors as
    determined by whatever data provider collected it (e.g. active
    listings above a relevance threshold) -- this module does not
    second-guess that determination, only normalizes it into a score.

    ``category_saturation_index``, if supplied, is a dedicated
    saturation signal on a 0-100 scale where *higher means more
    saturated* (the natural/raw direction); this module inverts it
    when producing `saturation_score` (see module docstring).
    """

    competitor_count: int | None = None
    competitors: Sequence[CompetitorObservation] = field(default_factory=tuple)
    category_saturation_index: Decimal | None = None
    source: str = ""
    observed_at: datetime | None = None


@dataclass(frozen=True)
class CompetitorAnalysisResult:
    """Structured output of `CompetitorAnalyzer.analyze()`."""

    status: AnalysisStatus

    competition_score: Decimal | None  # higher = less competition
    competition_quality: EvidenceQuality
    saturation_score: Decimal | None  # higher = less saturated
    saturation_quality: EvidenceQuality

    competitor_count: int | None
    average_competitor_strength: Decimal | None
    pricing_pressure: Decimal | None  # 0-100, higher = tighter/more pressured pricing
    differentiation_opportunity: Decimal | None  # 0-100, higher = more room to differentiate

    confidence: Decimal
    warnings: list[str] = field(default_factory=list)
    missing_data: list[str] = field(default_factory=list)


class CompetitorAnalyzer:
    """Stateless, deterministic analyzer over a `CompetitorEvidence` snapshot."""

    # Competitor-count -> competition_score mapping: each meaningful
    # competitor costs this many points off a perfect (no-competition)
    # 100, floored at 0. 20 competitors fully saturates the score --
    # an explicit, documented, easy-to-change initial model rather
    # than a claimed empirical constant.
    PENALTY_PER_COMPETITOR = Decimal("5")
    MAX_COUNTED_COMPETITORS = 20

    # How much an above/below-average competitor strength shifts
    # competition_score, on top of the count-based base score.
    STRENGTH_ADJUSTMENT_WEIGHT = Decimal("0.30")

    def analyze(self, evidence: CompetitorEvidence | None) -> CompetitorAnalysisResult:
        if evidence is None:
            return self._insufficient_data_result(["No competitor evidence was supplied."])

        warnings: list[str] = []
        competitors = self._valid_competitors(evidence.competitors, warnings)

        competition_score, competition_quality = self._competition(evidence, competitors, warnings)
        saturation_score, saturation_quality = self._saturation(evidence, warnings)

        avg_strength = self._average_strength(competitors)
        pricing_pressure = self._pricing_pressure(competitors, warnings)
        differentiation = self._differentiation_opportunity(avg_strength, pricing_pressure)

        missing_data: list[str] = []
        if competition_quality == EvidenceQuality.UNAVAILABLE:
            missing_data.append("competition")
        if saturation_quality == EvidenceQuality.UNAVAILABLE:
            missing_data.append("saturation")

        if not missing_data:
            status = AnalysisStatus.SUCCESS
        elif len(missing_data) < 2:
            status = AnalysisStatus.PARTIAL
        else:
            status = AnalysisStatus.INSUFFICIENT_DATA

        confidence = self._confidence(evidence, competitors, competition_quality, saturation_quality)

        return CompetitorAnalysisResult(
            status=status,
            competition_score=competition_score,
            competition_quality=competition_quality,
            saturation_score=saturation_score,
            saturation_quality=saturation_quality,
            competitor_count=evidence.competitor_count,
            average_competitor_strength=avg_strength,
            pricing_pressure=pricing_pressure,
            differentiation_opportunity=differentiation,
            confidence=confidence,
            warnings=warnings,
            missing_data=missing_data,
        )

    # -- Validation ------------------------------------------------------

    def _valid_competitors(
        self, competitors: Sequence[CompetitorObservation], warnings: list[str]
    ) -> list[CompetitorObservation]:
        valid: list[CompetitorObservation] = []
        for competitor in competitors:
            if competitor.estimated_strength is not None:
                try:
                    strength = Decimal(competitor.estimated_strength)
                except (InvalidOperation, TypeError, ValueError):
                    warnings.append(f"Discarded non-numeric strength for competitor {competitor.name!r}.")
                    continue
                if not strength.is_finite() or strength < _SCORE_MIN or strength > _SCORE_MAX:
                    warnings.append(
                        f"Discarded out-of-range strength for competitor {competitor.name!r}: {strength!r}."
                    )
                    continue
            valid.append(competitor)
        return valid

    # -- Competition -----------------------------------------------------

    def _competition(
        self,
        evidence: CompetitorEvidence,
        competitors: list[CompetitorObservation],
        warnings: list[str],
    ) -> tuple[Decimal | None, EvidenceQuality]:
        if evidence.competitor_count is None:
            return None, EvidenceQuality.UNAVAILABLE
        if evidence.competitor_count < 0:
            warnings.append(f"Ignored negative competitor_count: {evidence.competitor_count!r}.")
            return None, EvidenceQuality.UNAVAILABLE

        counted = min(evidence.competitor_count, self.MAX_COUNTED_COMPETITORS)
        base_score = _SCORE_MAX - (Decimal(counted) * self.PENALTY_PER_COMPETITOR)
        base_score = max(_SCORE_MIN, base_score)

        avg_strength = self._average_strength(competitors)
        if avg_strength is not None:
            # A high average competitor strength pulls the score
            # further down (tougher competition); a low one pulls it
            # up slightly (weak incumbents), scaled by a documented
            # weight so count still dominates.
            strength_adjustment = (Decimal("50") - avg_strength) * self.STRENGTH_ADJUSTMENT_WEIGHT
            adjusted = base_score + strength_adjustment
            score = max(_SCORE_MIN, min(_SCORE_MAX, adjusted)).quantize(_QUANTIZE, rounding=ROUND_HALF_UP)
            return score, EvidenceQuality.CALCULATED

        return base_score.quantize(_QUANTIZE, rounding=ROUND_HALF_UP), EvidenceQuality.CALCULATED

    # -- Saturation -----------------------------------------------------

    def _saturation(
        self, evidence: CompetitorEvidence, warnings: list[str]
    ) -> tuple[Decimal | None, EvidenceQuality]:
        if evidence.category_saturation_index is not None:
            try:
                raw = Decimal(evidence.category_saturation_index)
            except (InvalidOperation, TypeError, ValueError):
                warnings.append(f"Ignored non-numeric category_saturation_index: {evidence.category_saturation_index!r}.")
                raw = None
            if raw is not None:
                if not raw.is_finite() or raw < _SCORE_MIN or raw > _SCORE_MAX:
                    warnings.append(f"Ignored out-of-range category_saturation_index: {raw!r}.")
                else:
                    # Caller's raw index is "higher = more saturated";
                    # invert to this app's "higher = less saturated"
                    # convention.
                    inverted = (_SCORE_MAX - raw).quantize(_QUANTIZE, rounding=ROUND_HALF_UP)
                    quality = EvidenceQuality.OBSERVED if evidence.source else EvidenceQuality.CALCULATED
                    return inverted, quality

        if evidence.competitor_count is not None and evidence.competitor_count >= 0:
            warnings.append(
                "No dedicated saturation evidence available; saturation_score was estimated from "
                "competitor density instead. Treat as a lower-confidence proxy, not a direct measurement."
            )
            counted = min(evidence.competitor_count, self.MAX_COUNTED_COMPETITORS)
            # A gentler per-competitor penalty than competition_score:
            # saturation is meant to reflect how "played out" the
            # concept is, which a raw count only weakly proxies.
            estimated = max(_SCORE_MIN, _SCORE_MAX - (Decimal(counted) * (self.PENALTY_PER_COMPETITOR * Decimal("0.6"))))
            return estimated.quantize(_QUANTIZE, rounding=ROUND_HALF_UP), EvidenceQuality.ESTIMATED

        return None, EvidenceQuality.UNAVAILABLE

    # -- Supplementary, non-persisted signals -----------------------------

    def _average_strength(self, competitors: list[CompetitorObservation]) -> Decimal | None:
        strengths = [c.estimated_strength for c in competitors if c.estimated_strength is not None]
        if not strengths:
            return None
        return (sum(strengths) / Decimal(len(strengths))).quantize(_QUANTIZE, rounding=ROUND_HALF_UP)

    def _pricing_pressure(self, competitors: list[CompetitorObservation], warnings: list[str]) -> Decimal | None:
        prices = [c.price for c in competitors if c.price is not None and c.price > 0]
        if len(prices) < 2:
            return None
        mean_price = sum(prices) / Decimal(len(prices))
        if mean_price == 0:
            return None
        stdev = Decimal(str(statistics.pstdev([float(p) for p in prices])))
        coefficient_of_variation = min(Decimal("1"), stdev / mean_price)
        # Tightly clustered prices (low CV) => high pricing pressure
        # (hard to win on price alone); widely spread prices => low
        # pressure (room to position).
        pressure = (_SCORE_MAX * (Decimal("1") - coefficient_of_variation)).quantize(
            _QUANTIZE, rounding=ROUND_HALF_UP
        )
        return max(_SCORE_MIN, min(_SCORE_MAX, pressure))

    def _differentiation_opportunity(
        self, avg_strength: Decimal | None, pricing_pressure: Decimal | None
    ) -> Decimal | None:
        if avg_strength is None and pricing_pressure is None:
            return None
        components = []
        if avg_strength is not None:
            # Weak incumbents (low avg strength) => more room to
            # differentiate on quality/positioning.
            components.append(_SCORE_MAX - avg_strength)
        if pricing_pressure is not None:
            # High pricing pressure => less room to differentiate on
            # price specifically.
            components.append(_SCORE_MAX - pricing_pressure)
        value = (sum(components) / Decimal(len(components))).quantize(_QUANTIZE, rounding=ROUND_HALF_UP)
        return max(_SCORE_MIN, min(_SCORE_MAX, value))

    # -- Confidence -----------------------------------------------------

    def _confidence(
        self,
        evidence: CompetitorEvidence,
        competitors: list[CompetitorObservation],
        competition_quality: EvidenceQuality,
        saturation_quality: EvidenceQuality,
    ) -> Decimal:
        quality_weights = {
            EvidenceQuality.OBSERVED: Decimal("1.00"),
            EvidenceQuality.CALCULATED: Decimal("0.85"),
            EvidenceQuality.ESTIMATED: Decimal("0.60"),
            EvidenceQuality.INFERRED: Decimal("0.40"),
            EvidenceQuality.UNAVAILABLE: Decimal("0.00"),
        }
        qualities = [competition_quality, saturation_quality]
        avg_quality = sum(quality_weights[q] for q in qualities) / Decimal(len(qualities))
        detail_bonus = Decimal("0.10") if competitors else Decimal("0")
        confidence = min(_SCORE_MAX, (avg_quality * _SCORE_MAX) + (detail_bonus * _SCORE_MAX)).quantize(
            _QUANTIZE, rounding=ROUND_HALF_UP
        )
        return max(_SCORE_MIN, confidence)

    # -- Fallbacks -----------------------------------------------------

    def _insufficient_data_result(self, warnings: list[str]) -> CompetitorAnalysisResult:
        return CompetitorAnalysisResult(
            status=AnalysisStatus.INSUFFICIENT_DATA,
            competition_score=None,
            competition_quality=EvidenceQuality.UNAVAILABLE,
            saturation_score=None,
            saturation_quality=EvidenceQuality.UNAVAILABLE,
            competitor_count=None,
            average_competitor_strength=None,
            pricing_pressure=None,
            differentiation_opportunity=None,
            confidence=Decimal("0.00"),
            warnings=warnings,
            missing_data=["competition", "saturation"],
        )