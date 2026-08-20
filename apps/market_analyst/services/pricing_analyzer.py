"""
apps.market_analyst.services.pricing_analyzer
=================================================

Pricing and economic-viability analysis for a single candidate
product.

Consumes
--------
``PricingEvidence`` -- already-known cost/price figures supplied by
the caller (product cost, shipping cost, other known costs, a
candidate selling price, competitor prices). This module performs no
supplier/marketplace API calls; acquiring these figures is a data
provider's job (out of scope here).

Produces
--------
``PricingAnalysisResult``, covering the ``margin`` dimension
``ProductOpportunity`` persists, plus supplementary figures
(landed cost, gross profit, competitor price position, a cost-plus
recommended price) useful for the higher-level `Analyzer`'s narrative
explanation.

Financial precision
----------------------
Every monetary and percentage calculation here uses `Decimal`,
never binary floating point, per this app's financial-correctness
requirement. Division by a zero/negative selling price is guarded
explicitly rather than allowed to raise `ZeroDivisionError` or
silently produce a nonsensical negative-infinity margin.

No-fabrication discipline
---------------------------
* A cost component the caller did not supply is treated as unknown,
  never as zero -- `landed_cost` is only computed once `product_cost`
  is known (shipping/other costs default to 0 only when the caller
  supplies them as such; `None` means "unknown", not "free").
* `margin_score` is `unavailable` whenever `gross_margin_pct` cannot
  be computed (missing cost or price data, or a non-positive selling
  price) -- it is never defaulted to a placeholder score.
"""

from __future__ import annotations

import logging
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from enum import Enum

from apps.market_analyst.agents.market_analyst import AnalysisStatus, EvidenceQuality

logger = logging.getLogger(__name__)

__all__ = [
    "CompetitorPricePosition",
    "PricingEvidence",
    "PricingAnalysisResult",
    "PricingAnalyzer",
]

_QUANTIZE_MONEY = Decimal("0.01")
_QUANTIZE_SCORE = Decimal("0.01")
_SCORE_MIN = Decimal("0")
_SCORE_MAX = Decimal("100")


class CompetitorPricePosition(str, Enum):
    """Where the candidate selling price sits relative to observed competitor prices."""

    BELOW_MARKET = "below_market"
    MARKET_ALIGNED = "market_aligned"
    ABOVE_MARKET = "above_market"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass(frozen=True)
class PricingEvidence:
    """
    Already-known cost/price figures for one candidate product.

    Every field is optional and independently nullable: a missing
    field means "unknown", not "zero" (see module docstring).
    """

    product_cost: Decimal | None = None
    shipping_cost: Decimal | None = None
    other_costs: Decimal | None = None  # e.g. known taxes/transaction fees
    selling_price: Decimal | None = None
    competitor_prices: Sequence[Decimal] = field(default_factory=tuple)
    source: str = ""


@dataclass(frozen=True)
class PricingAnalysisResult:
    """Structured output of `PricingAnalyzer.analyze()`."""

    status: AnalysisStatus

    landed_cost: Decimal | None
    gross_profit: Decimal | None
    gross_margin_pct: Decimal | None
    margin_score: Decimal | None
    margin_quality: EvidenceQuality

    competitor_price_position: CompetitorPricePosition
    competitor_median_price: Decimal | None
    recommended_price: Decimal | None

    confidence: Decimal
    warnings: list[str] = field(default_factory=list)
    missing_data: list[str] = field(default_factory=list)


class PricingAnalyzer:
    """Stateless, deterministic analyzer over a `PricingEvidence` snapshot."""

    # margin_score reaches 100 at/above this gross margin percentage
    # and 0 at/below 0%; a simple, documented, linear mapping.
    MARGIN_SCORE_FULL_AT_PCT = Decimal("60")

    # A candidate price within this fraction of the competitor median
    # is considered "market aligned" rather than above/below market.
    MARKET_ALIGNED_BAND_PCT = Decimal("0.10")

    # Cost-plus target margin used only to *suggest* a price when the
    # caller didn't supply a candidate selling_price. Clearly reported
    # as a recommendation, never conflated with an observed price.
    TARGET_MARGIN_FOR_RECOMMENDATION_PCT = Decimal("50")

    def analyze(self, evidence: PricingEvidence | None) -> PricingAnalysisResult:
        if evidence is None:
            return self._insufficient_data_result(["No pricing evidence was supplied."])

        warnings: list[str] = []

        product_cost = self._validated_money("product_cost", evidence.product_cost, warnings)
        shipping_cost = self._validated_money("shipping_cost", evidence.shipping_cost, warnings)
        other_costs = self._validated_money("other_costs", evidence.other_costs, warnings)
        selling_price = self._validated_money("selling_price", evidence.selling_price, warnings, allow_none_ok=True)

        landed_cost = self._landed_cost(product_cost, shipping_cost, other_costs, warnings)
        gross_profit, gross_margin_pct = self._margin(landed_cost, selling_price, warnings)
        margin_score, margin_quality = self._margin_score(gross_margin_pct)

        competitor_prices = self._valid_competitor_prices(evidence.competitor_prices, warnings)
        competitor_median = self._median(competitor_prices)
        price_position = self._price_position(selling_price, competitor_median)

        recommended_price = None
        if selling_price is None and landed_cost is not None:
            recommended_price = self._cost_plus_price(landed_cost)
            warnings.append(
                f"No candidate selling_price supplied; recommended_price is a cost-plus estimate targeting "
                f"a {self.TARGET_MARGIN_FOR_RECOMMENDATION_PCT}% gross margin, not an observed price."
            )

        missing_data: list[str] = []
        if margin_quality == EvidenceQuality.UNAVAILABLE:
            missing_data.append("margin")

        status = AnalysisStatus.SUCCESS if not missing_data else AnalysisStatus.INSUFFICIENT_DATA
        confidence = self._confidence(margin_quality, price_position, competitor_prices)

        return PricingAnalysisResult(
            status=status,
            landed_cost=landed_cost,
            gross_profit=gross_profit,
            gross_margin_pct=gross_margin_pct,
            margin_score=margin_score,
            margin_quality=margin_quality,
            competitor_price_position=price_position,
            competitor_median_price=competitor_median,
            recommended_price=recommended_price,
            confidence=confidence,
            warnings=warnings,
            missing_data=missing_data,
        )

    # -- Validation ------------------------------------------------------

    def _validated_money(
        self, field_name: str, raw: Decimal | None, warnings: list[str], *, allow_none_ok: bool = False
    ) -> Decimal | None:
        if raw is None:
            return None
        try:
            value = Decimal(raw)
        except (InvalidOperation, TypeError, ValueError):
            warnings.append(f"Discarded non-numeric {field_name}: {raw!r}.")
            return None
        if not value.is_finite():
            warnings.append(f"Discarded non-finite {field_name}: {raw!r}.")
            return None
        if value < 0 and not allow_none_ok:
            warnings.append(f"Discarded negative {field_name}: {value!r}.")
            return None
        return value

    def _valid_competitor_prices(self, prices: Sequence[Decimal], warnings: list[str]) -> list[Decimal]:
        valid: list[Decimal] = []
        for price in prices:
            try:
                value = Decimal(price)
            except (InvalidOperation, TypeError, ValueError):
                warnings.append(f"Discarded non-numeric competitor price: {price!r}.")
                continue
            if not value.is_finite() or value <= 0:
                warnings.append(f"Discarded non-positive competitor price: {value!r}.")
                continue
            valid.append(value)
        return valid

    # -- Cost / margin -----------------------------------------------------

    def _landed_cost(
        self,
        product_cost: Decimal | None,
        shipping_cost: Decimal | None,
        other_costs: Decimal | None,
        warnings: list[str],
    ) -> Decimal | None:
        if product_cost is None:
            warnings.append("landed_cost unavailable: product_cost is unknown.")
            return None
        total = product_cost
        if shipping_cost is not None:
            total += shipping_cost
        if other_costs is not None:
            total += other_costs
        return total.quantize(_QUANTIZE_MONEY, rounding=ROUND_HALF_UP)

    def _margin(
        self, landed_cost: Decimal | None, selling_price: Decimal | None, warnings: list[str]
    ) -> tuple[Decimal | None, Decimal | None]:
        if landed_cost is None or selling_price is None:
            return None, None
        if selling_price <= 0:
            warnings.append(f"gross_margin_pct unavailable: selling_price must be positive, got {selling_price!r}.")
            return None, None

        gross_profit = (selling_price - landed_cost).quantize(_QUANTIZE_MONEY, rounding=ROUND_HALF_UP)
        gross_margin_pct = ((gross_profit / selling_price) * Decimal("100")).quantize(
            _QUANTIZE_SCORE, rounding=ROUND_HALF_UP
        )
        return gross_profit, gross_margin_pct

    def _margin_score(self, gross_margin_pct: Decimal | None) -> tuple[Decimal | None, EvidenceQuality]:
        if gross_margin_pct is None:
            return None, EvidenceQuality.UNAVAILABLE
        bounded = max(Decimal("0"), min(self.MARGIN_SCORE_FULL_AT_PCT, gross_margin_pct))
        score = ((bounded / self.MARGIN_SCORE_FULL_AT_PCT) * _SCORE_MAX).quantize(
            _QUANTIZE_SCORE, rounding=ROUND_HALF_UP
        )
        return max(_SCORE_MIN, min(_SCORE_MAX, score)), EvidenceQuality.CALCULATED

    # -- Competitor pricing -----------------------------------------------------

    def _median(self, prices: list[Decimal]) -> Decimal | None:
        if not prices:
            return None
        return Decimal(str(statistics.median([float(p) for p in prices]))).quantize(
            _QUANTIZE_MONEY, rounding=ROUND_HALF_UP
        )

    def _price_position(
        self, selling_price: Decimal | None, competitor_median: Decimal | None
    ) -> CompetitorPricePosition:
        if selling_price is None or competitor_median is None or competitor_median <= 0:
            return CompetitorPricePosition.INSUFFICIENT_DATA
        lower_band = competitor_median * (Decimal("1") - self.MARKET_ALIGNED_BAND_PCT)
        upper_band = competitor_median * (Decimal("1") + self.MARKET_ALIGNED_BAND_PCT)
        if selling_price < lower_band:
            return CompetitorPricePosition.BELOW_MARKET
        if selling_price > upper_band:
            return CompetitorPricePosition.ABOVE_MARKET
        return CompetitorPricePosition.MARKET_ALIGNED

    def _cost_plus_price(self, landed_cost: Decimal) -> Decimal:
        # selling_price = landed_cost / (1 - target_margin%) so that
        # (selling_price - landed_cost) / selling_price == target_margin%.
        target_fraction = self.TARGET_MARGIN_FOR_RECOMMENDATION_PCT / Decimal("100")
        denominator = Decimal("1") - target_fraction
        if denominator <= 0:
            return landed_cost
        price = (landed_cost / denominator).quantize(_QUANTIZE_MONEY, rounding=ROUND_HALF_UP)
        return price

    # -- Confidence -----------------------------------------------------

    def _confidence(
        self,
        margin_quality: EvidenceQuality,
        price_position: CompetitorPricePosition,
        competitor_prices: list[Decimal],
    ) -> Decimal:
        if margin_quality == EvidenceQuality.UNAVAILABLE:
            base = Decimal("0")
        else:
            base = Decimal("70")
        if price_position != CompetitorPricePosition.INSUFFICIENT_DATA:
            base += Decimal("20")
        if len(competitor_prices) >= 3:
            base += Decimal("10")
        return max(_SCORE_MIN, min(_SCORE_MAX, base)).quantize(_QUANTIZE_SCORE, rounding=ROUND_HALF_UP)

    # -- Fallbacks -----------------------------------------------------

    def _insufficient_data_result(self, warnings: list[str]) -> PricingAnalysisResult:
        return PricingAnalysisResult(
            status=AnalysisStatus.INSUFFICIENT_DATA,
            landed_cost=None,
            gross_profit=None,
            gross_margin_pct=None,
            margin_score=None,
            margin_quality=EvidenceQuality.UNAVAILABLE,
            competitor_price_position=CompetitorPricePosition.INSUFFICIENT_DATA,
            competitor_median_price=None,
            recommended_price=None,
            confidence=Decimal("0.00"),
            warnings=warnings,
            missing_data=["margin"],
        )