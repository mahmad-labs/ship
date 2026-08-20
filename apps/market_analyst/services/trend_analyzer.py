"""
apps.market_analyst.services.trend_analyzer
==============================================

Demand and trend analysis for a single candidate product/category.

Consumes
--------
A sequence of ``TrendObservation`` -- already-collected, already-dated
interest/demand signals (e.g. a Google-Trends-style relative interest
index, or any other 0-100-normalized demand signal) supplied by the
caller. This module does not fetch that data itself: no HTTP client,
no scraping, no third-party SDK. Acquiring the observations is a data
provider's job (out of scope for this file); interpreting them is this
module's job.

Produces
--------
``TrendAnalysisResult``, covering *two* of the five dimensions
``ProductOpportunity`` persists: ``demand`` (how strong current
interest looks) and ``trend`` (whether it is rising, stable,
declining, or too volatile/thin to call). Both come from the same
observation series because they are two readings of the same
evidence -- absolute level vs. rate of change -- not two independent
data collections. This keeps the app's five dimensions mapped onto a
small, explainable set of services rather than inventing a sixth
"demand-only" service with nowhere obviously different to source data
from.

No-fabrication discipline
---------------------------
* An observation outside the expected [0, 100] interest scale is
  rejected (not clamped) and reported in ``warnings`` -- silently
  squashing an out-of-range number into a valid-looking score is
  exactly the "quietly invented intelligence" this app must avoid.
* A trend direction/score is only ever produced from
  ``MIN_OBSERVATIONS_FOR_TREND`` or more valid, distinct-timestamp
  observations. One or two points cannot support a growth claim (see
  module docstring on statistical significance in the build spec);
  with fewer, ``trend_score`` is ``None`` and quality is
  ``UNAVAILABLE``, never guessed.
* Seasonality is only ever reported as "detected"/"not_detected" when
  the observation span and density given in ``SEASONALITY_MIN_*``
  are actually met; otherwise it is reported as insufficient data.
"""

from __future__ import annotations

import logging
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from datetime import timezone as dt_timezone
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from enum import Enum

from apps.market_analyst.agents.market_analyst import AnalysisStatus, EvidenceQuality

logger = logging.getLogger(__name__)

__all__ = [
    "TrendDirection",
    "SeasonalityAssessment",
    "TrendObservation",
    "TrendAnalysisResult",
    "TrendAnalyzer",
]

_QUANTIZE = Decimal("0.01")
_SCORE_MIN = Decimal("0")
_SCORE_MAX = Decimal("100")


class TrendDirection(str, Enum):
    """Controlled vocabulary for the direction of a demand trend."""

    RISING = "rising"
    STABLE = "stable"
    DECLINING = "declining"
    VOLATILE = "volatile"
    INSUFFICIENT_DATA = "insufficient_data"


class SeasonalityAssessment(str, Enum):
    """Controlled vocabulary for seasonality, kept separate from `TrendDirection`."""

    DETECTED = "detected"
    NOT_DETECTED = "not_detected"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass(frozen=True)
class TrendObservation:
    """
    A single, already-collected demand/interest signal.

    ``value`` is expected on a 0-100 relative-interest scale (the same
    convention Google Trends and Ship's own score fields use), so
    observations from different sources remain comparable without this
    module inventing its own normalization scheme. A data provider
    that only has raw counts (e.g. absolute search volume) is
    responsible for normalizing them onto that scale before handing
    them to this analyzer.
    """

    timestamp: datetime
    value: Decimal
    source: str = ""


@dataclass(frozen=True)
class TrendAnalysisResult:
    """Structured output of `TrendAnalyzer.analyze()`."""

    status: AnalysisStatus

    demand_score: Decimal | None
    demand_quality: EvidenceQuality
    trend_score: Decimal | None
    trend_quality: EvidenceQuality

    direction: TrendDirection
    growth_rate_pct: Decimal | None
    momentum_pct_per_week: Decimal | None
    seasonality: SeasonalityAssessment
    freshness_score: Decimal | None

    confidence: Decimal
    observations_used: int
    warnings: list[str] = field(default_factory=list)
    missing_data: list[str] = field(default_factory=list)


class TrendAnalyzer:
    """
    Stateless, deterministic analyzer over a series of `TrendObservation`.

    Construct once, call `analyze()` as many times as needed; no
    per-call state is retained, no database access occurs, and the
    same observations always produce the same result (aside from
    `freshness_score`, which is explicitly a function of "now" unless
    an `as_of` is supplied -- see `analyze()`).
    """

    # A trend claim requires at least this many valid, distinct-time
    # observations; fewer cannot support a growth/decline conclusion.
    MIN_OBSERVATIONS_FOR_TREND = 3

    # demand_score is the average of the most recent N valid
    # observations, smoothing single-point noise without diluting
    # recency too far into the past.
    RECENT_WINDOW = 3

    RISING_THRESHOLD_PCT = Decimal("5")
    DECLINING_THRESHOLD_PCT = Decimal("-5")

    # Coefficient of variation (stdev / mean) above which a series is
    # called volatile rather than confidently rising/stable/declining.
    VOLATILITY_COEFFICIENT_THRESHOLD = Decimal("0.35")

    # Freshness decays linearly from 100 (at/under FRESH_WITHIN_DAYS)
    # to 0 (at/over STALE_AFTER_DAYS), based on the most recent valid
    # observation's age.
    FRESH_WITHIN_DAYS = 7
    STALE_AFTER_DAYS = 90

    # Seasonality is only ever assessed (not just defaulted to
    # "not_detected") once the series spans at least this long and has
    # at least this many valid observations.
    SEASONALITY_MIN_SPAN_DAYS = 365
    SEASONALITY_MIN_OBSERVATIONS = 12

    def analyze(
        self,
        observations: Sequence[TrendObservation],
        *,
        as_of: datetime | None = None,
    ) -> TrendAnalysisResult:
        """
        Analyze a demand/trend observation series.

        ``as_of`` defaults to the current time and only affects
        ``freshness_score`` (an explicitly time-relative concept);
        passing it explicitly makes freshness calculations
        reproducible in tests without needing to mock the clock.
        """
        as_of = as_of or datetime.now(dt_timezone.utc)
        warnings: list[str] = []

        valid = self._valid_observations(observations, warnings)

        if not valid:
            return self._insufficient_data_result(warnings)

        valid.sort(key=lambda obs: obs.timestamp)

        demand_score, demand_quality = self._demand(valid)
        trend_score, trend_quality, direction, growth_rate, momentum = self._trend(valid, warnings)
        seasonality = self._seasonality(valid)
        freshness_score = self._freshness(valid[-1].timestamp, as_of)
        confidence = self._confidence(valid, freshness_score, trend_quality)

        missing_data: list[str] = []
        if demand_quality == EvidenceQuality.UNAVAILABLE:
            missing_data.append("demand")
        if trend_quality == EvidenceQuality.UNAVAILABLE:
            missing_data.append("trend")

        status = (
            AnalysisStatus.SUCCESS
            if not missing_data
            else (AnalysisStatus.PARTIAL if demand_quality != EvidenceQuality.UNAVAILABLE else AnalysisStatus.INSUFFICIENT_DATA)
        )

        return TrendAnalysisResult(
            status=status,
            demand_score=demand_score,
            demand_quality=demand_quality,
            trend_score=trend_score,
            trend_quality=trend_quality,
            direction=direction,
            growth_rate_pct=growth_rate,
            momentum_pct_per_week=momentum,
            seasonality=seasonality,
            freshness_score=freshness_score,
            confidence=confidence,
            observations_used=len(valid),
            warnings=warnings,
            missing_data=missing_data,
        )

    # -- Validation ------------------------------------------------------

    def _valid_observations(
        self, observations: Sequence[TrendObservation], warnings: list[str]
    ) -> list[TrendObservation]:
        valid: list[TrendObservation] = []
        for obs in observations:
            try:
                value = Decimal(obs.value)
            except (InvalidOperation, TypeError, ValueError):
                warnings.append(f"Discarded observation with non-numeric value: {obs.value!r}.")
                continue
            if not value.is_finite() or value < _SCORE_MIN or value > _SCORE_MAX:
                warnings.append(
                    f"Discarded observation outside the expected [0, 100] interest scale: {value!r}."
                )
                continue
            if obs.timestamp is None:
                warnings.append("Discarded observation with no timestamp.")
                continue
            valid.append(obs)
        return valid

    # -- Demand ------------------------------------------------------------

    def _demand(self, valid: list[TrendObservation]) -> tuple[Decimal | None, EvidenceQuality]:
        window = valid[-self.RECENT_WINDOW :]
        if len(window) == 1:
            return window[0].value.quantize(_QUANTIZE), EvidenceQuality.OBSERVED
        mean_value = (sum(obs.value for obs in window) / Decimal(len(window))).quantize(
            _QUANTIZE, rounding=ROUND_HALF_UP
        )
        return mean_value, EvidenceQuality.CALCULATED

    # -- Trend ---------------------------------------------------------------

    def _trend(
        self, valid: list[TrendObservation], warnings: list[str]
    ) -> tuple[Decimal | None, EvidenceQuality, TrendDirection, Decimal | None, Decimal | None]:
        if len(valid) < self.MIN_OBSERVATIONS_FOR_TREND:
            warnings.append(
                f"Only {len(valid)} valid observation(s); at least {self.MIN_OBSERVATIONS_FOR_TREND} are "
                "required to report a trend direction/score. No trend claim was made."
            )
            return None, EvidenceQuality.UNAVAILABLE, TrendDirection.INSUFFICIENT_DATA, None, None

        first, last = valid[0], valid[-1]
        growth_rate = self._growth_rate_pct(first.value, last.value)

        values = [obs.value for obs in valid]
        mean_value = sum(values) / Decimal(len(values))
        coefficient_of_variation = self._coefficient_of_variation(values, mean_value)

        momentum = self._weekly_momentum_pct(valid)

        if coefficient_of_variation is not None and coefficient_of_variation > self.VOLATILITY_COEFFICIENT_THRESHOLD:
            direction = TrendDirection.VOLATILE
        elif growth_rate is not None and growth_rate >= self.RISING_THRESHOLD_PCT:
            direction = TrendDirection.RISING
        elif growth_rate is not None and growth_rate <= self.DECLINING_THRESHOLD_PCT:
            direction = TrendDirection.DECLINING
        else:
            direction = TrendDirection.STABLE

        trend_score = self._trend_score(growth_rate, direction)
        return trend_score, EvidenceQuality.CALCULATED, direction, growth_rate, momentum

    def _growth_rate_pct(self, first_value: Decimal, last_value: Decimal) -> Decimal | None:
        if first_value == 0:
            # Cannot express growth from a zero baseline as a
            # percentage without dividing by zero; report as
            # unavailable rather than fabricating an infinite/undefined
            # rate.
            return None
        rate = ((last_value - first_value) / first_value * Decimal("100")).quantize(
            _QUANTIZE, rounding=ROUND_HALF_UP
        )
        return rate

    def _coefficient_of_variation(self, values: list[Decimal], mean_value: Decimal) -> Decimal | None:
        if len(values) < 2 or mean_value == 0:
            return None
        stdev = Decimal(str(statistics.pstdev([float(v) for v in values])))
        return (stdev / mean_value).quantize(_QUANTIZE, rounding=ROUND_HALF_UP)

    def _weekly_momentum_pct(self, valid: list[TrendObservation]) -> Decimal | None:
        """
        Simple linear-regression slope of value against time (in
        weeks), expressed as percent-of-mean-value per week. A
        smoother, less endpoint-sensitive companion to
        `growth_rate_pct` (which only looks at the first/last point).
        """
        span_days = (valid[-1].timestamp - valid[0].timestamp).total_seconds() / 86400
        if span_days <= 0:
            return None

        mean_value = sum(obs.value for obs in valid) / Decimal(len(valid))
        if mean_value == 0:
            return None

        t0 = valid[0].timestamp
        xs = [Decimal(str((obs.timestamp - t0).total_seconds() / 86400)) for obs in valid]
        ys = [obs.value for obs in valid]
        n = Decimal(len(valid))
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n

        numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        denominator = sum((x - mean_x) ** 2 for x in xs)
        if denominator == 0:
            return None

        slope_per_day = numerator / denominator  # value-units per day
        weekly_pct = (slope_per_day * Decimal("7") / mean_value * Decimal("100")).quantize(
            _QUANTIZE, rounding=ROUND_HALF_UP
        )
        return weekly_pct

    def _trend_score(self, growth_rate: Decimal | None, direction: TrendDirection) -> Decimal | None:
        if direction == TrendDirection.VOLATILE:
            # A volatile series is explicitly *not* a confident
            # rising/declining call; score it around neutral rather
            # than letting a noisy endpoint-to-endpoint growth figure
            # dominate.
            return Decimal("50.00")
        if growth_rate is None:
            return None
        # 0% growth -> neutral 50; +/-100% (or more, clamped) growth ->
        # 100/0. A simple, explainable, bounded transform -- not a
        # claim of statistical calibration.
        bounded = max(Decimal("-100"), min(Decimal("100"), growth_rate))
        score = (Decimal("50") + bounded / Decimal("2")).quantize(_QUANTIZE, rounding=ROUND_HALF_UP)
        return max(_SCORE_MIN, min(_SCORE_MAX, score))

    # -- Seasonality -----------------------------------------------------

    def _seasonality(self, valid: list[TrendObservation]) -> SeasonalityAssessment:
        span_days = (valid[-1].timestamp - valid[0].timestamp).total_seconds() / 86400
        if span_days < self.SEASONALITY_MIN_SPAN_DAYS or len(valid) < self.SEASONALITY_MIN_OBSERVATIONS:
            return SeasonalityAssessment.INSUFFICIENT_DATA

        by_month: dict[int, list[Decimal]] = {}
        for obs in valid:
            by_month.setdefault(obs.timestamp.month, []).append(obs.value)
        if len(by_month) < 6:
            # Too few distinct calendar months actually represented to
            # say anything about month-to-month seasonality, even
            # though the raw span/count thresholds were met.
            return SeasonalityAssessment.INSUFFICIENT_DATA

        monthly_means = [sum(vals) / Decimal(len(vals)) for vals in by_month.values()]
        overall_mean = sum(obs.value for obs in valid) / Decimal(len(valid))
        if overall_mean == 0:
            return SeasonalityAssessment.INSUFFICIENT_DATA

        monthly_stdev = Decimal(str(statistics.pstdev([float(m) for m in monthly_means])))
        relative_spread = monthly_stdev / overall_mean
        # Heuristic threshold: month-to-month averages varying by more
        # than 20% of the overall mean is treated as a seasonal
        # pattern. Documented as a heuristic, not a statistical test.
        return SeasonalityAssessment.DETECTED if relative_spread > Decimal("0.20") else SeasonalityAssessment.NOT_DETECTED

    # -- Freshness -----------------------------------------------------

    def _freshness(self, last_timestamp: datetime, as_of: datetime) -> Decimal:
        age_days = max(Decimal("0"), Decimal(str((as_of - last_timestamp).total_seconds() / 86400)))
        if age_days <= self.FRESH_WITHIN_DAYS:
            return _SCORE_MAX
        if age_days >= self.STALE_AFTER_DAYS:
            return _SCORE_MIN
        span = Decimal(self.STALE_AFTER_DAYS - self.FRESH_WITHIN_DAYS)
        decayed = _SCORE_MAX * (Decimal(1) - (age_days - self.FRESH_WITHIN_DAYS) / span)
        return decayed.quantize(_QUANTIZE, rounding=ROUND_HALF_UP)

    # -- Confidence -----------------------------------------------------

    def _confidence(
        self, valid: list[TrendObservation], freshness_score: Decimal, trend_quality: EvidenceQuality
    ) -> Decimal:
        """
        Confidence blends three independent signals: how much data
        there is (relative to a comfortable sample size), how fresh
        it is, and whether a trend claim could even be made at all.
        Deliberately independent of the score values themselves.
        """
        sample_adequacy = min(Decimal("1"), Decimal(len(valid)) / Decimal(10))
        trend_available = Decimal("1") if trend_quality != EvidenceQuality.UNAVAILABLE else Decimal("0.5")
        confidence = (sample_adequacy * (freshness_score / _SCORE_MAX) * trend_available * _SCORE_MAX).quantize(
            _QUANTIZE, rounding=ROUND_HALF_UP
        )
        return max(_SCORE_MIN, min(_SCORE_MAX, confidence))

    # -- Fallbacks -----------------------------------------------------

    def _insufficient_data_result(self, warnings: list[str]) -> TrendAnalysisResult:
        warnings.append("No usable trend observations were supplied.")
        return TrendAnalysisResult(
            status=AnalysisStatus.INSUFFICIENT_DATA,
            demand_score=None,
            demand_quality=EvidenceQuality.UNAVAILABLE,
            trend_score=None,
            trend_quality=EvidenceQuality.UNAVAILABLE,
            direction=TrendDirection.INSUFFICIENT_DATA,
            growth_rate_pct=None,
            momentum_pct_per_week=None,
            seasonality=SeasonalityAssessment.INSUFFICIENT_DATA,
            freshness_score=None,
            confidence=Decimal("0.00"),
            observations_used=0,
            warnings=warnings,
            missing_data=["demand", "trend"],
        )