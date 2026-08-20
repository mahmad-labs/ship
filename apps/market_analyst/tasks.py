"""
apps.market_analyst.tasks
============================

Celery task entry points for running analysis on a `ProductOpportunity`.

Now that Celery + Redis are configured for this project, this module
wraps the analysis pipeline in a `@shared_task`. The task still accepts
`opportunity_id` (a serializable primary key, never a model instance)
so it stays safe to pass through the broker.

    receives opportunity_id
        -> loads ProductOpportunity
        -> builds DimensionEvidence from the opportunity's persisted
           score fields (see `_build_evidence_from_opportunity` below)
        -> hands that evidence to `MarketAnalystAgent.analyze()`, the
           orchestration layer in `apps.market_analyst.agents.market_analyst`
        -> persists the returned `MarketAnalystResult` as a `MarketAnalysis`
        -> returns a small, serializable result dict

Evidence sourcing
------------------
`MarketAnalystAgent` performs no I/O and collects no evidence itself
(see its module docstring) -- something upstream has to hand it
`DimensionEvidence`. Until a dedicated evidence-collection service
exists, this task builds that evidence directly from the five score
fields already persisted on `ProductOpportunity`
(`demand_score`, `trend_score`, `margin_score`, `competition_score`,
`saturation_score`). Each populated field becomes `EvidenceQuality.OBSERVED`
evidence, tagged with its source as "product_opportunity.<field>_score" --
these are real, already-stored values on the record, not estimates or
inferences, so OBSERVED is the accurate quality tier. A field left as
`None` becomes no evidence at all (not a fabricated zero), which is
exactly what `MarketAnalystAgent` expects for a genuinely missing
dimension: it will show up in the result's `missing_data`.

This is a stopgap, not a replacement for real evidence collection
(trend APIs, competitor scraping, etc.) -- when that lands as its own
service, only `_build_evidence_from_opportunity` needs to change; the
rest of this task's shape (call the agent, persist the result) stays
the same.

Unexpected errors are logged and re-raised so Celery's `autoretry_for`
/ retry policy can act on them; expected "nothing to do" outcomes
(missing record, duplicate trigger) are returned as plain result
dicts rather than raised, since they aren't retryable conditions.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from celery import shared_task
from celery.utils.log import get_task_logger
from django.core.exceptions import ValidationError
from django.db import OperationalError
from django.utils import timezone

from apps.market_analyst.agents.market_analyst import (
    AnalysisDimension,
    DimensionEvidence,
    EvidenceQuality,
    MarketAnalystAgent,
)
from apps.market_analyst.models import (
    MarketAnalysis,
    MarketAnalysisRecommendation,
    ProductOpportunity,
    ProductOpportunityStatus,
)

logger = get_task_logger(__name__)

# If a MarketAnalysis was already created for this opportunity within
# this window, a re-run is treated as a duplicate trigger (e.g. a
# retried task, or a duplicate message redelivered by the broker)
# rather than a legitimate re-analysis, and is skipped.
DUPLICATE_ANALYSIS_WINDOW = timedelta(minutes=5)

# Retry/backoff tuning for the task itself. Kept as module-level
# constants so they're easy to find and tune without hunting through
# decorator kwargs.
TASK_MAX_RETRIES = 3
TASK_RETRY_BACKOFF = True  # exponential backoff starting at 1s
TASK_RETRY_BACKOFF_MAX = 60  # seconds
TASK_RETRY_JITTER = True

# Maps each AnalysisDimension to the ProductOpportunity field that
# currently backs it. A 1:1 mirror of the model's score fields, same
# rationale as AnalysisDimension itself: there's nowhere else to pull
# an "observed" value from until a real evidence service exists.
_DIMENSION_SOURCE_FIELDS: dict[AnalysisDimension, str] = {
    AnalysisDimension.DEMAND: "demand_score",
    AnalysisDimension.TREND: "trend_score",
    AnalysisDimension.MARGIN: "margin_score",
    AnalysisDimension.COMPETITION: "competition_score",
    AnalysisDimension.SATURATION: "saturation_score",
}


def _build_evidence_from_opportunity(
    opportunity: ProductOpportunity,
) -> dict[str, DimensionEvidence]:
    """
    Build the `evidence` mapping `MarketAnalystAgent.analyze()` expects,
    sourced from `opportunity`'s own persisted score fields.

    Only fields that are actually populated (`is not None`) produce
    evidence, and that evidence is always `EvidenceQuality.OBSERVED` --
    these are real stored values, not estimates. Unpopulated fields are
    simply omitted; the agent treats an omitted dimension as missing
    rather than as a zero.
    """
    evidence: dict[str, DimensionEvidence] = {}
    for dimension, field_name in _DIMENSION_SOURCE_FIELDS.items():
        value = getattr(opportunity, field_name)
        if value is None:
            continue
        evidence[dimension.value] = DimensionEvidence(
            value=value,
            quality=EvidenceQuality.OBSERVED,
            source=f"product_opportunity.{field_name}",
        )
    return evidence


@shared_task(
    bind=True,
    name="market_analyst.analyze_product_opportunity",
    autoretry_for=(OperationalError,),
    max_retries=TASK_MAX_RETRIES,
    retry_backoff=TASK_RETRY_BACKOFF,
    retry_backoff_max=TASK_RETRY_BACKOFF_MAX,
    retry_jitter=TASK_RETRY_JITTER,
    acks_late=True,
)
def analyze_product_opportunity(self, opportunity_id: str) -> dict[str, Any]:
    """
    Run analysis for a single `ProductOpportunity` and persist the
    result as a `MarketAnalysis`.

    Accepts a primary key (str/UUID), never a model instance, since
    Celery serializes task arguments through the broker (Redis).

    Returns a small, serializable result dict rather than raising for
    expected outcomes (missing record, duplicate trigger). Unexpected
    errors are logged and re-raised, which triggers Celery's retry
    policy (see `autoretry_for` / `max_retries` above) -- transient
    DB hiccups (`OperationalError`) are retried automatically;
    everything else propagates to Celery after logging so it shows up
    as a FAILURE with a full traceback.

    `acks_late=True` means the broker only removes the message from
    the queue after this task finishes (success or final failure), so
    a worker that dies mid-run doesn't silently lose the job -- it
    gets redelivered. The duplicate-analysis guard below is what makes
    that redelivery safe rather than causing a double analysis.
    """
    try:
        opportunity = ProductOpportunity.objects.get(pk=opportunity_id)
    except (ProductOpportunity.DoesNotExist, ValueError, ValidationError):
        logger.warning("opportunity_id=%s not found", opportunity_id)
        return {"status": "not_found", "opportunity_id": str(opportunity_id)}

    recent_cutoff = timezone.now() - DUPLICATE_ANALYSIS_WINDOW
    if opportunity.analyses.filter(created_at__gte=recent_cutoff).exists():
        logger.info("skipping duplicate run for opportunity_id=%s", opportunity_id)
        return {"status": "skipped_duplicate", "opportunity_id": str(opportunity_id)}

    opportunity.status = ProductOpportunityStatus.ANALYZING
    opportunity.save(update_fields=["status", "updated_at"])

    try:
        evidence = _build_evidence_from_opportunity(opportunity)
        agent_result = MarketAnalystAgent().analyze(opportunity, evidence)
        analysis = MarketAnalysis.objects.create(
            opportunity=opportunity,
            **agent_result.as_market_analysis_kwargs(),
        )
    except OperationalError:
        # Let autoretry_for handle this: log with the current retry
        # count for visibility, then re-raise so Celery schedules
        # the retry with backoff.
        logger.warning(
            "analysis failed for opportunity_id=%s (attempt %s/%s), retrying",
            opportunity_id,
            self.request.retries + 1,
            TASK_MAX_RETRIES,
        )
        raise
    except Exception:
        logger.exception("analysis failed for opportunity_id=%s", opportunity_id)
        raise

    opportunity.status = (
        ProductOpportunityStatus.INVESTIGATE
        if analysis.recommendation
        in (MarketAnalysisRecommendation.STRONG_OPPORTUNITY, MarketAnalysisRecommendation.INVESTIGATE)
        else ProductOpportunityStatus.MONITOR
    )
    opportunity.save(update_fields=["status", "updated_at"])

    logger.info(
        "completed for opportunity_id=%s recommendation=%s",
        opportunity_id,
        analysis.recommendation,
    )
    return {
        "status": "completed",
        "opportunity_id": str(opportunity_id),
        "analysis_id": str(analysis.id),
        "recommendation": analysis.recommendation,
    }