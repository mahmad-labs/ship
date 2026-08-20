"""
apps.market_analyst.management.commands.analyze_market
=========================================================

Operational CLI entry point for manually triggering Market Analyst
analysis:

    python manage.py analyze_market --opportunity <uuid>
    python manage.py analyze_market --all [--market <uuid>] [--limit N]

This command is a thin orchestration layer. It parses arguments,
resolves and validates its target(s), and delegates the actual
analysis to the existing `MarketAnalystAgent`
(`apps.market_analyst.agents.market_analyst`). It contains no
scoring, trend, competition, pricing, customer, or saturation
algorithm of its own -- see that module for the analysis engine and
`apps.market_analyst.models` for what is persisted.

Evidence
--------
`MarketAnalystAgent.analyze()` takes an evidence mapping built by a
caller; the agent explicitly does not collect evidence itself (see its
module docstring). Ship does not yet have a dedicated evidence
collection service -- `apps.market_analyst.services.*` are still empty
placeholders. Until that service exists, the only real, already-vetted
signal available to this command is whatever component scores
(`demand_score`, `trend_score`, `margin_score`, `competition_score`,
`saturation_score`) already live on the `ProductOpportunity` row --
the same five fields `apps.market_analyst.tasks._run_placeholder_analysis`
inspects today. This command reads those fields as-is and wraps each
non-null one in a `DimensionEvidence(quality=CALCULATED)` for the
agent; it never estimates, invents, or fetches a value from anywhere
else. Opportunities with no component scores yet simply yield an
`AnalysisStatus.INSUFFICIENT_DATA` result from the agent -- exactly
the outcome the agent is designed to produce in that case.

Persistence
-----------
The agent never writes to the database (see its docstring). Its own
module docstring documents the intended caller-side persistence
pattern:

    MarketAnalysis.objects.create(
        opportunity=opportunity,
        **result.as_market_analysis_kwargs(),
    )

This command follows that documented pattern directly. It also
mirrors the opportunity status transition and duplicate-analysis
idempotency window already established in
`apps.market_analyst.tasks.analyze_product_opportunity` (reusing its
`DUPLICATE_ANALYSIS_WINDOW` constant rather than inventing a second
value), since `tasks.py` is out of this change's file scope and
cannot be modified to expose that behaviour as a shared helper.
`--dry-run` skips this section entirely: `agent.analyze()` itself has
no side effects, so a dry run is genuine, not simulated.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict
from decimal import Decimal
from typing import Any

from django.core.management.base import BaseCommand, CommandError, CommandParser
from django.db import transaction
from django.utils import timezone

from apps.market_analyst.agents.market_analyst import (
    AnalysisDimension,
    AnalysisStatus,
    DimensionEvidence,
    EvidenceQuality,
    MarketAnalystAgent,
    MarketAnalystResult,
)
from apps.market_analyst.models import (
    Market,
    MarketAnalysis,
    MarketAnalysisRecommendation,
    ProductOpportunity,
    ProductOpportunityStatus,
)
from apps.market_analyst.tasks import DUPLICATE_ANALYSIS_WINDOW

logger = logging.getLogger(__name__)

# Statuses eligible for (re-)analysis under `--all`. Terminal/settled
# states (approved, rejected, archived) are excluded so a batch run
# never touches a decision that has already been made. `ANALYZING` is
# also excluded -- it marks an opportunity currently mid-flight in
# another run, not one waiting to be picked up.
ELIGIBLE_STATUSES: tuple[str, ...] = (
    ProductOpportunityStatus.DISCOVERED,
    ProductOpportunityStatus.INVESTIGATE,
    ProductOpportunityStatus.MONITOR,
)

# The five component-score fields this command is allowed to read off
# ProductOpportunity and forward to the agent as evidence. Mirrors
# AnalysisDimension exactly -- see module docstring.
_SCORE_FIELD_BY_DIMENSION: dict[AnalysisDimension, str] = {
    AnalysisDimension.DEMAND: "demand_score",
    AnalysisDimension.TREND: "trend_score",
    AnalysisDimension.MARGIN: "margin_score",
    AnalysisDimension.COMPETITION: "competition_score",
    AnalysisDimension.SATURATION: "saturation_score",
}

# Mirrors apps.market_analyst.tasks.analyze_product_opportunity's status
# transition on a completed analysis.
_STATUS_BY_RECOMMENDATION: dict[str, str] = {
    MarketAnalysisRecommendation.STRONG_OPPORTUNITY: ProductOpportunityStatus.INVESTIGATE,
    MarketAnalysisRecommendation.INVESTIGATE: ProductOpportunityStatus.INVESTIGATE,
    MarketAnalysisRecommendation.MONITOR: ProductOpportunityStatus.MONITOR,
    MarketAnalysisRecommendation.AVOID: ProductOpportunityStatus.MONITOR,
}


def _parse_uuid(raw: str, *, label: str) -> uuid.UUID:
    try:
        return uuid.UUID(str(raw))
    except (ValueError, AttributeError, TypeError) as exc:
        raise CommandError(f"Invalid {label}: {raw!r} is not a valid UUID.") from exc


def _build_evidence(opportunity: ProductOpportunity) -> dict[str, DimensionEvidence]:
    """
    Wrap whatever component scores already exist on `opportunity` as
    agent evidence. Pure data mapping -- no calculation, estimation,
    or scoring happens here.
    """
    evidence: dict[str, DimensionEvidence] = {}
    for dimension, field_name in _SCORE_FIELD_BY_DIMENSION.items():
        value = getattr(opportunity, field_name)
        if value is not None:
            evidence[dimension.value] = DimensionEvidence(
                value=value,
                quality=EvidenceQuality.CALCULATED,
                source="product_opportunity_component_score",
            )
    return evidence


def _json_default(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, uuid.UUID):
        return str(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _result_to_dict(opportunity: ProductOpportunity, result: MarketAnalystResult) -> dict[str, Any]:
    data = asdict(result)
    data["status"] = result.status.value
    data["opportunity_name"] = opportunity.name
    data["market_id"] = str(opportunity.market_id)
    return data


class Command(BaseCommand):
    help = (
        "Run Market Analyst analysis for a product opportunity via the existing "
        "MarketAnalystAgent. Targets a single opportunity (--opportunity), or a "
        "batch of eligible opportunities (--all, optionally scoped with --market "
        "and --limit). Supports --dry-run (no persistence), --force (bypass the "
        "recent-duplicate-analysis skip), --json (machine-readable output), and "
        "the standard --verbosity flag."
    )

    def add_arguments(self, parser: CommandParser) -> None:
        target = parser.add_mutually_exclusive_group(required=True)
        target.add_argument(
            "--opportunity",
            metavar="UUID",
            help="Analyze a single ProductOpportunity by id.",
        )
        target.add_argument(
            "--all",
            action="store_true",
            help=(
                "Analyze all eligible opportunities "
                f"(status in: {', '.join(s.label for s in ELIGIBLE_STATUSES)}). "
                "Combine with --market and/or --limit to narrow the batch."
            ),
        )
        parser.add_argument(
            "--market",
            metavar="UUID",
            help="Restrict --all to opportunities in this Market. Ignored with --opportunity.",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=None,
            metavar="N",
            help="Maximum number of opportunities to analyze under --all. Must be a positive integer.",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help=(
                "Validate target(s) and run the agent, but do not persist a "
                "MarketAnalysis or change opportunity status. The agent itself "
                "performs no database writes, so this is a genuine dry run."
            ),
        )
        parser.add_argument(
            "--force",
            action="store_true",
            help=(
                "Analyze even if a MarketAnalysis was already created for the "
                f"opportunity within the last {int(DUPLICATE_ANALYSIS_WINDOW.total_seconds() // 60)} "
                "minutes (the same duplicate-run window used by the existing task layer). "
                "Does not bypass any other validation."
            ),
        )
        parser.add_argument(
            "--json",
            action="store_true",
            help="Emit machine-readable JSON to stdout instead of human-readable text.",
        )

    def handle(self, *args: Any, **options: Any) -> None:
        opportunity_arg: str | None = options["opportunity"]
        run_all: bool = options["all"]
        market_arg: str | None = options["market"]
        limit: int | None = options["limit"]
        dry_run: bool = options["dry_run"]
        force: bool = options["force"]
        as_json: bool = options["json"]
        verbosity: int = options["verbosity"]

        if limit is not None and limit <= 0:
            raise CommandError("--limit must be a positive integer.")
        if market_arg is not None and not run_all:
            raise CommandError("--market can only be used together with --all.")

        agent = MarketAnalystAgent()

        if opportunity_arg is not None:
            opportunity_id = _parse_uuid(opportunity_arg, label="--opportunity")
            outcome = self._analyze_one(
                agent,
                opportunity_id=opportunity_id,
                dry_run=dry_run,
                force=force,
                verbosity=verbosity,
                as_json=as_json,
            )
            if as_json:
                self.stdout.write(json.dumps(outcome, default=_json_default))
            if outcome["outcome"] == "error":
                raise CommandError(outcome["message"])
            return

        # --all
        market_id: uuid.UUID | None = None
        if market_arg is not None:
            market_id = _parse_uuid(market_arg, label="--market")
            if not Market.objects.filter(pk=market_id).exists():
                raise CommandError(f"No Market found with id {market_id}.")

        self._analyze_all(
            agent,
            market_id=market_id,
            limit=limit,
            dry_run=dry_run,
            force=force,
            verbosity=verbosity,
            as_json=as_json,
        )

    # -- Single opportunity ------------------------------------------------

    def _analyze_one(
        self,
        agent: MarketAnalystAgent,
        *,
        opportunity_id: uuid.UUID,
        dry_run: bool,
        force: bool,
        verbosity: int,
        as_json: bool,
    ) -> dict[str, Any]:
        try:
            opportunity = ProductOpportunity.objects.select_related("market").get(pk=opportunity_id)
        except ProductOpportunity.DoesNotExist:
            return {
                "outcome": "error",
                "opportunity_id": str(opportunity_id),
                "message": f"No ProductOpportunity found with id {opportunity_id}.",
            }

        if not force:
            skip_reason = self._duplicate_skip_reason(opportunity)
            if skip_reason is not None:
                if not as_json:
                    self.stdout.write(self.style.WARNING(skip_reason))
                return {
                    "outcome": "skipped_duplicate",
                    "opportunity_id": str(opportunity_id),
                    "message": skip_reason,
                }

        if not as_json and verbosity >= 1:
            self.stdout.write(f"Starting Market Analyst... (dry-run={dry_run})")

        try:
            evidence = _build_evidence(opportunity)
            result = agent.analyze(opportunity, evidence)
        except Exception as exc:  # noqa: BLE001 - surfaced via CommandError/stderr, not swallowed
            logger.exception("analyze_market: agent failed for opportunity_id=%s", opportunity_id)
            message = f"Analysis failed for opportunity {opportunity_id}: {exc}"
            if not as_json:
                self.stderr.write(self.style.ERROR(message))
            return {"outcome": "error", "opportunity_id": str(opportunity_id), "message": message}

        analysis_id: str | None = None
        if not dry_run:
            analysis_id = self._persist(opportunity, result)

        if not as_json:
            self._write_human_result(opportunity, result, dry_run=dry_run, verbosity=verbosity)

        payload = _result_to_dict(opportunity, result)
        payload["outcome"] = "success"
        payload["dry_run"] = dry_run
        payload["analysis_id"] = analysis_id
        return payload

    # -- Batch ---------------------------------------------------------------

    def _analyze_all(
        self,
        agent: MarketAnalystAgent,
        *,
        market_id: uuid.UUID | None,
        limit: int | None,
        dry_run: bool,
        force: bool,
        verbosity: int,
        as_json: bool,
    ) -> None:
        queryset = (
            ProductOpportunity.objects.select_related("market")
            .filter(status__in=ELIGIBLE_STATUSES)
            .order_by("-created_at")
        )
        if market_id is not None:
            queryset = queryset.filter(market_id=market_id)
        if limit is not None:
            queryset = queryset[:limit]

        total = 0
        successful = 0
        failed = 0
        skipped = 0
        results: list[dict[str, Any]] = []

        for opportunity in queryset.iterator():
            total += 1

            if not force:
                skip_reason = self._duplicate_skip_reason(opportunity)
                if skip_reason is not None:
                    skipped += 1
                    if not as_json and verbosity >= 2:
                        self.stdout.write(self.style.WARNING(f"[{opportunity.id}] {skip_reason}"))
                    results.append({"outcome": "skipped_duplicate", "opportunity_id": str(opportunity.id)})
                    continue

            try:
                evidence = _build_evidence(opportunity)
                result = agent.analyze(opportunity, evidence)
            except Exception as exc:  # noqa: BLE001 - one bad opportunity must not abort the batch
                failed += 1
                logger.exception("analyze_market: agent failed for opportunity_id=%s", opportunity.id)
                if not as_json:
                    self.stderr.write(self.style.ERROR(f"[{opportunity.id}] Analysis failed: {exc}"))
                results.append({"outcome": "error", "opportunity_id": str(opportunity.id), "message": str(exc)})
                continue

            analysis_id: str | None = None
            if not dry_run:
                analysis_id = self._persist(opportunity, result)

            successful += 1
            if not as_json and verbosity >= 1:
                self._write_human_result(opportunity, result, dry_run=dry_run, verbosity=verbosity)

            payload = _result_to_dict(opportunity, result)
            payload["outcome"] = "success"
            payload["dry_run"] = dry_run
            payload["analysis_id"] = analysis_id
            results.append(payload)

        summary = {
            "total": total,
            "successful": successful,
            "failed": failed,
            "skipped": skipped,
            "dry_run": dry_run,
        }

        if as_json:
            self.stdout.write(json.dumps({"summary": summary, "results": results}, default=_json_default))
        else:
            self.stdout.write("")
            self.stdout.write(
                self.style.SUCCESS(
                    f"Done. Total: {total}  Successful: {successful}  Failed: {failed}  Skipped: {skipped}"
                )
            )

        if total > 0 and successful == 0 and (failed > 0 or skipped == total):
            raise CommandError(
                f"No opportunities were successfully analyzed (total={total}, failed={failed}, skipped={skipped})."
            )

    # -- Shared helpers ------------------------------------------------------

    def _duplicate_skip_reason(self, opportunity: ProductOpportunity) -> str | None:
        """
        Mirrors apps.market_analyst.tasks.analyze_product_opportunity's
        duplicate-run guard: skip if a MarketAnalysis already exists for
        this opportunity within DUPLICATE_ANALYSIS_WINDOW. Reuses that
        module's constant rather than a second hardcoded value.
        """
        recent_cutoff = timezone.now() - DUPLICATE_ANALYSIS_WINDOW
        if opportunity.analyses.filter(created_at__gte=recent_cutoff).exists():
            minutes = int(DUPLICATE_ANALYSIS_WINDOW.total_seconds() // 60)
            return (
                f"Skipping {opportunity.id}: a MarketAnalysis was already created "
                f"within the last {minutes} minutes. Use --force to re-analyze anyway."
            )
        return None

    def _persist(self, opportunity: ProductOpportunity, result: MarketAnalystResult) -> str:
        """
        Persist `result` following the pattern documented in
        MarketAnalystResult.as_market_analysis_kwargs()'s docstring, and
        mirror the opportunity status transition used by
        apps.market_analyst.tasks.analyze_product_opportunity for a
        completed analysis. Wrapped in a single short transaction --
        the agent has already finished its (in-memory only) work by
        this point, so no long-running operation is held inside it.
        """
        with transaction.atomic():
            analysis = MarketAnalysis.objects.create(
                opportunity=opportunity,
                **result.as_market_analysis_kwargs(),
            )
            new_status = _STATUS_BY_RECOMMENDATION.get(
                result.recommendation, ProductOpportunityStatus.MONITOR
            )
            if opportunity.status != new_status:
                opportunity.status = new_status
                opportunity.save(update_fields=["status", "updated_at"])
        return str(analysis.id)

    def _write_human_result(
        self,
        opportunity: ProductOpportunity,
        result: MarketAnalystResult,
        *,
        dry_run: bool,
        verbosity: int,
    ) -> None:
        self.stdout.write(f"Opportunity: {opportunity.name}")
        if result.status == AnalysisStatus.INSUFFICIENT_DATA:
            self.stdout.write(self.style.WARNING("Analysis: insufficient data."))
        else:
            style = self.style.SUCCESS if result.status == AnalysisStatus.SUCCESS else self.style.NOTICE
            self.stdout.write(style("Analysis completed."))
            if result.overall_score is not None:
                self.stdout.write(f"Score: {result.overall_score:.2f}")
        self.stdout.write(f"Recommendation: {result.recommendation}")
        self.stdout.write(f"Confidence: {result.confidence_score:.2f}")
        if result.missing_data:
            self.stdout.write(f"Missing data: {', '.join(result.missing_data)}")
        if dry_run:
            self.stdout.write(self.style.NOTICE("(dry-run: no MarketAnalysis was persisted)"))
        if verbosity >= 2:
            self.stdout.write(f"Summary: {result.summary}")
            self.stdout.write(f"Engine: {result.analysis_metadata.get('engine_version')}")
            self.stdout.write(f"Scoring source: {result.analysis_metadata.get('scoring_source')}")
        self.stdout.write("")