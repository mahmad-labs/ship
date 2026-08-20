"""
apps.market_analyst.views
============================

Ship's API is DRF-based (see `apps.authentication.views`), so this app
follows the same architecture rather than introducing traditional HTML
views. Views stay thin: they authenticate, validate input via
`forms.py`, call straightforward model/queryset operations, and shape
a response. No analytical logic lives here -- see `models.py` and
`tasks.py`'s module docstrings for where that belongs.

Response shaping: this app has no `serializers.py` (outside the
eight-file build scope), so plain `_serialize_*` helper functions
below turn model instances into JSON-safe dicts. This is a deliberate,
documented compatibility decision -- not an attempt to reinvent DRF
serialization -- kept intentionally small since only a handful of
read-only shapes are needed.

Every endpoint requires authentication. Markets and product
opportunities are shared Ship business intelligence (not per-user
private data -- there is no per-user ownership concept anywhere else
in the current architecture), so any authenticated user may read them;
write access is only exposed for submitting a new opportunity, and
`submitted_by` is always taken from `request.user`, never from client
input.
"""

from __future__ import annotations

from typing import Any

from django.db.models import QuerySet
from django.http import Http404
from rest_framework import permissions, status
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.market_analyst.forms import (
    ProductOpportunityCreateForm,
    ProductOpportunityFilterForm,
)
from apps.market_analyst.models import (
    Market,
    MarketAnalysis,
    ProductOpportunity,
)

DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _paginate(
    request: Request,
    queryset: QuerySet,
) -> tuple[QuerySet, dict[str, Any]]:
    """
    Minimal, explicit offset pagination. No DRF pagination class is
    configured project-wide (see `REST_FRAMEWORK` in settings), so this
    keeps behavior self-contained rather than guessing at a global
    default that doesn't exist. Page size is capped to avoid an
    unbounded response from a client-supplied value.
    """
    try:
        page = max(int(request.query_params.get("page", 1)), 1)
    except (TypeError, ValueError):
        page = 1

    try:
        page_size = int(
            request.query_params.get(
                "page_size",
                DEFAULT_PAGE_SIZE,
            )
        )
    except (TypeError, ValueError):
        page_size = DEFAULT_PAGE_SIZE

    page_size = min(max(page_size, 1), MAX_PAGE_SIZE)

    total = queryset.count()
    start = (page - 1) * page_size
    page_items = queryset[start : start + page_size]

    meta = {
        "page": page,
        "page_size": page_size,
        "total": total,
        "has_next": start + page_size < total,
    }

    return page_items, meta


def _form_errors(form) -> dict[str, list[str]]:
    return {
        field: errors
        for field, errors in form.errors.items()
    }


def _serialize_market(market: Market) -> dict[str, Any]:
    return {
        "id": str(market.id),
        "name": market.name,
        "country_code": market.country_code,
        "region": market.region,
        "currency": market.currency,
        "is_active": market.is_active,
        "created_at": market.created_at.isoformat(),
        "updated_at": market.updated_at.isoformat(),
    }


def _serialize_opportunity(
    opportunity: ProductOpportunity,
    *,
    detail: bool = False,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "id": str(opportunity.id),
        "market_id": str(opportunity.market_id),
        "name": opportunity.name,
        "category": opportunity.category,
        "status": opportunity.status,
        "demand_score": opportunity.demand_score,
        "trend_score": opportunity.trend_score,
        "margin_score": opportunity.margin_score,
        "competition_score": opportunity.competition_score,
        "saturation_score": opportunity.saturation_score,
        "overall_score": opportunity.overall_score,
        "is_scored": opportunity.is_scored,
        "created_at": opportunity.created_at.isoformat(),
        "updated_at": opportunity.updated_at.isoformat(),
    }

    if detail:
        data["description"] = opportunity.description
        data["submitted_by_id"] = (
            str(opportunity.submitted_by_id)
            if opportunity.submitted_by_id
            else None
        )

    return data


def _serialize_analysis(analysis: MarketAnalysis) -> dict[str, Any]:
    return {
        "id": str(analysis.id),
        "opportunity_id": str(analysis.opportunity_id),
        "summary": analysis.summary,
        "recommendation": analysis.recommendation,
        "confidence_score": analysis.confidence_score,
        "structured_data": analysis.structured_data,
        "analysis_version": analysis.analysis_version,
        "source": analysis.source,
        "created_at": analysis.created_at.isoformat(),
    }


# ---------------------------------------------------------------------------
# Markets
# ---------------------------------------------------------------------------


class MarketListView(APIView):
    """GET /markets/ — authenticated. Lists active markets, paginated."""

    permission_classes = [permissions.IsAuthenticated]

    def get(self, request: Request) -> Response:
        queryset = (
            Market.objects
            .filter(is_active=True)
            .order_by("name")
        )

        page_items, meta = _paginate(request, queryset)

        return Response(
            {
                "results": [
                    _serialize_market(m)
                    for m in page_items
                ],
                **meta,
            },
            status=status.HTTP_200_OK,
        )


class MarketDetailView(APIView):
    """GET /markets/<uuid:pk>/ — authenticated. Retrieve a single market."""

    permission_classes = [permissions.IsAuthenticated]

    def get(
        self,
        request: Request,
        pk: str,
    ) -> Response:
        try:
            market = Market.objects.get(pk=pk)
        except (
            Market.DoesNotExist,
            ValueError,
            Http404,
        ):
            return Response(
                {"detail": "Market not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        return Response(
            _serialize_market(market),
            status=status.HTTP_200_OK,
        )


# ---------------------------------------------------------------------------
# Product opportunities
# ---------------------------------------------------------------------------


class ProductOpportunityListCreateView(APIView):
    """
    GET  /opportunities/ — authenticated. Lists opportunities, filtered
         and paginated via `ProductOpportunityFilterForm`.

    POST /opportunities/ — authenticated. Submits a new opportunity for
         analysis via `ProductOpportunityCreateForm`. `submitted_by` is
         always the requesting user; scores are never accepted here.
    """

    permission_classes = [permissions.IsAuthenticated]

    def get(self, request: Request) -> Response:
        filter_form = ProductOpportunityFilterForm(
            data=request.query_params
        )

        if not filter_form.is_valid():
            return Response(
                _form_errors(filter_form),
                status=status.HTTP_400_BAD_REQUEST,
            )

        queryset = (
            ProductOpportunity.objects
            .select_related("market")
            .order_by("-created_at")
        )

        cleaned = filter_form.cleaned_data

        if cleaned.get("market") is not None:
            queryset = queryset.filter(
                market=cleaned["market"]
            )

        if cleaned.get("status"):
            queryset = queryset.filter(
                status=cleaned["status"]
            )

        if cleaned.get("category"):
            queryset = queryset.filter(
                category__iexact=cleaned["category"]
            )

        if cleaned.get("min_overall_score") is not None:
            queryset = queryset.filter(
                overall_score__gte=cleaned["min_overall_score"]
            )

        page_items, meta = _paginate(
            request,
            queryset,
        )

        return Response(
            {
                "results": [
                    _serialize_opportunity(o)
                    for o in page_items
                ],
                **meta,
            },
            status=status.HTTP_200_OK,
        )

    def post(self, request: Request) -> Response:
        form = ProductOpportunityCreateForm(
            data=request.data
        )

        if not form.is_valid():
            return Response(
                _form_errors(form),
                status=status.HTTP_400_BAD_REQUEST,
            )

        opportunity = form.save(commit=False)
        opportunity.submitted_by = request.user
        opportunity.save()

        return Response(
            _serialize_opportunity(
                opportunity,
                detail=True,
            ),
            status=status.HTTP_201_CREATED,
        )


class ProductOpportunityDetailView(APIView):
    """GET /opportunities/<uuid:pk>/ — authenticated. Retrieve a single opportunity."""

    permission_classes = [permissions.IsAuthenticated]

    def get(
        self,
        request: Request,
        pk: str,
    ) -> Response:
        try:
            opportunity = (
                ProductOpportunity.objects
                .select_related("market")
                .get(pk=pk)
            )
        except (
            ProductOpportunity.DoesNotExist,
            ValueError,
            Http404,
        ):
            return Response(
                {"detail": "Product opportunity not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        return Response(
            _serialize_opportunity(
                opportunity,
                detail=True,
            ),
            status=status.HTTP_200_OK,
        )


# ---------------------------------------------------------------------------
# Market analyses
# ---------------------------------------------------------------------------


class MarketAnalysisListView(APIView):
    """
    GET /opportunities/<uuid:pk>/analyses/ — authenticated.
    Lists analysis results for an opportunity, newest first.
    """

    permission_classes = [permissions.IsAuthenticated]

    def get(
        self,
        request: Request,
        pk: str,
    ) -> Response:
        try:
            opportunity = ProductOpportunity.objects.get(pk=pk)
        except (
            ProductOpportunity.DoesNotExist,
            ValueError,
            Http404,
        ):
            return Response(
                {"detail": "Product opportunity not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        queryset = (
            opportunity.analyses
            .all()
            .order_by("-created_at")
        )

        page_items, meta = _paginate(
            request,
            queryset,
        )

        return Response(
            {
                "results": [
                    _serialize_analysis(a)
                    for a in page_items
                ],
                **meta,
            },
            status=status.HTTP_200_OK,
        )