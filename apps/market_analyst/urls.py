"""
apps.market_analyst.urls
============================

Self-contained Market Analyst routes, matching the pattern used by
`apps.authentication.urls`. Include from the project root as:

    path("api/v1/market-analyst/", include("apps.market_analyst.urls"))
"""

from django.urls import path

from apps.market_analyst.views import (
    MarketAnalysisListView,
    MarketDetailView,
    MarketListView,
    ProductOpportunityDetailView,
    ProductOpportunityListCreateView,
)

app_name = "market_analyst"

urlpatterns = [
    path(
        "markets/",
        MarketListView.as_view(),
        name="market-list",
    ),
    path(
        "markets/<uuid:pk>/",
        MarketDetailView.as_view(),
        name="market-detail",
    ),
    path(
        "opportunities/",
        ProductOpportunityListCreateView.as_view(),
        name="opportunity-list-create",
    ),
    path(
        "opportunities/<uuid:pk>/",
        ProductOpportunityDetailView.as_view(),
        name="opportunity-detail",
    ),
    path(
        "opportunities/<uuid:pk>/analyses/",
        MarketAnalysisListView.as_view(),
        name="opportunity-analyses",
    ),
]