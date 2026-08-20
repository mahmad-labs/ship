"""
apps.market_analyst.tests
============================

Tests against the real, unmodified `models.py` / `forms.py` /
`views.py` / `urls.py` / `tasks.py`. No factory library is used
elsewhere in this project, so plain helper methods are used here too.
No external API/network calls are made anywhere in this app, so no
mocking of external integrations is required.
"""

from __future__ import annotations

import uuid
from decimal import Decimal

from django.contrib.auth import get_user_model
from django.db import IntegrityError, transaction
from django.core.exceptions import ValidationError
from django.test import TestCase
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase

from apps.market_analyst.forms import ProductOpportunityCreateForm, ProductOpportunityFilterForm
from apps.market_analyst.models import (
    Market,
    MarketAnalysis,
    MarketAnalysisRecommendation,
    ProductOpportunity,
    ProductOpportunityStatus,
)
from apps.market_analyst.tasks import analyze_product_opportunity

User = get_user_model()


class MarketAnalystTestHelpersMixin:
    """Small, local object-creation helpers, mirroring `apps.authentication.tests` style."""

    def make_user(self, email="user@example.com", password="Str0ng-Pass!2024", **extra):
        return User.objects.create_user(email=email, password=password, **extra)

    def make_market(self, name="United States", country_code="US", region="", currency="USD", **extra):
        return Market.objects.create(
            name=name, country_code=country_code, region=region, currency=currency, **extra
        )

    def make_opportunity(self, market=None, **extra):
        if market is None:
            market, _ = Market.objects.get_or_create(
                name="United States",
                country_code="US",
                region="",
                defaults={"currency": "USD"},
            )
        defaults = {
            "market": market,
            "name": "Test Product",
            "category": "Jewelry",
        }
        defaults.update(extra)
        return ProductOpportunity.objects.create(**defaults)

    def make_analysis(self, opportunity=None, **extra):
        opportunity = opportunity or self.make_opportunity()
        defaults = {
            "opportunity": opportunity,
            "summary": "Looks promising.",
            "recommendation": MarketAnalysisRecommendation.INVESTIGATE,
            "confidence_score": Decimal("70.00"),
        }
        defaults.update(extra)
        return MarketAnalysis.objects.create(**defaults)


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class MarketModelTests(MarketAnalystTestHelpersMixin, TestCase):
    def test_market_creation(self):
        market = self.make_market()
        self.assertIsInstance(market.id, uuid.UUID)
        self.assertTrue(market.is_active)

    def test_market_str(self):
        market = self.make_market(name="United States", country_code="US")
        self.assertEqual(str(market), "United States (US)")

    def test_market_str_includes_region(self):
        market = self.make_market(name="United States", country_code="US", region="West Coast")
        self.assertEqual(str(market), "United States (US) - West Coast")

    def test_market_ordering_is_by_name(self):
        self.make_market(name="Zealand Market", country_code="NZ")
        self.make_market(name="Atlas Market", country_code="US")
        names = list(Market.objects.values_list("name", flat=True))
        self.assertEqual(names, sorted(names))

    def test_duplicate_market_same_name_country_region_rejected(self):
        self.make_market(name="United States", country_code="US", region="")
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                self.make_market(name="United States", country_code="US", region="")

    def test_duplicate_market_case_insensitive_name_rejected(self):
        self.make_market(name="United States", country_code="US", region="")
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                self.make_market(name="UNITED STATES", country_code="US", region="")

    def test_same_name_different_region_allowed(self):
        self.make_market(name="United States", country_code="US", region="East Coast")
        # Should not raise.
        self.make_market(name="United States", country_code="US", region="West Coast")
        self.assertEqual(Market.objects.count(), 2)


class ProductOpportunityModelTests(MarketAnalystTestHelpersMixin, TestCase):
    def test_opportunity_creation_defaults(self):
        opportunity = self.make_opportunity()
        self.assertEqual(opportunity.status, ProductOpportunityStatus.DISCOVERED)
        self.assertIsNone(opportunity.overall_score)
        self.assertFalse(opportunity.is_scored)

    def test_opportunity_str(self):
        market = self.make_market(country_code="US")
        opportunity = self.make_opportunity(market=market, name="Bracelet")
        self.assertIn("Bracelet", str(opportunity))
        self.assertIn("US", str(opportunity))
        self.assertIn(ProductOpportunityStatus.DISCOVERED, str(opportunity))

    def test_is_scored_true_once_overall_score_set(self):
        opportunity = self.make_opportunity(overall_score=Decimal("82.50"))
        self.assertTrue(opportunity.is_scored)

    def test_valid_status_values_accepted(self):
        for value in ProductOpportunityStatus.values:
            opportunity = self.make_opportunity(status=value)
            opportunity.full_clean()

    def test_score_within_range_passes_full_clean(self):
        opportunity = self.make_opportunity(demand_score=Decimal("50.00"))
        opportunity.full_clean()

    def test_score_above_max_fails_full_clean(self):
        market = self.make_market()
        opportunity = ProductOpportunity(
            market=market, name="X", category="Y", demand_score=Decimal("150.00")
        )
        with self.assertRaises(ValidationError):
            opportunity.full_clean()

    def test_score_below_min_fails_full_clean(self):
        market = self.make_market()
        opportunity = ProductOpportunity(
            market=market, name="X", category="Y", demand_score=Decimal("-1.00")
        )
        with self.assertRaises(ValidationError):
            opportunity.full_clean()

    def test_score_above_max_rejected_at_database_level(self):
        """Constraint must hold even bypassing model validation via .update()."""
        opportunity = self.make_opportunity()
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                ProductOpportunity.objects.filter(pk=opportunity.pk).update(demand_score=Decimal("101.00"))

    def test_deleting_market_with_opportunities_is_protected(self):
        market = self.make_market()
        self.make_opportunity(market=market)
        from django.db.models import ProtectedError

        with self.assertRaises(ProtectedError):
            market.delete()

    def test_ordering_is_newest_first(self):
        first = self.make_opportunity(name="First")
        second = self.make_opportunity(name="Second")
        ids = list(ProductOpportunity.objects.values_list("id", flat=True))
        self.assertEqual(ids, [second.id, first.id])


class MarketAnalysisModelTests(MarketAnalystTestHelpersMixin, TestCase):
    def test_analysis_creation_and_relationship(self):
        opportunity = self.make_opportunity()
        analysis = self.make_analysis(opportunity=opportunity)
        self.assertEqual(analysis.opportunity, opportunity)
        self.assertIn(analysis, opportunity.analyses.all())

    def test_analysis_str(self):
        analysis = self.make_analysis(recommendation=MarketAnalysisRecommendation.MONITOR)
        self.assertIn("monitor", str(analysis))

    def test_confidence_score_out_of_range_fails_full_clean(self):
        opportunity = self.make_opportunity()
        analysis = MarketAnalysis(
            opportunity=opportunity,
            summary="Looks promising.",
            recommendation=MarketAnalysisRecommendation.INVESTIGATE,
            confidence_score=Decimal("200.00"),
        )
        with self.assertRaises(ValidationError):
            analysis.full_clean()

    def test_confidence_score_out_of_range_rejected_at_database_level(self):
        analysis = self.make_analysis()
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                MarketAnalysis.objects.filter(pk=analysis.pk).update(confidence_score=Decimal("-5.00"))

    def test_deleting_opportunity_cascades_to_analyses(self):
        opportunity = self.make_opportunity()
        analysis = self.make_analysis(opportunity=opportunity)
        opportunity.delete()
        self.assertFalse(MarketAnalysis.objects.filter(pk=analysis.pk).exists())

    def test_structured_data_defaults_to_empty_dict(self):
        analysis = self.make_analysis()
        self.assertEqual(analysis.structured_data, {})

    def test_invalid_recommendation_choice_fails_full_clean(self):
        analysis = self.make_analysis(recommendation="not_a_real_choice")
        with self.assertRaises(ValidationError):
            analysis.full_clean()


# ---------------------------------------------------------------------------
# Form tests
# ---------------------------------------------------------------------------


class ProductOpportunityCreateFormTests(MarketAnalystTestHelpersMixin, TestCase):
    def test_valid_input_creates_opportunity(self):
        market = self.make_market()
        form = ProductOpportunityCreateForm(
            data={"market": market.pk, "name": "Necklace", "category": "Jewelry", "description": "Shiny."}
        )
        self.assertTrue(form.is_valid(), form.errors)
        opportunity = form.save()
        self.assertEqual(opportunity.name, "Necklace")
        self.assertEqual(opportunity.status, ProductOpportunityStatus.DISCOVERED)

    def test_missing_required_fields_invalid(self):
        form = ProductOpportunityCreateForm(data={})
        self.assertFalse(form.is_valid())
        self.assertIn("market", form.errors)
        self.assertIn("name", form.errors)
        self.assertIn("category", form.errors)

    def test_blank_name_invalid(self):
        market = self.make_market()
        form = ProductOpportunityCreateForm(data={"market": market.pk, "name": "   ", "category": "Jewelry"})
        self.assertFalse(form.is_valid())
        self.assertIn("name", form.errors)

    def test_inactive_market_rejected(self):
        market = self.make_market(is_active=False)
        form = ProductOpportunityCreateForm(data={"market": market.pk, "name": "Ring", "category": "Jewelry"})
        self.assertFalse(form.is_valid())
        self.assertIn("market", form.errors)

    def test_form_does_not_expose_score_fields(self):
        self.assertNotIn("overall_score", ProductOpportunityCreateForm.base_fields)
        self.assertNotIn("status", ProductOpportunityCreateForm.base_fields)

    def test_nonexistent_market_invalid(self):
        form = ProductOpportunityCreateForm(
            data={"market": str(uuid.uuid4()), "name": "Ring", "category": "Jewelry"}
        )
        self.assertFalse(form.is_valid())
        self.assertIn("market", form.errors)


class ProductOpportunityFilterFormTests(MarketAnalystTestHelpersMixin, TestCase):
    def test_empty_data_is_valid(self):
        form = ProductOpportunityFilterForm(data={})
        self.assertTrue(form.is_valid())

    def test_valid_status_accepted(self):
        form = ProductOpportunityFilterForm(data={"status": ProductOpportunityStatus.MONITOR})
        self.assertTrue(form.is_valid())

    def test_invalid_status_rejected(self):
        form = ProductOpportunityFilterForm(data={"status": "not_a_status"})
        self.assertFalse(form.is_valid())
        self.assertIn("status", form.errors)

    def test_min_overall_score_out_of_range_rejected(self):
        form = ProductOpportunityFilterForm(data={"min_overall_score": "150"})
        self.assertFalse(form.is_valid())
        self.assertIn("min_overall_score", form.errors)

    def test_non_numeric_min_overall_score_rejected(self):
        form = ProductOpportunityFilterForm(data={"min_overall_score": "not-a-number"})
        self.assertFalse(form.is_valid())


# ---------------------------------------------------------------------------
# URL tests
# ---------------------------------------------------------------------------


class URLResolutionTests(MarketAnalystTestHelpersMixin, TestCase):
    def test_market_list_url_resolves(self):
        self.assertEqual(reverse("market_analyst:market-list"), "/api/v1/market-analyst/markets/")

    def test_market_detail_url_resolves(self):
        market_id = uuid.uuid4()
        url = reverse("market_analyst:market-detail", kwargs={"pk": market_id})
        self.assertEqual(url, f"/api/v1/market-analyst/markets/{market_id}/")

    def test_opportunity_list_create_url_resolves(self):
        self.assertEqual(
            reverse("market_analyst:opportunity-list-create"), "/api/v1/market-analyst/opportunities/"
        )

    def test_opportunity_detail_url_resolves(self):
        opportunity_id = uuid.uuid4()
        url = reverse("market_analyst:opportunity-detail", kwargs={"pk": opportunity_id})
        self.assertEqual(url, f"/api/v1/market-analyst/opportunities/{opportunity_id}/")

    def test_opportunity_analyses_url_resolves(self):
        opportunity_id = uuid.uuid4()
        url = reverse("market_analyst:opportunity-analyses", kwargs={"pk": opportunity_id})
        self.assertEqual(url, f"/api/v1/market-analyst/opportunities/{opportunity_id}/analyses/")


# ---------------------------------------------------------------------------
# View tests
# ---------------------------------------------------------------------------


def url(name, **kwargs):
    return reverse(f"market_analyst:{name}", kwargs=kwargs or None)


class MarketAnalystAPITestCase(MarketAnalystTestHelpersMixin, APITestCase):
    def authenticate_as(self, user):
        self.client.force_authenticate(user=user)


class MarketViewTests(MarketAnalystAPITestCase):
    def test_list_requires_authentication(self):
        resp = self.client.get(url("market-list"))
        self.assertEqual(resp.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_list_returns_active_markets_for_authenticated_user(self):
        user = self.make_user()
        self.authenticate_as(user)
        self.make_market(name="United States", country_code="US")
        self.make_market(name="Retired Market", country_code="CA", is_active=False)

        resp = self.client.get(url("market-list"))
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        body = resp.json()
        names = [item["name"] for item in body["results"]]
        self.assertIn("United States", names)
        self.assertNotIn("Retired Market", names)

    def test_detail_requires_authentication(self):
        market = self.make_market()
        resp = self.client.get(url("market-detail", pk=market.pk))
        self.assertEqual(resp.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_detail_returns_market_for_authenticated_user(self):
        user = self.make_user()
        self.authenticate_as(user)
        market = self.make_market(name="United States")

        resp = self.client.get(url("market-detail", pk=market.pk))
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertEqual(resp.json()["name"], "United States")

    def test_detail_missing_market_returns_404(self):
        user = self.make_user()
        self.authenticate_as(user)
        resp = self.client.get(url("market-detail", pk=uuid.uuid4()))
        self.assertEqual(resp.status_code, status.HTTP_404_NOT_FOUND)


class ProductOpportunityViewTests(MarketAnalystAPITestCase):
    def test_list_requires_authentication(self):
        resp = self.client.get(url("opportunity-list-create"))
        self.assertEqual(resp.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_create_requires_authentication(self):
        market = self.make_market()
        resp = self.client.post(
            url("opportunity-list-create"),
            {"market": str(market.pk), "name": "Ring", "category": "Jewelry"},
        )
        self.assertEqual(resp.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_authenticated_create_succeeds_and_sets_submitted_by(self):
        user = self.make_user()
        self.authenticate_as(user)
        market = self.make_market()

        resp = self.client.post(
            url("opportunity-list-create"),
            {"market": str(market.pk), "name": "Ring", "category": "Jewelry", "description": "Nice."},
        )
        self.assertEqual(resp.status_code, status.HTTP_201_CREATED)
        opportunity = ProductOpportunity.objects.get(name="Ring")
        self.assertEqual(opportunity.submitted_by, user)

    def test_create_ignores_client_supplied_submitted_by_and_scores(self):
        user = self.make_user()
        other_user = self.make_user(email="other@example.com")
        self.authenticate_as(user)
        market = self.make_market()

        resp = self.client.post(
            url("opportunity-list-create"),
            {
                "market": str(market.pk),
                "name": "Ring",
                "category": "Jewelry",
                "submitted_by": str(other_user.pk),
                "overall_score": "99.00",
            },
        )
        self.assertEqual(resp.status_code, status.HTTP_201_CREATED)
        opportunity = ProductOpportunity.objects.get(name="Ring")
        self.assertEqual(opportunity.submitted_by, user)
        self.assertIsNone(opportunity.overall_score)

    def test_create_invalid_payload_returns_400(self):
        user = self.make_user()
        self.authenticate_as(user)
        resp = self.client.post(url("opportunity-list-create"), {})
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_list_filters_by_status(self):
        user = self.make_user()
        self.authenticate_as(user)
        self.make_opportunity(name="Discovered One", status=ProductOpportunityStatus.DISCOVERED)
        self.make_opportunity(name="Monitored One", status=ProductOpportunityStatus.MONITOR)

        resp = self.client.get(url("opportunity-list-create"), {"status": ProductOpportunityStatus.MONITOR})
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        names = [item["name"] for item in resp.json()["results"]]
        self.assertEqual(names, ["Monitored One"])

    def test_list_invalid_filter_returns_400(self):
        user = self.make_user()
        self.authenticate_as(user)
        resp = self.client.get(url("opportunity-list-create"), {"status": "bogus"})
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_detail_requires_authentication(self):
        opportunity = self.make_opportunity()
        resp = self.client.get(url("opportunity-detail", pk=opportunity.pk))
        self.assertEqual(resp.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_detail_missing_returns_404(self):
        user = self.make_user()
        self.authenticate_as(user)
        resp = self.client.get(url("opportunity-detail", pk=uuid.uuid4()))
        self.assertEqual(resp.status_code, status.HTTP_404_NOT_FOUND)

    def test_detail_returns_opportunity_for_any_authenticated_user(self):
        """
        Markets/opportunities are shared Ship business intelligence,
        not per-user private data -- no ownership model exists for
        them anywhere in the current architecture (see views.py module
        docstring). Any authenticated user may read any opportunity,
        including ones submitted by a different user.
        """
        owner = self.make_user(email="owner@example.com")
        other = self.make_user(email="other@example.com")
        opportunity = self.make_opportunity(submitted_by=owner)

        self.authenticate_as(other)
        resp = self.client.get(url("opportunity-detail", pk=opportunity.pk))
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertEqual(resp.json()["id"], str(opportunity.id))


class MarketAnalysisViewTests(MarketAnalystAPITestCase):
    def test_list_requires_authentication(self):
        opportunity = self.make_opportunity()
        resp = self.client.get(url("opportunity-analyses", pk=opportunity.pk))
        self.assertEqual(resp.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_missing_opportunity_returns_404(self):
        user = self.make_user()
        self.authenticate_as(user)
        resp = self.client.get(url("opportunity-analyses", pk=uuid.uuid4()))
        self.assertEqual(resp.status_code, status.HTTP_404_NOT_FOUND)

    def test_list_returns_analyses_newest_first(self):
        user = self.make_user()
        self.authenticate_as(user)
        opportunity = self.make_opportunity()
        first = self.make_analysis(opportunity=opportunity, summary="First analysis")
        second = self.make_analysis(opportunity=opportunity, summary="Second analysis")

        resp = self.client.get(url("opportunity-analyses", pk=opportunity.pk))
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        ids = [item["id"] for item in resp.json()["results"]]
        self.assertEqual(ids, [str(second.id), str(first.id)])


# ---------------------------------------------------------------------------
# Task tests
# ---------------------------------------------------------------------------


class AnalyzeProductOpportunityTaskTests(MarketAnalystTestHelpersMixin, TestCase):
    def test_missing_opportunity_returns_not_found_without_raising(self):
        result = analyze_product_opportunity(str(uuid.uuid4()))
        self.assertEqual(result["status"], "not_found")

    def test_invalid_uuid_returns_not_found_without_raising(self):
        result = analyze_product_opportunity("not-a-uuid")
        self.assertEqual(result["status"], "not_found")

    def test_successful_execution_creates_analysis_and_updates_status(self):
        opportunity = self.make_opportunity()
        result = analyze_product_opportunity(str(opportunity.pk))

        self.assertEqual(result["status"], "completed")
        opportunity.refresh_from_db()
        self.assertEqual(opportunity.analyses.count(), 1)
        self.assertIn(
            opportunity.status,
            (ProductOpportunityStatus.INVESTIGATE, ProductOpportunityStatus.MONITOR),
        )

    def test_no_component_scores_yields_monitor_recommendation(self):
        opportunity = self.make_opportunity()
        analyze_product_opportunity(str(opportunity.pk))
        analysis = opportunity.analyses.first()
        self.assertEqual(analysis.recommendation, MarketAnalysisRecommendation.MONITOR)

    def test_duplicate_run_within_window_is_skipped(self):
        opportunity = self.make_opportunity()
        first_result = analyze_product_opportunity(str(opportunity.pk))
        self.assertEqual(first_result["status"], "completed")

        second_result = analyze_product_opportunity(str(opportunity.pk))
        self.assertEqual(second_result["status"], "skipped_duplicate")
        self.assertEqual(opportunity.analyses.count(), 1)

    def test_result_is_serializable_primitive_types(self):
        opportunity = self.make_opportunity()
        result = analyze_product_opportunity(str(opportunity.pk))
        for value in result.values():
            self.assertIsInstance(value, str)