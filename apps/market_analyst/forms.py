"""
apps.market_analyst.forms
============================

Server-side input validation for the Market Analyst app.

Ship's API is DRF-based (see `apps.authentication.views`/`serializers`
for the established pattern), but this app is only responsible for the
eight files listed in its build scope, which does not include a
`serializers.py`. Django's own form/model-form machinery is used
instead as the validation layer, and is deliberately built to accept
plain `dict`-like data (as DRF's `request.data` is), so `views.py` can
do `SomeForm(data=request.data)` exactly as it would with a DRF
serializer. This keeps validation fully server-side and out of the
views, per the project's separation-of-concerns requirement, without
inventing a second, undocumented input-validation convention.

Only forms with a real caller in `views.py` are defined here:
    ProductOpportunityCreateForm  Validates a new opportunity submission.
    ProductOpportunityFilterForm  Validates list-view query parameters.

Neither form ever exposes a way to set opportunity scores,
`overall_score`, or any `MarketAnalysis` field -- those are derived
intelligence, written only by the analysis pipeline (see models.py's
module docstring and `tasks.py`).
"""

from __future__ import annotations

from decimal import Decimal

from django import forms

from apps.market_analyst.models import (
    Market,
    ProductOpportunity,
    ProductOpportunityStatus,
)


class ProductOpportunityCreateForm(forms.ModelForm):
    """
    Validates a user's submission of a new product opportunity for
    analysis.

    Deliberately a narrow field set: `market`, `name`, `category`,
    `description` only. Status defaults to DISCOVERED in the model;
    scores are left unset (None) until the analysis pipeline runs.
    """

    class Meta:
        model = ProductOpportunity
        fields = ["market", "name", "category", "description"]
        widgets = {
            "description": forms.Textarea(attrs={"rows": 4}),
        }

    def clean_market(self) -> Market:
        market = self.cleaned_data["market"]

        if not market.is_active:
            raise forms.ValidationError(
                "This market is not currently active for new opportunities."
            )

        return market

    def clean_name(self) -> str:
        name = self.cleaned_data["name"].strip()

        if not name:
            raise forms.ValidationError("Product name cannot be blank.")

        return name

    def clean_category(self) -> str:
        category = self.cleaned_data["category"].strip()

        if not category:
            raise forms.ValidationError("Category cannot be blank.")

        return category


class ProductOpportunityFilterForm(forms.Form):
    """
    Validates optional query-string filters accepted by the product
    opportunity list view. Every field is optional; an empty form is a
    valid, unfiltered request.
    """

    market = forms.ModelChoiceField(
        queryset=Market.objects.all(),
        required=False,
    )

    status = forms.ChoiceField(
        choices=ProductOpportunityStatus.choices,
        required=False,
    )

    category = forms.CharField(
        max_length=150,
        required=False,
    )

    min_overall_score = forms.DecimalField(
        max_digits=5,
        decimal_places=2,
        min_value=Decimal("0"),
        max_value=Decimal("100"),
        required=False,
    )

    def clean_category(self) -> str:
        return self.cleaned_data["category"].strip()