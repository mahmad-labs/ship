"""
apps.market_analyst.apps
=========================

AppConfig for Ship's Market Analyst app.

This app owns market and product-opportunity intelligence: the
persistent record of which markets/categories Ship has looked at, what
opportunities were discovered inside them, and what analytical
conclusions (scores, recommendations, confidence) were reached.

It depends on ``authentication`` (for ``AUTH_USER_MODEL``, to record
who submitted an opportunity) but owns no identity/credential concerns
itself, keeping the dependency graph acyclic:
``core <- authentication <- market_analyst``.

No database queries, API calls, or other side effects are performed at
import/startup time -- ``ready()`` is intentionally not overridden.
"""

from django.apps import AppConfig


class MarketAnalystConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.market_analyst"
    label = "market_analyst"
    verbose_name = "Market Analyst"