"""
apps.market_analyst.services
===============================

The analytical/domain layer of the Market Analyst pipeline: deterministic,
stateless, testable analysis over already-collected evidence. See each
module's docstring for its specific responsibility:

    analyzer              High-level composition of the services below.
    trend_analyzer         Demand/trend analysis.
    competitor_analyzer     Competitive intelligence analysis.
    customer_analyzer       Customer sentiment/pain-point analysis.
    pricing_analyzer        Pricing/margin analysis.
    opportunity_scorer      Deterministic overall opportunity scoring.

Deliberately empty otherwise: no package-level re-exports and no
service instantiation at import time, so importing this package (or
any submodule) never has a side effect. Callers import the specific
module/class they need, e.g.:

    from apps.market_analyst.services.analyzer import Analyzer
"""

from __future__ import annotations