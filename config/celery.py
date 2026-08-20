"""
config.celery
==============

Ship's project-level Celery application.

Django settings are the single source of Celery configuration: every
``CELERY_*`` setting defined on the active Django settings module
(``config/settings/base.py``/``development.py``/``production.py``/
``testing.py``) is loaded here via ``config_from_object``'s
``CELERY_`` namespace -- this file intentionally carries no broker
URL, credentials, or other environment-specific values of its own.

Task autodiscovery walks every app in ``INSTALLED_APPS`` (e.g.
``apps.market_analyst``) for a ``tasks.py`` module, so a task defined
there is picked up automatically without being registered here.
"""

import os

from celery import Celery

# Matches the default already used by manage.py/wsgi.py/asgi.py: the
# environment is expected to override this (e.g. to
# `config.settings.production`) rather than this file guessing at it.
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.development")

app = Celery("config")

# Read CELERY_* settings from the active Django settings module.
app.config_from_object("django.conf:settings", namespace="CELERY")

# Discover a `tasks.py` in each app listed in INSTALLED_APPS.
app.autodiscover_tasks()