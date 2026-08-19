"""
apps.authentication.apps
=========================

AppConfig for Ship's authentication app.

This app owns the custom User model and nothing else. It is a
dependency of most other Ship apps (they will reference AUTH_USER_MODEL)
but must not import or depend on any of them, to keep the dependency
graph acyclic: core <- authentication <- everything else.
"""

from django.apps import AppConfig


class AuthenticationConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.authentication"
    label = "authentication"
    verbose_name = "Authentication"