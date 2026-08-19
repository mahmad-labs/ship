"""
apps.core.models
=================

Shared, generic, platform-level infrastructure for Ship.

This module intentionally contains NO business-domain models (Product,
Supplier, Store, Order, Customer, InventoryItem, ShopifyStore, AI Agent,
Campaign, PricingRule, SourcingOpportunity, Subscription, Payment, etc).
Those belong to their own apps (products, suppliers, stores, orders,
inventory, integrations, pricing, sourcing, ai, ...).

Core owns only cross-cutting concerns that every other app can safely
depend on without creating circular imports or coupling Core to any
domain's business logic:

    1. UUIDModel / TimeStampedModel / BaseModel
       Identity and lifecycle-timestamp foundations.

    2. SoftDeleteModel (+ managers/queryset)
       Reversible deletion for domain models that opt in.

    3. AuditEvent
       A generic, append-only "who/what changed, when, in what context"
       record that works across every domain without Core importing any
       domain model.

    4. IdempotencyRecord / ProcessedEvent
       Generic duplicate-suppression primitives for retried requests and
       retried/duplicated inbound webhooks. These are infrastructural
       because *every* future integration (Shopify, suppliers, payments)
       needs them, and building it once here avoids each integration
       reinventing an idempotency table.

Explicitly NOT implemented here (and why):
    - SystemConfiguration / FeatureFlag: nothing in the current
      architecture consumes platform-wide runtime config or flags yet.
      Environment/deployment configuration covers today's needs. Adding
      an untyped key-value table speculatively is exactly the kind of
      premature abstraction this module avoids. Add it in a future
      migration when a real consumer exists.

Design invariants:
    - Core has zero foreign keys into other Ship apps. Cross-app
      references (e.g. an audit event's "resource") are represented as
      plain strings/UUIDs, never Django ForeignKey/GenericForeignKey,
      so Core never needs to import authentication/stores/products/etc.
      This is a deliberate trade-off: we give up DB-level referential
      integrity and `select_related` convenience on these references in
      exchange for Core having no import-time or migration-time
      dependency on any domain app. Domain apps are free to add their
      own FK-backed audit/event tables if they need stronger guarantees
      for a specific relationship.
    - No signals. Side effects triggered by saving/deleting Core models
      are implemented explicitly by callers (services), not hidden in
      models.py.
    - No secrets. JSONField payloads on these models must never contain
      passwords, API keys, access tokens, or other credentials. This is
      a hard rule enforced by code review / service-layer discipline,
      not by the database (the DB cannot know what a value "means").
"""

from __future__ import annotations

import uuid
from typing import Any

from django.db import models
from django.utils import timezone


# ---------------------------------------------------------------------------
# Identity & timestamp foundations
# ---------------------------------------------------------------------------


class UUIDModel(models.Model):
    """
    Abstract base providing a UUID4 primary key.

    Why UUID over an auto-incrementing integer:
        - Public-facing identifiers must not be sequential/enumerable
          (avoids leaking record counts / enabling ID-scanning attacks
          across a future public API).
        - A single ID format works uniformly across every Ship app
          without coordinating integer PK ranges.

    Why application-generated (``default=uuid.uuid4``) instead of a
    database-generated UUID (e.g. Postgres ``gen_random_uuid()``):
        - The ID exists in Python before the row is inserted, which is
          required for assigning it to related objects prior to
          `save()`, for deterministic tests, and for idempotent
          client-side retry logic.
        - Avoids a hard dependency on the Postgres ``pgcrypto``/
          ``uuid-ossp`` extension and keeps SQLite usable for local/
          limited test runs (SQLite has no native UUID generation).
        - Cost: one extra client-side RNG call per insert, which is
          negligible.

    ``uuid.uuid4`` is used (not uuid1/uuid7) because it is fully random
    and carries no embedded timestamp or MAC-address-derived data,
    which matters for identifiers that may become externally visible.
    """

    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
        help_text="Globally unique, non-sequential identifier.",
    )

    class Meta:
        abstract = True


class TimeStampedModel(models.Model):
    """
    Abstract base providing timezone-aware ``created_at``/``updated_at``.

    Both fields are managed automatically by Django (``auto_now_add`` /
    ``auto_now``) and are timezone-aware, matching ``USE_TZ = True``.

    Limitation (documented, not silently papered over): ``auto_now``
    only fires on ``Model.save()``. It does **not** fire on
    ``QuerySet.update()`` or ``bulk_update()``. Any code performing bulk
    writes against a model built on this base must explicitly set
    ``updated_at=timezone.now()`` in that call, e.g.:

        MyModel.objects.filter(...).update(
            status="done", updated_at=timezone.now()
        )

    No index is declared here. Indexing ``created_at``/``updated_at`` is
    a decision each concrete model should make based on its own query
    patterns (e.g. ``AuditEvent`` indexes ``created_at``; most domain
    models will not need to).
    """

    created_at = models.DateTimeField(
        auto_now_add=True,
        help_text="When this record was first created (UTC).",
    )
    updated_at = models.DateTimeField(
        auto_now=True,
        help_text=(
            "When this record was last saved (UTC). Not updated by "
            "QuerySet.update()/bulk_update() — see class docstring."
        ),
    )

    class Meta:
        abstract = True


class BaseModel(UUIDModel, TimeStampedModel):
    """
    The default foundation for platform-level and domain models:
    UUID identity + creation/modification timestamps.

    Deliberately does NOT include soft-delete behavior — see
    ``SoftDeleteModel`` for why that is a separate, opt-in mixin rather
    than being folded in here.

    Usage in a domain app:

        class Product(BaseModel):
            ...

        class Supplier(BaseModel, SoftDeleteModel):
            ...
    """

    class Meta:
        abstract = True


# ---------------------------------------------------------------------------
# Soft-delete infrastructure
# ---------------------------------------------------------------------------


class SoftDeleteQuerySet(models.QuerySet):
    """
    QuerySet with explicit, hard-to-misuse soft-delete semantics.

    ``.delete()`` on this queryset is a BULK SOFT delete (an UPDATE, not
    a DELETE) to keep the "objects.delete() never physically destroys
    rows" invariant true for bulk operations too, matching instance-level
    behavior on ``SoftDeleteModel``. Physical bulk removal must go
    through the explicitly-named ``.hard_delete()``.
    """

    def active(self) -> "SoftDeleteQuerySet":
        return self.filter(is_deleted=False)

    def deleted(self) -> "SoftDeleteQuerySet":
        return self.filter(is_deleted=True)

    def delete(self) -> tuple[int, dict[str, int]]:
        """Bulk soft-delete: flips is_deleted/deleted_at, no physical DELETE."""
        now = timezone.now()
        updated = self.update(is_deleted=True, deleted_at=now)
        return updated, {self.model._meta.label: updated}

    def hard_delete(self) -> tuple[int, dict[str, int]]:
        """Bulk PHYSICAL delete. Explicit, intentional, bypasses soft-delete."""
        return super().delete()

    def restore(self) -> int:
        """Bulk restore: clears is_deleted/deleted_at."""
        return self.update(is_deleted=False, deleted_at=None)


class SoftDeleteManager(models.Manager):
    """
    Default manager for soft-deletable models: exposes ACTIVE rows only.

    This is intentionally the manager assigned to ``objects`` (and thus
    becomes ``_default_manager``), so ordinary application code, the
    Django admin, and related-object access (``parent.children.all()``)
    all see active rows by default without extra ceremony — matching
    the "difficult to misuse" requirement: you have to explicitly reach
    for ``all_objects`` to see deleted rows.
    """

    def get_queryset(self) -> SoftDeleteQuerySet:
        return SoftDeleteQuerySet(self.model, using=self._db).active()


class AllObjectsManager(models.Manager):
    """Manager exposing every row, active or soft-deleted, unfiltered."""

    def get_queryset(self) -> SoftDeleteQuerySet:
        return SoftDeleteQuerySet(self.model, using=self._db)


class SoftDeleteModel(models.Model):
    """
    Opt-in abstract mixin providing reversible ("soft") deletion.

    Composed explicitly by domain models that need it, e.g.:

        class Supplier(BaseModel, SoftDeleteModel):
            ...

    Deletion semantics (explicit, per requirement #11):
        - ``instance.delete()``      -> SOFT delete (default, safe).
        - ``instance.hard_delete()`` -> PHYSICAL delete (explicit, rare).
        - ``instance.restore()``     -> clears the soft-delete state.

    Manager surface:
        - ``Model.objects``      -> active (non-deleted) rows only.
        - ``Model.all_objects``  -> every row, active or deleted.

    Important, deliberately documented limitations:

    1. Reverse relations. Django's reverse FK/related managers
       (``parent.children.all()``) use the model's default manager
       (``objects``) unless a manager is customized on the related
       descriptor. That means soft-deleted children are, by default,
       invisible via reverse traversal too — consistent with the
       "objects = active only" contract, but worth knowing: if a
       domain app needs to traverse *all* related rows including
       deleted ones, it must do so via ``all_objects`` explicitly
       (e.g. ``Child.all_objects.filter(parent=parent)``).

    2. Cascading on hard delete. Django's deletion collector (which
       walks ``on_delete=CASCADE`` relationships) uses the model's
       *base* manager, not its default manager. If the base manager
       filtered out soft-deleted rows, physically hard-deleting a
       parent could fail to find already soft-deleted children,
       potentially leaving orphaned rows or raising IntegrityError
       depending on the FK's ``on_delete``. To prevent this, this
       class sets ``Meta.base_manager_name = "all_objects"`` so the
       collector always sees every row, while ``objects`` (the
       *default* manager used by everyday code, admin, and related
       managers) remains active-only.

    3. Cascading on soft delete. Soft-deleting a parent does NOT
       automatically soft-delete its children — Django's CASCADE
       machinery only runs on a real physical ``DELETE``. If a domain
       app needs "soft-deleting the parent should soft-delete its
       children", that cascade must be implemented explicitly in that
       app's service layer. Core does not guess at this on your behalf.

    4. Uniqueness. A soft-deleted row still physically exists, so a
       plain ``unique=True``/``UniqueConstraint`` on a field will still
       block creating a new row with the same value after the old one
       is "deleted". Domain models that need "unique among active rows
       only" should use a partial unique constraint, e.g.:

           models.UniqueConstraint(
               fields=["sku"],
               condition=models.Q(is_deleted=False),
               name="unique_active_sku",
           )

       This is a decision for the owning domain model, not something
       Core can safely impose generically.
    """

    is_deleted = models.BooleanField(
        default=False,
        db_index=True,
        help_text="True if this record has been soft-deleted.",
    )
    deleted_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text="When this record was soft-deleted; null while active.",
    )

    objects = SoftDeleteManager()
    all_objects = AllObjectsManager()

    class Meta:
        abstract = True
        base_manager_name = "all_objects"
        constraints = [
            models.CheckConstraint(
                condition=(
                    models.Q(is_deleted=False, deleted_at__isnull=True)
                    | models.Q(is_deleted=True, deleted_at__isnull=False)
                ),
                name="%(app_label)s_%(class)s_soft_delete_state_valid",
            ),
        ]

    def soft_delete(self, using: str | None = None) -> None:
        """Mark this record as deleted without physically removing it."""
        self.is_deleted = True
        self.deleted_at = timezone.now()
        self.save(using=using, update_fields=["is_deleted", "deleted_at"])

    def restore(self, using: str | None = None) -> None:
        """Clear soft-delete state, making the record active again."""
        self.is_deleted = False
        self.deleted_at = None
        self.save(using=using, update_fields=["is_deleted", "deleted_at"])

    def delete(self, using: str | None = None, *args: Any, **kwargs: Any) -> None:
        """
        Overridden to mean SOFT delete by default. This is deliberate:
        normal application code calling ``instance.delete()`` should
        never be able to accidentally physically destroy data. Use
        ``hard_delete()`` for that, explicitly.
        """
        self.soft_delete(using=using)

    def hard_delete(self, using: str | None = None, *args: Any, **kwargs: Any):
        """Explicit, intentional PHYSICAL delete. Bypasses soft-delete."""
        return super().delete(using=using, *args, **kwargs)

    @property
    def is_active(self) -> bool:
        return not self.is_deleted


# ---------------------------------------------------------------------------
# Audit infrastructure
# ---------------------------------------------------------------------------


class ActorType(models.TextChoices):
    """Who/what initiated an audited action.

    Deliberately NOT a foreign key to the authentication User model:
    Core must not depend on the authentication app (or any app), and
    "actor" must also represent non-human actors (background jobs, AI
    agents, external integrations) that may not have a User row at all.
    """

    USER = "user", "User"
    SYSTEM = "system", "System"
    AI = "ai", "AI"
    INTEGRATION = "integration", "Integration"
    API = "api", "API"


class AuditAction(models.TextChoices):
    """
    Generic, domain-agnostic action vocabulary.

    This is intentionally small. Domain-specific nuance (e.g. "Shopify
    order sync" vs. "supplier price sync") belongs in ``resource_type``
    and ``metadata``, not in an ever-growing Core enum that every app
    would need to keep patching.
    """

    CREATE = "create", "Create"
    UPDATE = "update", "Update"
    DELETE = "delete", "Delete"
    RESTORE = "restore", "Restore"
    LOGIN = "login", "Login"
    SYNC = "sync", "Sync"
    IMPORT = "import", "Import"
    EXPORT = "export", "Export"
    APPROVE = "approve", "Approve"
    REJECT = "reject", "Reject"
    EXECUTE = "execute", "Execute"
    OTHER = "other", "Other"


class AuditEvent(UUIDModel):
    """
    Generic, append-only record answering: who/what changed data, what
    changed, when, and in what operational context.

    Why this belongs in Core: every domain app (products, suppliers,
    orders, integrations, ai, ...) needs auditability, and duplicating
    an audit table per app would fragment cross-domain investigations
    (e.g. "show everything that happened in correlation X across
    Shopify sync, pricing, and inventory"). A single, generic table
    lets that query exist at all.

    How a resource is identified (deliberately no ForeignKey):
        ``resource_type`` is a dotted "app_label.ModelName" string, e.g.
        "products.Product" or "integrations.ShopifyStore".
        ``resource_id`` is that object's stringified primary key.
        This lets any app write audit events for any model without
        Core importing that model (which would create circular
        imports) and without a GenericForeignKey (which would require
        Core to depend on ``django.contrib.contenttypes`` and would
        make querying by ``resource_type`` string less direct than a
        plain indexed CharField). The trade-off is that the database
        cannot enforce that ``resource_id`` actually points at a real
        row — this table is a log, not a referential-integrity-backed
        relationship, and is not queried via ``select_related``.

    Change payload (deliberately lightweight, not full snapshots):
        ``changes`` stores a small structured diff, e.g.
        ``{"price": {"old": "9.99", "new": "12.99"}}``, rather than a
        full before/after object snapshot. Full snapshots were
        considered and rejected as the default: they grow audit-table
        size roughly with full object size (not the size of what
        actually changed), and materially increase the risk of a
        sensitive field being copied into the audit log by accident.
        Callers with a genuine need for full snapshots can put a
        deliberately-curated, secret-free subset into ``metadata``.

    Security: ``changes``/``metadata`` must NEVER contain passwords,
    tokens, API keys, or other credentials. This is a service-layer/
    code-review responsibility — the database cannot know what a JSON
    value "means".

    Immutability: audit rows are append-only. ``save()`` refuses any
    update to an existing row (only the initial INSERT is allowed) and
    ``delete()`` is disabled outright. Retention/purge, if ever needed,
    must go through an explicit administrative script operating outside
    the ORM's normal `delete()` path (e.g. raw QuerySet `.delete()` on
    `AuditEvent.objects` run by an ops process), not through casual
    application code.

    Correlation vs. request:
        ``correlation_id`` identifies a broader multi-step business
        workflow (e.g. one Shopify webhook triggering a sync, a
        supplier lookup, and an AI repricing decision) and is
        deliberately decoupled from any single HTTP request so
        background jobs can share it too. A separate ``request_id`` is
        NOT included here: an individual HTTP request identifier is a
        tracing/observability concern (APM, logs) with much higher
        cardinality and much lower long-term business value than a
        workflow correlation id, and does not need a permanent, indexed
        database column. If a specific event genuinely needs to record
        the originating request, it can be placed in ``metadata``.
    """

    actor_type = models.CharField(max_length=20, choices=ActorType.choices)
    actor_id = models.CharField(
        max_length=255,
        null=True,
        blank=True,
        help_text=(
            "Identifier of the actor within its actor_type namespace "
            "(e.g. a user's UUID as a string, an integration's slug, "
            "an AI agent's identifier). Null for actor_type=SYSTEM when "
            "there is no meaningful sub-identifier."
        ),
    )

    action = models.CharField(max_length=20, choices=AuditAction.choices)

    resource_type = models.CharField(
        max_length=255,
        help_text='Dotted reference to the affected model, e.g. "products.Product".',
    )
    resource_id = models.CharField(
        max_length=255,
        help_text="Stringified primary key of the affected resource.",
    )

    changes = models.JSONField(
        default=dict,
        blank=True,
        help_text=(
            "Lightweight structured diff, e.g. "
            '{"field": {"old": ..., "new": ...}}. Never store secrets. '
            "Keep small — this is a change summary, not a full snapshot."
        ),
    )
    metadata = models.JSONField(
        default=dict,
        blank=True,
        help_text="Additional non-secret context for this event.",
    )

    correlation_id = models.UUIDField(
        null=True,
        blank=True,
        help_text="Groups events belonging to the same multi-step workflow.",
    )

    created_at = models.DateTimeField(
        auto_now_add=True,
        help_text="When this event was recorded (UTC). Immutable.",
    )

    class Meta:
        verbose_name = "Audit event"
        verbose_name_plural = "Audit events"
        ordering = ["-created_at"]
        indexes = [
            # "audit events for a resource" — the most common lookup.
            models.Index(fields=["resource_type", "resource_id"], name="audit_by_resource"),
            # "audit events by actor"
            models.Index(fields=["actor_type", "actor_id"], name="audit_by_actor"),
            # "audit events by action" (e.g. all DELETEs in a range)
            models.Index(fields=["action", "created_at"], name="audit_by_action_time"),
            # chronological range scans / "recent activity" views
            models.Index(fields=["created_at"], name="audit_by_created_at"),
            # "everything in this workflow"
            models.Index(fields=["correlation_id"], name="audit_by_correlation"),
        ]
        constraints = [
            models.CheckConstraint(
                condition=models.Q(actor_type__in=ActorType.values),
                name="audit_event_valid_actor_type",
            ),
            models.CheckConstraint(
                condition=models.Q(action__in=AuditAction.values),
                name="audit_event_valid_action",
            ),
        ]

    def __str__(self) -> str:
        return f"{self.get_action_display()} {self.resource_type}:{self.resource_id}"

    def save(self, *args: Any, **kwargs: Any) -> None:
        """Allow the initial INSERT only; reject any later UPDATE."""
        if not self._state.adding:
            raise ValueError(
                "AuditEvent records are immutable and cannot be updated "
                "after creation."
            )
        super().save(*args, **kwargs)

    def delete(self, *args: Any, **kwargs: Any) -> None:
        """Audit records are append-only; there is no normal delete path."""
        raise PermissionError(
            "AuditEvent records cannot be deleted through the ORM. "
            "Retention/purge must be handled by an explicit, audited "
            "administrative process."
        )


# ---------------------------------------------------------------------------
# Idempotency infrastructure
# ---------------------------------------------------------------------------


class IdempotencyStatus(models.TextChoices):
    PENDING = "pending", "Pending"
    COMPLETED = "completed", "Completed"
    FAILED = "failed", "Failed"


class IdempotencyRecord(BaseModel):
    """
    Prevents a side-effecting operation from being performed twice when
    a caller (a client, or an external system like Shopify) retries the
    same logical request.

    Why this belongs in Core rather than ``integrations``: idempotent
    request handling is not specific to Shopify or any one integration
    — it is needed anywhere Ship exposes an API that must tolerate
    network retries, and anywhere Ship calls out to an external system
    that itself retries webhooks/callbacks. Building it once in Core
    avoids every future integration reimplementing the same
    key/scope/expiry/status machinery.

    Usage pattern:
        1. Caller supplies (or Ship derives) an idempotency key and a
           scope naming the operation, e.g.
           scope="shopify.webhook.orders_create", key=<shopify event id>.
        2. Before performing the operation, look up
           ``IdempotencyRecord.objects.filter(scope=scope, key=key)``.
           If found and COMPLETED, return the cached response instead
           of re-running the operation.
        3. If not found, create a PENDING record (the unique constraint
           on (scope, key) makes concurrent duplicate creation fail
           safely at the database level — the loser of the race should
           treat the IntegrityError as "someone else is handling this").
        4. On completion, call ``mark_completed()``/``mark_failed()``.

    ``request_hash`` (SHA-256 hex digest of the normalized request
    payload) exists for conflict detection: if the same
    ``(scope, key)`` is reused with a *different* payload, that is a
    client bug or a key collision, not a legitimate retry, and should
    be rejected by the caller rather than silently returning a stale
    cached response.

    ``response_body`` is a small cached JSON response returned to
    retrying callers. It must never contain secrets, and should stay
    small (this is a response cache, not a general blob store).

    Expiry (``expires_at``) bounds how long a key is remembered. This
    model does not perform its own cleanup; a periodic operational job
    (outside Core) is expected to delete expired rows.
    """

    key = models.CharField(
        max_length=255,
        help_text="Caller-supplied or derived idempotency key.",
    )
    scope = models.CharField(
        max_length=255,
        help_text=(
            'Namespace for the key, e.g. "shopify.webhook.orders_create" '
            'or "api.orders.create", preventing collisions across '
            "unrelated operations that might reuse the same key value."
        ),
    )
    request_hash = models.CharField(
        max_length=64,
        blank=True,
        help_text="SHA-256 hex digest of the normalized request payload, for conflict detection.",
    )
    status = models.CharField(
        max_length=20,
        choices=IdempotencyStatus.choices,
        default=IdempotencyStatus.PENDING,
    )
    response_status = models.PositiveSmallIntegerField(
        null=True,
        blank=True,
        help_text="HTTP-style status code of the cached result, if applicable.",
    )
    response_body = models.JSONField(
        null=True,
        blank=True,
        help_text="Small cached response returned to retrying callers. Never store secrets.",
    )
    expires_at = models.DateTimeField(
        help_text="When this key may be forgotten/reused. Cleanup is an external job.",
    )

    class Meta:
        verbose_name = "Idempotency record"
        verbose_name_plural = "Idempotency records"
        ordering = ["-created_at"]
        constraints = [
            models.UniqueConstraint(
                fields=["scope", "key"],
                name="unique_idempotency_key_per_scope",
            ),
        ]
        indexes = [
            models.Index(fields=["expires_at"], name="idempotency_by_expiry"),
            models.Index(fields=["status"], name="idempotency_by_status"),
        ]

    def __str__(self) -> str:
        return f"{self.scope}:{self.key} [{self.status}]"

    @property
    def is_expired(self) -> bool:
        return timezone.now() >= self.expires_at

    def mark_completed(
        self, response_status: int | None = None, response_body: dict | None = None
    ) -> None:
        self.status = IdempotencyStatus.COMPLETED
        self.response_status = response_status
        self.response_body = response_body
        self.save(update_fields=["status", "response_status", "response_body", "updated_at"])

    def mark_failed(self) -> None:
        self.status = IdempotencyStatus.FAILED
        self.save(update_fields=["status", "updated_at"])


# ---------------------------------------------------------------------------
# Inbound event de-duplication
# ---------------------------------------------------------------------------


class ProcessedEventStatus(models.TextChoices):
    RECEIVED = "received", "Received"
    PROCESSING = "processing", "Processing"
    PROCESSED = "processed", "Processed"
    FAILED = "failed", "Failed"


class ProcessedEvent(BaseModel):
    """
    Tracks inbound external events (webhooks, callbacks) to prevent the
    same event from being processed more than once.

    Distinct from ``AuditEvent`` and from ``IdempotencyRecord``:
        - ``AuditEvent`` records what Ship *did* and why, for business/
          compliance visibility, keyed by (actor, action, resource).
        - ``IdempotencyRecord`` de-duplicates *requests Ship receives on
          its own API* (or operations Ship itself initiates), keyed by
          a caller-supplied idempotency key.
        - ``ProcessedEvent`` de-duplicates *events an external system
          delivers to Ship* (e.g. a Shopify webhook redelivery), keyed
          by that system's own event identifier — which Ship does not
          control and which is not necessarily a client-supplied
          idempotency key at all.
        Collapsing these into one model would conflate a compliance
        log, a request-dedup cache, and a webhook-delivery ledger that
        have different volumes, different retention needs, and
        different callers.

    Usage pattern: on receiving a webhook, upsert-attempt a row keyed
    on ``(source, external_event_id)``. The unique constraint makes a
    duplicate delivery fail fast at the database level; the handler
    should treat that as "already seen, skip" rather than an error.

    ``metadata`` is intentionally for small, structured context (e.g.
    shop domain, topic) — not a dumping ground for the full webhook
    body, which may be large and could contain data that shouldn't be
    retained indefinitely in Core.
    """

    source = models.CharField(
        max_length=100,
        help_text='System that delivered the event, e.g. "shopify", "stripe".',
    )
    external_event_id = models.CharField(
        max_length=255,
        help_text="Event identifier as assigned by the source system.",
    )
    event_type = models.CharField(
        max_length=150,
        help_text='Event type/topic as reported by the source, e.g. "orders/create".',
    )
    status = models.CharField(
        max_length=20,
        choices=ProcessedEventStatus.choices,
        default=ProcessedEventStatus.RECEIVED,
    )
    processed_at = models.DateTimeField(null=True, blank=True)
    metadata = models.JSONField(
        default=dict,
        blank=True,
        help_text="Small structured context about the event. Not the full raw payload.",
    )

    class Meta:
        verbose_name = "Processed event"
        verbose_name_plural = "Processed events"
        ordering = ["-created_at"]
        constraints = [
            models.UniqueConstraint(
                fields=["source", "external_event_id"],
                name="unique_processed_event_per_source",
            ),
        ]
        indexes = [
            models.Index(fields=["source", "event_type"], name="proc_event_by_source_type"),
            models.Index(fields=["status"], name="proc_event_by_status"),
            models.Index(fields=["processed_at"], name="proc_event_by_processed_at"),
        ]

    def __str__(self) -> str:
        return f"{self.source}:{self.external_event_id} ({self.event_type})"

    def mark_processing(self) -> None:
        self.status = ProcessedEventStatus.PROCESSING
        self.save(update_fields=["status", "updated_at"])

    def mark_processed(self) -> None:
        self.status = ProcessedEventStatus.PROCESSED
        self.processed_at = timezone.now()
        self.save(update_fields=["status", "processed_at", "updated_at"])

    def mark_failed(self) -> None:
        self.status = ProcessedEventStatus.FAILED
        self.save(update_fields=["status", "updated_at"])