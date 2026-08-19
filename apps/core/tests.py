"""
apps.core.tests
================

Behavioral tests for apps.core.models, run against the real, finalized
model definitions. No fields or methods are invented here.

One exception, explained: SoftDeleteModel is an abstract mixin with no
concrete model of its own in apps.core.models. To exercise its actual
behavior (not just read the code), this module defines a minimal,
test-only concrete model, `SoftDeletableTestThing(BaseModel,
SoftDeleteModel)`, and creates/drops its table manually around the test
class using the schema editor. This is test-only scaffolding — nothing
in apps/core/models.py is modified, and no migration is added for it.
"""

import uuid

from django.db import IntegrityError, connection, models, transaction
from django.test import TestCase, TransactionTestCase
from django.utils import timezone

from apps.core.models import (
    ActorType,
    AuditAction,
    AuditEvent,
    BaseModel,
    IdempotencyRecord,
    IdempotencyStatus,
    ProcessedEvent,
    ProcessedEventStatus,
    SoftDeleteModel,
)


class SoftDeletableTestThing(BaseModel, SoftDeleteModel):
    """
    Test-only concrete model used solely to exercise SoftDeleteModel.

    NOTE: Meta explicitly subclasses SoftDeleteModel.Meta
    (`class Meta(SoftDeleteModel.Meta)`), not a bare `class Meta`. This
    is required — see test_meta_must_subclass_parent_to_inherit_
    constraints_and_base_manager below for why declaring a bare Meta
    silently drops the abstract base's CheckConstraint and
    base_manager_name. This is standard Django Meta-inheritance
    behavior, not a bug in this test file.
    """

    name = models.CharField(max_length=50)

    class Meta(SoftDeleteModel.Meta):
        app_label = "core"


class SoftDeleteTableMixin:
    """Creates/drops the test-only table around a TestCase's DB transaction."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        with connection.schema_editor() as editor:
            editor.create_model(SoftDeletableTestThing)

    @classmethod
    def tearDownClass(cls):
        with connection.schema_editor() as editor:
            editor.delete_model(SoftDeletableTestThing)
        super().tearDownClass()


# ---------------------------------------------------------------------------
# UUID / timestamp foundation
# ---------------------------------------------------------------------------


class BaseModelIdentityTests(SoftDeleteTableMixin, TransactionTestCase):
    def test_id_is_uuid_and_auto_generated(self):
        obj = SoftDeletableTestThing.objects.create(name="a")
        self.assertIsInstance(obj.id, uuid.UUID)

    def test_two_instances_get_distinct_uuids(self):
        a = SoftDeletableTestThing.objects.create(name="a")
        b = SoftDeletableTestThing.objects.create(name="b")
        self.assertNotEqual(a.id, b.id)

    def test_id_field_not_editable(self):
        field = SoftDeletableTestThing._meta.get_field("id")
        self.assertFalse(field.editable)

    def test_created_at_set_on_creation_and_timezone_aware(self):
        obj = SoftDeletableTestThing.objects.create(name="a")
        self.assertIsNotNone(obj.created_at)
        self.assertTrue(timezone.is_aware(obj.created_at))

    def test_updated_at_changes_on_save(self):
        obj = SoftDeletableTestThing.objects.create(name="a")
        first_updated_at = obj.updated_at
        obj.name = "b"
        obj.save(update_fields=["name", "updated_at"])
        obj.refresh_from_db()
        self.assertGreaterEqual(obj.updated_at, first_updated_at)


# ---------------------------------------------------------------------------
# Soft delete
# ---------------------------------------------------------------------------


class SoftDeleteModelTests(SoftDeleteTableMixin, TransactionTestCase):
    def test_soft_delete_marks_deleted_and_sets_deleted_at(self):
        obj = SoftDeletableTestThing.objects.create(name="a")
        obj.soft_delete()
        self.assertTrue(obj.is_deleted)
        self.assertIsNotNone(obj.deleted_at)

    def test_restore_clears_deletion_state(self):
        obj = SoftDeletableTestThing.objects.create(name="a")
        obj.soft_delete()
        obj.restore()
        self.assertFalse(obj.is_deleted)
        self.assertIsNone(obj.deleted_at)

    def test_is_active_property(self):
        obj = SoftDeletableTestThing.objects.create(name="a")
        self.assertTrue(obj.is_active)
        obj.soft_delete()
        self.assertFalse(obj.is_active)

    def test_instance_delete_performs_soft_delete(self):
        obj = SoftDeletableTestThing.objects.create(name="a")
        pk = obj.pk
        obj.delete()
        self.assertTrue(SoftDeletableTestThing.all_objects.filter(pk=pk, is_deleted=True).exists())
        self.assertTrue(SoftDeletableTestThing.all_objects.filter(pk=pk).exists())

    def test_hard_delete_physically_removes_row(self):
        obj = SoftDeletableTestThing.objects.create(name="a")
        pk = obj.pk
        obj.hard_delete()
        self.assertFalse(SoftDeletableTestThing.all_objects.filter(pk=pk).exists())

    def test_default_manager_excludes_soft_deleted(self):
        active = SoftDeletableTestThing.objects.create(name="active")
        deleted = SoftDeletableTestThing.objects.create(name="deleted")
        deleted.soft_delete()

        active_ids = set(SoftDeletableTestThing.objects.values_list("pk", flat=True))
        self.assertIn(active.pk, active_ids)
        self.assertNotIn(deleted.pk, active_ids)

    def test_all_objects_includes_active_and_deleted(self):
        active = SoftDeletableTestThing.objects.create(name="active")
        deleted = SoftDeletableTestThing.objects.create(name="deleted")
        deleted.soft_delete()

        all_ids = set(SoftDeletableTestThing.all_objects.values_list("pk", flat=True))
        self.assertIn(active.pk, all_ids)
        self.assertIn(deleted.pk, all_ids)

    def test_queryset_active_and_deleted_filters(self):
        """
        NOTE: SoftDeleteManager/AllObjectsManager in models.py are plain
        `models.Manager` subclasses that override `get_queryset()` — they
        are not built with `Manager.from_queryset(SoftDeleteQuerySet)` (or
        `SoftDeleteQuerySet.as_manager()`), so `.active()`/`.deleted()`/
        `.restore()`/`.hard_delete()` are NOT reachable directly off
        `Model.objects`/`Model.all_objects` (that raises AttributeError —
        confirmed while writing this test). They are only reachable via
        `.get_queryset()`, as done below, or transitively through
        `Model.objects.filter(...).delete()` etc., which resolve to
        SoftDeleteQuerySet methods because `objects.filter()` already
        returns a SoftDeleteQuerySet instance. This is a minor ergonomic
        gap worth knowing about (`Model.all_objects.deleted()` reads as
        the natural API but does not exist), not a correctness defect —
        flagged here rather than changed in models.py per instructions.
        """
        active = SoftDeletableTestThing.objects.create(name="active")
        deleted = SoftDeletableTestThing.objects.create(name="deleted")
        deleted.soft_delete()

        active_qs = SoftDeletableTestThing.all_objects.get_queryset().active()
        deleted_qs = SoftDeletableTestThing.all_objects.get_queryset().deleted()

        self.assertIn(active.pk, active_qs.values_list("pk", flat=True))
        self.assertNotIn(deleted.pk, active_qs.values_list("pk", flat=True))
        self.assertIn(deleted.pk, deleted_qs.values_list("pk", flat=True))
        self.assertNotIn(active.pk, deleted_qs.values_list("pk", flat=True))

    def test_manager_does_not_proxy_queryset_only_methods(self):
        """Documents the gap described above rather than silently avoiding it."""
        with self.assertRaises(AttributeError):
            SoftDeletableTestThing.all_objects.deleted()

    def test_queryset_bulk_delete_is_soft(self):
        obj = SoftDeletableTestThing.objects.create(name="a")
        SoftDeletableTestThing.objects.filter(pk=obj.pk).delete()

        self.assertFalse(SoftDeletableTestThing.objects.filter(pk=obj.pk).exists())
        row = SoftDeletableTestThing.all_objects.get(pk=obj.pk)
        self.assertTrue(row.is_deleted)
        self.assertIsNotNone(row.deleted_at)

    def test_queryset_hard_delete_is_physical(self):
        obj = SoftDeletableTestThing.objects.create(name="a")
        SoftDeletableTestThing.all_objects.filter(pk=obj.pk).hard_delete()
        self.assertFalse(SoftDeletableTestThing.all_objects.filter(pk=obj.pk).exists())

    def test_queryset_restore(self):
        obj = SoftDeletableTestThing.objects.create(name="a")
        obj.soft_delete()
        SoftDeletableTestThing.all_objects.filter(pk=obj.pk).restore()
        self.assertFalse(SoftDeletableTestThing.objects.filter(pk=obj.pk, is_deleted=True).exists())
        self.assertTrue(SoftDeletableTestThing.objects.filter(pk=obj.pk).exists())

    def test_invalid_soft_delete_state_rejected_by_db_constraint(self):
        """is_deleted=True with deleted_at=NULL must violate the CheckConstraint."""
        obj = SoftDeletableTestThing.objects.create(name="a")
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                SoftDeletableTestThing.all_objects.filter(pk=obj.pk).update(
                    is_deleted=True, deleted_at=None
                )

    def test_meta_must_subclass_parent_to_inherit_constraints_and_base_manager(self):
        """
        Documents a real Django Meta-inheritance gotcha that affects every
        future domain model built on SoftDeleteModel: Django only carries
        an abstract base's `Meta.constraints`/`Meta.base_manager_name`
        (etc.) onto a concrete subclass if that subclass either declares
        no Meta at all, or explicitly subclasses the parent's Meta
        (`class Meta(SoftDeleteModel.Meta)`). A bare `class Meta: ...` on
        the child SILENTLY drops both the soft-delete state
        CheckConstraint and the `base_manager_name = "all_objects"`
        cascade-safety fix, with no error raised anywhere.

        This is standard Django behavior, not a defect in models.py, but
        it is exactly the kind of thing a domain-app author is likely to
        get wrong when they add their own Meta (e.g. for `ordering`).
        Flagging it here rather than in models.py per instructions not to
        modify the finalized model file; consider calling this out in
        SoftDeleteModel's docstring or a docs/CONTRIBUTING note so every
        domain app remembers `class Meta(SoftDeleteModel.Meta)`.
        """
        class BareMetaThing(BaseModel, SoftDeleteModel):
            class Meta:
                app_label = "core"

        class SubclassedMetaThing(BaseModel, SoftDeleteModel):
            class Meta(SoftDeleteModel.Meta):
                app_label = "core"

        self.assertIsNone(BareMetaThing._meta.base_manager_name)
        self.assertEqual(BareMetaThing._meta.constraints, [])

        self.assertEqual(SubclassedMetaThing._meta.base_manager_name, "all_objects")
        self.assertEqual(len(SubclassedMetaThing._meta.constraints), 1)


# ---------------------------------------------------------------------------
# AuditEvent
# ---------------------------------------------------------------------------


class AuditEventTests(TestCase):
    def _make_event(self, **overrides):
        defaults = dict(
            actor_type=ActorType.USER,
            actor_id="user-123",
            action=AuditAction.UPDATE,
            resource_type="products.Product",
            resource_id=str(uuid.uuid4()),
            changes={"price": {"old": "9.99", "new": "12.99"}},
            metadata={"source": "admin-panel"},
            correlation_id=uuid.uuid4(),
        )
        defaults.update(overrides)
        return AuditEvent.objects.create(**defaults)

    def test_create_audit_event(self):
        event = self._make_event()
        self.assertIsInstance(event.id, uuid.UUID)
        self.assertEqual(event.actor_type, ActorType.USER)
        self.assertEqual(event.action, AuditAction.UPDATE)

    def test_choice_fields_accept_declared_enum_values(self):
        event = self._make_event(actor_type=ActorType.AI, action=AuditAction.EXECUTE)
        event.refresh_from_db()
        self.assertEqual(event.actor_type, ActorType.AI)
        self.assertEqual(event.action, AuditAction.EXECUTE)

    def test_json_fields_round_trip(self):
        event = self._make_event(
            changes={"status": {"old": "draft", "new": "active"}},
            metadata={"ip": "127.0.0.1"},
        )
        event.refresh_from_db()
        self.assertEqual(event.changes["status"]["new"], "active")
        self.assertEqual(event.metadata["ip"], "127.0.0.1")

    def test_correlation_id_is_nullable_and_queryable(self):
        cid = uuid.uuid4()
        self._make_event(correlation_id=cid)
        self._make_event(correlation_id=None)
        self.assertEqual(AuditEvent.objects.filter(correlation_id=cid).count(), 1)
        self.assertEqual(AuditEvent.objects.filter(correlation_id__isnull=True).count(), 1)

    def test_str_representation(self):
        event = self._make_event(
            action=AuditAction.DELETE,
            resource_type="suppliers.Supplier",
            resource_id="abc-123",
        )
        self.assertIn("suppliers.Supplier:abc-123", str(event))

    def test_audit_event_cannot_be_updated_after_creation(self):
        event = self._make_event()
        event.action = AuditAction.DELETE
        with self.assertRaises(ValueError):
            event.save()

    def test_audit_event_cannot_be_deleted_via_instance_delete(self):
        event = self._make_event()
        with self.assertRaises(PermissionError):
            event.delete()
        self.assertTrue(AuditEvent.objects.filter(pk=event.pk).exists())

    def test_invalid_actor_type_rejected_by_db_constraint(self):
        """Bypasses Python-level `choices` validation to hit the CheckConstraint directly."""
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                with connection.cursor() as cursor:
                    cursor.execute(
                        "INSERT INTO core_auditevent "
                        "(id, actor_type, actor_id, action, resource_type, "
                        "resource_id, changes, metadata, correlation_id, created_at) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                        [
                            str(uuid.uuid4()),
                            "not-a-real-actor-type",
                            "x",
                            AuditAction.CREATE,
                            "products.Product",
                            "1",
                            "{}",
                            "{}",
                            None,
                            timezone.now(),
                        ],
                    )

    def test_lookup_by_resource_returns_matching_events_only(self):
        rid = str(uuid.uuid4())
        self._make_event(resource_type="products.Product", resource_id=rid)
        self._make_event(resource_type="products.Product", resource_id=str(uuid.uuid4()))
        matches = AuditEvent.objects.filter(resource_type="products.Product", resource_id=rid)
        self.assertEqual(matches.count(), 1)


# ---------------------------------------------------------------------------
# IdempotencyRecord
# ---------------------------------------------------------------------------


class IdempotencyRecordTests(TestCase):
    def test_create_idempotency_record(self):
        record = IdempotencyRecord.objects.create(
            key="key-1", scope="api.orders.create", expires_at=timezone.now()
        )
        self.assertEqual(record.status, IdempotencyStatus.PENDING)

    def test_scope_and_key_uniqueness_enforced(self):
        IdempotencyRecord.objects.create(
            key="key-1", scope="api.orders.create", expires_at=timezone.now()
        )
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                IdempotencyRecord.objects.create(
                    key="key-1", scope="api.orders.create", expires_at=timezone.now()
                )

    def test_same_key_different_scope_is_allowed(self):
        IdempotencyRecord.objects.create(
            key="key-1", scope="api.orders.create", expires_at=timezone.now()
        )
        IdempotencyRecord.objects.create(
            key="key-1", scope="shopify.webhook.orders_create", expires_at=timezone.now()
        )
        self.assertEqual(IdempotencyRecord.objects.filter(key="key-1").count(), 2)

    def test_is_expired_true_in_past_false_in_future(self):
        past = IdempotencyRecord.objects.create(
            key="past", scope="s", expires_at=timezone.now() - timezone.timedelta(seconds=1)
        )
        future = IdempotencyRecord.objects.create(
            key="future", scope="s", expires_at=timezone.now() + timezone.timedelta(hours=1)
        )
        self.assertTrue(past.is_expired)
        self.assertFalse(future.is_expired)

    def test_mark_completed_updates_status_and_response(self):
        record = IdempotencyRecord.objects.create(
            key="k", scope="s", expires_at=timezone.now() + timezone.timedelta(hours=1)
        )
        record.mark_completed(response_status=201, response_body={"id": "abc"})
        record.refresh_from_db()
        self.assertEqual(record.status, IdempotencyStatus.COMPLETED)
        self.assertEqual(record.response_status, 201)
        self.assertEqual(record.response_body, {"id": "abc"})

    def test_mark_failed_updates_status(self):
        record = IdempotencyRecord.objects.create(
            key="k", scope="s", expires_at=timezone.now() + timezone.timedelta(hours=1)
        )
        record.mark_failed()
        record.refresh_from_db()
        self.assertEqual(record.status, IdempotencyStatus.FAILED)

    def test_updated_at_advances_on_status_change(self):
        record = IdempotencyRecord.objects.create(
            key="k", scope="s", expires_at=timezone.now() + timezone.timedelta(hours=1)
        )
        original_updated_at = record.updated_at
        record.mark_completed()
        record.refresh_from_db()
        self.assertGreaterEqual(record.updated_at, original_updated_at)


# ---------------------------------------------------------------------------
# ProcessedEvent
# ---------------------------------------------------------------------------


class ProcessedEventTests(TestCase):
    def test_create_processed_event(self):
        event = ProcessedEvent.objects.create(
            source="shopify", external_event_id="evt-1", event_type="orders/create"
        )
        self.assertEqual(event.status, ProcessedEventStatus.RECEIVED)

    def test_source_and_external_event_id_uniqueness_enforced(self):
        ProcessedEvent.objects.create(
            source="shopify", external_event_id="evt-1", event_type="orders/create"
        )
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                ProcessedEvent.objects.create(
                    source="shopify", external_event_id="evt-1", event_type="orders/create"
                )

    def test_same_external_event_id_different_source_is_allowed(self):
        ProcessedEvent.objects.create(
            source="shopify", external_event_id="evt-1", event_type="orders/create"
        )
        ProcessedEvent.objects.create(
            source="stripe", external_event_id="evt-1", event_type="charge.succeeded"
        )
        self.assertEqual(ProcessedEvent.objects.filter(external_event_id="evt-1").count(), 2)

    def test_status_transition_received_to_processing_to_processed(self):
        event = ProcessedEvent.objects.create(
            source="shopify", external_event_id="evt-2", event_type="orders/create"
        )
        self.assertEqual(event.status, ProcessedEventStatus.RECEIVED)

        event.mark_processing()
        event.refresh_from_db()
        self.assertEqual(event.status, ProcessedEventStatus.PROCESSING)
        self.assertIsNone(event.processed_at)

        event.mark_processed()
        event.refresh_from_db()
        self.assertEqual(event.status, ProcessedEventStatus.PROCESSED)
        self.assertIsNotNone(event.processed_at)

    def test_status_transition_to_failed(self):
        event = ProcessedEvent.objects.create(
            source="shopify", external_event_id="evt-3", event_type="orders/create"
        )
        event.mark_failed()
        event.refresh_from_db()
        self.assertEqual(event.status, ProcessedEventStatus.FAILED)
        self.assertIsNone(event.processed_at)

    def test_str_representation(self):
        event = ProcessedEvent.objects.create(
            source="shopify", external_event_id="evt-4", event_type="orders/create"
        )
        self.assertIn("shopify:evt-4", str(event))
        self.assertIn("orders/create", str(event))