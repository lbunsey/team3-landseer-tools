"""In-memory patch records. Nothing here reads or writes repository files.

Records disappear when the process exits. Approval functions are trusted
application entry points, never agent tools; this module cannot authenticate
human approval. File validation and application belong to later modules.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
import hashlib
from threading import RLock
from uuid import uuid4


class PatchStatus(str, Enum):
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    SUPERSEDED = "superseded"
    REJECTED = "rejected"
    APPLIED = "applied"


@dataclass(frozen=True)
class PatchRecord:
    patch_id: str
    session_id: str
    repo_name: str
    file_path: str
    original_hash: str
    original_content: bytes
    updated_content: bytes
    diff: str
    created_at: datetime
    parent_patch_id: str | None = None
    status: PatchStatus = PatchStatus.PENDING_REVIEW
    approved_at: datetime | None = None
    applied_at: datetime | None = None
    operation: str = "modify"
    testing_required: bool = False
    origin: str = "general"


_records: dict[str, PatchRecord] = {}
_lock = RLock()
_dataset_scopes: set[tuple[str, str]] = set()
_proposal_context = ContextVar("patch_proposal_context", default=None)


@contextmanager
def model_workflow(session_id: str, repo_name: str):
    """Trusted runtime boundary, not an agent input or permission grant.

    Lets the unchanged add_model implementation use shared patch helpers
    alongside dataset work. Context is isolated across concurrent calls.
    """
    if not session_id or not repo_name:
        raise ValueError("Model workflow requires a session and repository")
    token = _proposal_context.set((session_id, repo_name.strip().rstrip("/").casefold()))
    try:
        yield
    finally:
        _proposal_context.reset(token)


def require_dataset_workflow(session_id: str, repo_name: str) -> None:
    """Bind writes for this conversation/repo to tested dataset batches."""
    with _lock:
        _dataset_scopes.add((session_id, repo_name.casefold()))


def assert_dataset_route(record: PatchRecord) -> None:
    with _lock:
        if (record.session_id, record.repo_name.casefold()) in _dataset_scopes and not record.testing_required and record.origin != "model":
            raise ValueError("Active dataset workflow requires association. In the official CLI use the shared preview_patch; runtime records the task association.")



def get_patch(patch_id: str) -> PatchRecord:
    """Return an immutable snapshot; raise KeyError for an unknown ID."""
    with _lock:
        return _records[patch_id]


def list_session_patches(session_id: str) -> tuple[PatchRecord, ...]:
    """Return immutable snapshots in creation order."""
    with _lock:
        return tuple(p for p in _records.values() if p.session_id == session_id)


def list_pending_patches() -> tuple[PatchRecord, ...]:
    """Return pending proposals; model calls exclude dataset-owned proposals.

    The unchanged add_model checks this list before proposing model changes.
    Dataset proposals must not make its model-only pending check fail.
    Outside that trusted context the complete pending list is returned.
    """
    with _lock:
        context = _proposal_context.get()
        return tuple(p for p in _records.values() if p.status == PatchStatus.PENDING_REVIEW
                     and not (context and p.repo_name.casefold() == context[1] and p.testing_required))


def create_patch_record(
    *,
    session_id: str,
    repo_name: str,
    file_path: str,
    original_hash: str,
    original_content: bytes,
    updated_content: bytes,
    diff: str,
    parent_patch_id: str | None = None,
    operation: str = "modify",
    testing_required: bool = False,
) -> PatchRecord:
    """Save prepared contents, superseding an active parent atomically.

Parent links describe proposal ancestry, not an application dependency.
Restoring a terminal proposal can create a new child without changing it.
Paths, hashes and diffs must be validated by the preview workflow.
"""
    for name, value in (
        ("session_id", session_id), ("repo_name", repo_name),
        ("file_path", file_path), ("original_hash", original_hash),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
    if not isinstance(original_content, bytes) or not isinstance(updated_content, bytes):
        raise TypeError("Contents must be immutable bytes")
    if not isinstance(diff, str):
        raise TypeError("diff must be a string")
    if operation not in {"modify", "create"}:
        raise ValueError("operation must be modify or create")
    if operation == "create" and (original_content != b"" or original_hash != hashlib.sha256(b"").hexdigest()):
        raise ValueError("Create patches must have an empty original and its hash")
    if original_content == updated_content:
        raise ValueError("Patch must change the contents")

    with _lock:
        parent = get_patch(parent_patch_id) if parent_patch_id is not None else None
        if parent is not None and (
            parent.session_id, parent.repo_name, parent.file_path
        ) != (session_id, repo_name, file_path):
            raise ValueError("Parent must belong to the same session and target")
        if parent is not None and parent.operation != operation:
            raise ValueError("Parent must use the same patch operation")
        requires_testing = testing_required or bool(parent and parent.testing_required)
        if requires_testing:
            origin = "dataset"
        elif _proposal_context.get() == (session_id, repo_name.casefold()):
            origin = "model"
        else:
            origin = "general"
        if (session_id, repo_name.casefold()) in _dataset_scopes and not testing_required and origin != "model":
            raise ValueError("Active dataset workflow requires association. Use preview_patch through the official CLI so runtime records the task association.")
        patch_id = uuid4().hex
        while patch_id in _records:
            patch_id = uuid4().hex
        record = PatchRecord(
            patch_id=patch_id, session_id=session_id, repo_name=repo_name,
            file_path=file_path, original_hash=original_hash,
            original_content=original_content, updated_content=updated_content,
            diff=diff, created_at=datetime.now(timezone.utc),
            parent_patch_id=parent_patch_id,
            operation=operation,
            testing_required=requires_testing, origin=origin,
        )
        if parent is not None and parent.status in (
            PatchStatus.PENDING_REVIEW, PatchStatus.APPROVED,
        ):
            _records[parent.patch_id] = replace(parent, status=PatchStatus.SUPERSEDED)
        _records[patch_id] = record
        return record


def create_patch_batch(proposals: list[dict]) -> tuple[PatchRecord, ...]:
    """Save a proposal batch atomically in memory, including parent transitions.

This is not a filesystem transaction: application remains per Patch ID.
"""
    with _lock:
        before = _records.copy()
        try:
            return tuple(create_patch_record(**proposal) for proposal in proposals)
        except Exception:
            _records.clear()
            _records.update(before)
            raise


def _transition(patch_id: str, target: PatchStatus) -> PatchRecord:
    allowed = {
        PatchStatus.PENDING_REVIEW: {PatchStatus.APPROVED, PatchStatus.REJECTED},
        PatchStatus.APPROVED: {PatchStatus.REJECTED, PatchStatus.APPLIED},
    }
    with _lock:
        record = get_patch(patch_id)
        if target not in allowed.get(record.status, set()):
            raise ValueError(f"Cannot transition {record.status.value} to {target.value}")
        changes = {"status": target}
        if target == PatchStatus.APPROVED:
            changes["approved_at"] = datetime.now(timezone.utc)
        elif target == PatchStatus.APPLIED:
            changes["applied_at"] = datetime.now(timezone.utc)
        updated = replace(record, **changes)
        _records[patch_id] = updated
        return updated


def approve_patch(patch_id: str) -> PatchRecord:
    """Trusted application only: call after direct human confirmation of this ID."""
    return _transition(patch_id, PatchStatus.APPROVED)


def reject_patch(patch_id: str) -> PatchRecord:
    """Reject an active proposal, retaining its contents and approval history."""
    return _transition(patch_id, PatchStatus.REJECTED)


def mark_applied(patch_id: str) -> PatchRecord:
    """Apply workflow only: call AFTER the approved contents were written.

This records a status, not a file operation or proof of a successful write.
"""
    return _transition(patch_id, PatchStatus.APPLIED)


@contextmanager
def approved_patch_for_application(patch_id: str):
    """Internal apply entry: hold lifecycle lock through validation and writing.

Prevents other threads in this process from revoking/superseding approval
mid-write. Does not lock files or authenticate human approval. The caller
must mark_applied only after a successful write; failures retain approval.
"""
    with _lock:
        record = get_patch(patch_id)
        if record.status != PatchStatus.APPROVED or record.approved_at is None:
            raise ValueError(f"Patch is not approved for application: {record.status.value}")
        yield record
