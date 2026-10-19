"""In-memory patch records. Nothing here reads or writes repository files.

Records disappear when the process exits. Approval functions are trusted
application entry points, never agent tools; this module cannot authenticate
human approval. File validation and application belong to later modules.
"""

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
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


_records: dict[str, PatchRecord] = {}
_lock = RLock()


def get_patch(patch_id: str) -> PatchRecord:
    """Return an immutable snapshot; raise KeyError for an unknown ID."""
    with _lock:
        return _records[patch_id]


def list_session_patches(session_id: str) -> tuple[PatchRecord, ...]:
    """Return immutable snapshots in creation order."""
    with _lock:
        return tuple(p for p in _records.values() if p.session_id == session_id)


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
    if original_content == updated_content:
        raise ValueError("Patch must change the contents")

    with _lock:
        parent = get_patch(parent_patch_id) if parent_patch_id is not None else None
        if parent is not None and (
            parent.session_id, parent.repo_name, parent.file_path
        ) != (session_id, repo_name, file_path):
            raise ValueError("Parent must belong to the same session and target")
        patch_id = uuid4().hex
        while patch_id in _records:
            patch_id = uuid4().hex
        record = PatchRecord(
            patch_id=patch_id, session_id=session_id, repo_name=repo_name,
            file_path=file_path, original_hash=original_hash,
            original_content=original_content, updated_content=updated_content,
            diff=diff, created_at=datetime.now(timezone.utc),
            parent_patch_id=parent_patch_id,
        )
        if parent is not None and parent.status in (
            PatchStatus.PENDING_REVIEW, PatchStatus.APPROVED,
        ):
            _records[parent.patch_id] = replace(parent, status=PatchStatus.SUPERSEDED)
        _records[patch_id] = record
        return record


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
