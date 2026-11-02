"""Apply exact saved bytes after trusted application approval.

Human Y/N confirmation belongs to the CLI, not the model.
Checks reduce, but do not eliminate, external hash-check/replace races.
Atomic replacement preserves basic mode bits, not arbitrary ACLs/metadata;
it is not a transaction with the volatile patch store or crash recovery.
"""

import os
from pathlib import Path
import stat
import tempfile
from typing import Callable

from langchain_core.tools import tool

from . import patch_store, patch_utils


class ApplyStatusError(RuntimeError):
    """Writing succeeded but a subsequent cleanup or status operation failed."""


def atomic_write(target: Path, content: bytes, *, before_replace: Callable[[], None], create: bool = False) -> None:
    """Stage/fsync bytes beside target, revalidate, then replace atomically.

On failure before replacement, the original is untouched and temporary
files are removed. Does not promise durability across power failure.
"""
    mode = 0o600 if create else stat.S_IMODE(target.stat().st_mode)
    temporary = None
    published = False
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=target.parent, prefix=".patch-", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        before_replace()
        if create:
            # Same-directory hard link publishes complete staged bytes and
            # atomically fails if another process created the destination.
            # No overwrite/partial-write fallback on unsupported filesystems.
            os.link(temporary, target)
        else:
            os.replace(temporary, target)
        published = True
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as exc:
                if published:
                    raise ApplyStatusError(
                        "File was written but staging cleanup failed; inspect file and record before retrying"
                    ) from exc
                raise


def apply_saved_patch(patch_id: str) -> patch_store.PatchRecord:
    """Apply an approved saved version, never accepting regenerated contents."""
    print(f"[DEBUG][patch_apply] Checking approval: {patch_id}", flush=True)
    with patch_store.approved_patch_for_application(patch_id) as record:
        if record.operation not in {"modify", "create"}:
            raise ValueError("Unknown patch operation")
        if patch_utils.compute_file_hash(record.original_content) != record.original_hash:
            raise ValueError("Saved original contents do not match the recorded hash")
        if len(record.updated_content) > patch_utils.MAX_FILE_BYTES:
            raise ValueError("Saved proposed contents exceed the patch size limit")
        updated_text = record.updated_content.decode("utf-8")
        if "\x00" in updated_text:
            raise ValueError("Saved proposed contents contain NUL characters")
        creating = record.operation == "create"
        target = patch_utils.resolve_target(record.repo_name, record.file_path, allow_missing=creating)

        def validate_current() -> None:
            current_target = patch_utils.resolve_target(record.repo_name, record.file_path, allow_missing=creating)
            if current_target != target:
                raise ValueError("Target path changed after validation")
            if creating:
                if os.path.lexists(current_target):
                    raise FileExistsError("Create target now exists; no file was overwritten")
                return
            with current_target.open("rb") as stream:
                current = stream.read(patch_utils.MAX_FILE_BYTES + 1)
            if patch_utils.compute_file_hash(current) != record.original_hash:
                raise ValueError("Stale patch: file changed; create and review a new preview")

        print(f"[DEBUG][patch_apply] Checking current file hash: {patch_id}", flush=True)
        validate_current()
        if creating:
            atomic_write(target, record.updated_content, before_replace=validate_current, create=True)
        else:
            atomic_write(target, record.updated_content, before_replace=validate_current)
        try:
            applied = patch_store.mark_applied(patch_id)
        except Exception as exc:
            raise ApplyStatusError(
                "File was written but status update failed; inspect file and record before retrying"
            ) from exc
    print(f"[DEBUG][patch_apply] Applied patch: {patch_id}", flush=True)
    return applied


@tool
def apply_patch(patch_id: str) -> dict:
    """Apply only a specifically approved saved Patch ID.

Does not accept approval flags or new contents and cannot grant approval.
Rejects pending, superseded, rejected, applied and stale proposals.
"""
    try:
        record = apply_saved_patch(patch_id)
    except ApplyStatusError as exc:
        print(f"[ERROR][patch_apply] {exc}", flush=True)
        return {"ok": False, "patch_id": patch_id, "file_written": True, "error": str(exc)}
    except (ValueError, TypeError, KeyError, OSError) as exc:
        message = f"{type(exc).__name__}: {exc}"
        print(f"[ERROR][patch_apply] Apply rejected: {message}", flush=True)
        return {"ok": False, "patch_id": patch_id, "error": message}
    return {
        "ok": True, "patch_id": record.patch_id, "repo_name": record.repo_name,
        "file_path": record.file_path, "status": record.status.value,
        "updated_hash": patch_utils.compute_file_hash(record.updated_content),
        "operation": record.operation,
    }
