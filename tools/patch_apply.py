"""Apply exact saved bytes after trusted application approval.

Not registered in tools.TOOLS yet. Human Y/N confirmation belongs to the CLI.
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
    """File replacement succeeded but recording applied status failed."""


def atomic_write(target: Path, content: bytes, *, before_replace: Callable[[], None]) -> None:
    """Stage/fsync bytes beside target, revalidate, then replace atomically.

On failure before replacement, the original is untouched and temporary
files are removed. Does not promise durability across power failure.
"""
    mode = stat.S_IMODE(target.stat().st_mode)
    temporary = None
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
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def apply_saved_patch(patch_id: str) -> patch_store.PatchRecord:
    """Apply an approved saved version, never accepting regenerated contents."""
    print(f"[DEBUG][patch_apply] Checking approval: {patch_id}", flush=True)
    with patch_store.approved_patch_for_application(patch_id) as record:
        if patch_utils.compute_file_hash(record.original_content) != record.original_hash:
            raise ValueError("Saved original contents do not match the recorded hash")
        if len(record.updated_content) > patch_utils.MAX_FILE_BYTES:
            raise ValueError("Saved proposed contents exceed the patch size limit")
        updated_text = record.updated_content.decode("utf-8")
        if "\x00" in updated_text:
            raise ValueError("Saved proposed contents contain NUL characters")
        target = patch_utils.resolve_target(record.repo_name, record.file_path)

        def validate_current() -> None:
            current_target = patch_utils.resolve_target(record.repo_name, record.file_path)
            if current_target != target:
                raise ValueError("Target path changed after validation")
            with current_target.open("rb") as stream:
                current = stream.read(patch_utils.MAX_FILE_BYTES + 1)
            if patch_utils.compute_file_hash(current) != record.original_hash:
                raise ValueError("Stale patch: file changed; create and review a new preview")

        print(f"[DEBUG][patch_apply] Checking current file hash: {patch_id}", flush=True)
        validate_current()
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
    }
