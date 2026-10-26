"""Read-only previews of single-file UTF-8 patches; no repository writes.

Callers supply the FULL proposed file and the hash of the base they inspected.
Proposed line endings follow the original file; BOM is not normalized.
Records are process-local.
CLI confirmation is required before the saved proposal can be applied.
"""

from dataclasses import dataclass
import difflib
import os

from langchain_core.tools import tool

from . import patch_store, patch_utils


@dataclass(frozen=True)
class PreparedPatch:
    repo_name: str
    file_path: str
    original_hash: str
    original_content: bytes
    updated_content: bytes
    diff: str
    operation: str = "modify"


def _preserve_line_endings(original: str, proposed: str) -> str:
    """Keep a uniform base's LF/CRLF style before diffing and saving.

Reject mixed or bare-CR bases rather than silently rewriting unrelated
lines. A base with no newline uses LF for newly introduced lines. Whether
the proposal ends with a newline is preserved.
"""
    original_lf = original.replace("\r\n", "\n")
    if "\r" in original_lf:
        raise ValueError("Bare CR line endings are not supported")
    if "\r\n" in original and "\n" in original.replace("\r\n", ""):
        raise ValueError("Mixed LF/CRLF original: normalize the file separately before previewing")
    proposed_lf = proposed.replace("\r\n", "\n")
    if "\r" in proposed_lf:
        raise ValueError("Proposed content contains unsupported bare CR line endings")
    return proposed_lf.replace("\n", "\r\n") if "\r\n" in original else proposed_lf


def prepare_patch(
    repo_name: str, file_path: str, updated_content: str, expected_hash: str,
    operation: str = "modify",
) -> PreparedPatch:
    """Validate the base and proposed text, returning an unsaved proposal.

The expected hash is SHA-256 of exact base bytes, not text normalized by a
reader. Raises on stale bases, no-op changes, binary/non-UTF-8 or oversized
contents. Valid UTF-8 alone does not imply syntactically valid Python.
"""
    if not isinstance(updated_content, str):
        raise TypeError("updated_content must be the full file as a UTF-8 string")
    if operation not in {"modify", "create"}:
        raise ValueError("operation must be modify or create")
    if operation == "create":
        if expected_hash != "":
            raise ValueError("Create requires expected_hash='' to assert that the file is absent")
    elif (
        not isinstance(expected_hash, str) or len(expected_hash) != 64
        or any(c not in "0123456789abcdef" for c in expected_hash)
    ):
        raise ValueError("expected_hash must be a lowercase SHA-256 hex digest")
    target = patch_utils.resolve_target(repo_name, file_path, allow_missing=operation == "create")
    # Bounded read also covers file growth after resolve_target's size check.
    if operation == "create":
        if os.path.lexists(target):
            raise FileExistsError("Create target already exists; use a modify proposal")
        original = b""
    else:
        with target.open("rb") as stream:
            original = stream.read(patch_utils.MAX_FILE_BYTES + 1)
    if len(original) > patch_utils.MAX_FILE_BYTES:
        raise ValueError("Original or proposed file exceeds the patch size limit")
    original_hash = patch_utils.compute_file_hash(original)
    if operation == "modify" and original_hash != expected_hash:
        raise ValueError("Stale base: file changed; inspect it again before previewing")
    original_text = original.decode("utf-8")
    if "\x00" in original_text or "\x00" in updated_content:
        raise ValueError("NUL-containing files are not supported text files")
    updated_content = _preserve_line_endings(original_text, updated_content)
    updated = updated_content.encode("utf-8")
    if len(updated) > patch_utils.MAX_FILE_BYTES:
        raise ValueError("Original or proposed file exceeds the patch size limit")
    if original == updated:
        raise ValueError("Patch must change the contents")

    relative = file_path.replace("\\", "/")
    diff_lines = difflib.unified_diff(
        original_text.splitlines(keepends=True),
        updated_content.splitlines(keepends=True),
        fromfile="/dev/null" if operation == "create" else f"a/{relative}", tofile=f"b/{relative}",
    )
    # difflib omits this marker; without it, EOF additions/removals can appear
    # concatenated or indistinguishable. Preserve CRLF inside the diff too.
    diff = "".join(
        line if line.endswith("\n") else line + "\n\\ No newline at end of file\n"
        for line in diff_lines
    )
    return PreparedPatch(
        repo_name=repo_name, file_path=relative, original_hash=original_hash,
        original_content=original, updated_content=updated, diff=diff, operation=operation,
    )


def _save_prepared(
    prepared: PreparedPatch, session_id: str, parent_patch_id: str | None,
) -> patch_store.PatchRecord:
    if parent_patch_id is not None:
        parent = patch_store.get_patch(parent_patch_id)
        if parent.original_hash != prepared.original_hash:
            raise ValueError("Parent has a different base; start a new proposal")
    return patch_store.create_patch_record(
        session_id=session_id, repo_name=prepared.repo_name,
        file_path=prepared.file_path, original_hash=prepared.original_hash,
        original_content=prepared.original_content,
        updated_content=prepared.updated_content, diff=prepared.diff,
        parent_patch_id=parent_patch_id,
        operation=prepared.operation,
    )


def restore_patch(source_patch_id: str) -> patch_store.PatchRecord:
    """Restore saved contents as a fresh child of the source; never write files.

The source remains in history and its approval is not inherited. Other
branches are not superseded automatically. This is an internal helper, not
an agent tool. A changed base requires a newly prepared proposal instead.
"""
    print(f"[DEBUG][patch_preview] Restoring proposal: {source_patch_id}", flush=True)
    source = patch_store.get_patch(source_patch_id)
    prepared = prepare_patch(
        source.repo_name, source.file_path,
        source.updated_content.decode("utf-8"),
        "" if source.operation == "create" else source.original_hash,
        operation=source.operation,
    )
    record = _save_prepared(prepared, source.session_id, source.patch_id)
    print(f"[DEBUG][patch_preview] Saved restored patch: {record.patch_id}", flush=True)
    return record


@tool
def restore_patch_proposal(source_patch_id: str) -> dict:
    """Restore a previous saved proposal as a NEW pending-review Patch ID.

No file is written, no approval is inherited. Other branches remain in
history. Rejects changed original files; inspect again for a new base.
"""
    try:
        record = restore_patch(source_patch_id)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        message = f"{type(exc).__name__}: {exc}"
        print(f"[ERROR][patch_preview] Restore rejected: {message}", flush=True)
        return {"ok": False, "error": message}
    return {
        "ok": True, "patch_id": record.patch_id, "session_id": record.session_id,
        "repo_name": record.repo_name, "file_path": record.file_path,
        "parent_patch_id": record.parent_patch_id,
        "status": record.status.value, "diff": record.diff,
        "operation": record.operation,
    }


@tool
def preview_patch(
    repo_name: str,
    file_path: str,
    updated_content: str,
    expected_hash: str,
    session_id: str,
    parent_patch_id: str | None = None,
    operation: str = "modify",
) -> dict:
    """Preview modifying or creating one UTF-8 file without writing it.

For a new file use operation='create', expected_hash=''; the parent directory
must exist. Existing paths are never overwritten by create proposals.

Supply the FULL proposed file and SHA-256 of the exact inspected base bytes.
Proposed LF/CRLF endings are converted to the original file's uniform style
before diff generation and storage. Mixed-ending originals are rejected.
Use the same session_id and parent_patch_id for a revision. Every successful
call saves a new pending-review ID and returns the full, untruncated diff.
No approval or application occurs. Errors do not save or supersede records.
"""
    print(f"[DEBUG][patch_preview] Preparing proposal: {repo_name}/{file_path}", flush=True)
    try:
        prepared = prepare_patch(repo_name, file_path, updated_content, expected_hash, operation)
        record = _save_prepared(prepared, session_id, parent_patch_id)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        message = f"{type(exc).__name__}: {exc}"
        print(f"[ERROR][patch_preview] Preview rejected: {message}", flush=True)
        return {"ok": False, "error": message}
    print(f"[DEBUG][patch_preview] Saved patch: {record.patch_id}", flush=True)
    return {
        "ok": True, "patch_id": record.patch_id,
        "session_id": record.session_id, "repo_name": record.repo_name,
        "file_path": record.file_path, "original_hash": record.original_hash,
        "parent_patch_id": record.parent_patch_id, "status": record.status.value,
        "diff": record.diff,
        "operation": record.operation,
    }
