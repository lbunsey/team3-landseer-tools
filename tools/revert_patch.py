"""revert_patch: proposes undoing an applied patch as a NEW pending-review proposal.
It writes no repository file and grants no approval: applying the new Patch ID still needs the
CLI Y/N confirmation, like any other patch. Records are process-local."""

import difflib

from langchain_core.tools import tool

from . import patch_store, patch_utils

Status = patch_store.PatchStatus


def _diff(original, updated, path):
    """Unified diff text, written the same way as patch_preview so the CLI shows a familiar diff."""
    lines = difflib.unified_diff(original.splitlines(keepends=True), updated.splitlines(keepends=True),
                                 fromfile=f"a/{path}", tofile=f"b/{path}")
    return "".join(l if l.endswith("\n") else l + "\n\\ No newline at end of file\n" for l in lines)


def revert_patch_proposal(patch_id):
    """Save a pending-review proposal that restores the contents an applied patch replaced."""
    try:
        source = patch_store.get_patch(patch_id)
    except KeyError:
        raise ValueError("Unknown Patch ID. Use list_patches to see this session's IDs.") from None
    if source.status != Status.APPLIED:
        raise ValueError(f"Only applied patches can be reverted; this one is {source.status.value}. "
                         "It never changed the file, so there is nothing to undo.")
    if source.operation == "create":
        raise ValueError("This patch created a file. The patch workflow cannot delete files, "
                         "so delete it by hand.")

    target = patch_utils.resolve_target(source.repo_name, source.file_path)
    with target.open("rb") as stream:
        current = stream.read(patch_utils.MAX_FILE_BYTES + 1)
    if len(current) > patch_utils.MAX_FILE_BYTES:
        raise ValueError("File exceeds the patch size limit")
    current_hash = patch_utils.compute_file_hash(current)

    if current_hash != patch_utils.compute_file_hash(source.updated_content):
        if current_hash == source.original_hash:
            raise ValueError("The file already has its pre-patch contents; nothing to revert.")
        later = [p.patch_id for p in patch_store.list_session_patches(source.session_id)
                 if (p.repo_name, p.file_path, p.status) == (source.repo_name, source.file_path, Status.APPLIED)
                 and p.applied_at > source.applied_at]
        raise ValueError(
            "The file no longer matches what this patch wrote. "
            + (f"Later applied patches on it: {', '.join(later)}. Revert those first, newest first."
               if later else "It was changed outside the patch workflow; inspect it again and preview a new patch."))

    for p in patch_store.list_session_patches(source.session_id):   # asking twice reuses the open proposal
        if p.parent_patch_id == source.patch_id and p.status in (Status.PENDING_REVIEW, Status.APPROVED) \
                and p.original_hash == current_hash and p.updated_content == source.original_content:
            return p, True

    diff = _diff(current.decode("utf-8"), source.original_content.decode("utf-8"), source.file_path)
    record = patch_store.create_patch_record(
        session_id=source.session_id, repo_name=source.repo_name, file_path=source.file_path,
        original_hash=current_hash, original_content=current, updated_content=source.original_content,
        diff=diff, parent_patch_id=source.patch_id, operation="modify")
    return record, False


@tool
def revert_patch(patch_id: str) -> dict:
    """Propose undoing an APPLIED patch as a NEW pending-review Patch ID.

    No file is written and no approval is inherited: the user must ask for the new ID to be
    applied and confirm Y/N in the CLI. Refuses if the patch is not applied, if it created a
    file (deleting is not supported), or if the file changed since the patch was applied
    (for example by a later patch: revert the newest first). Asking twice returns the same open
    proposal. Show the user the returned diff.
    """
    print(f"[DEBUG][revert_patch] Proposing revert of: {patch_id}", flush=True)
    try:
        record, reused = revert_patch_proposal(patch_id)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        message = f"{type(exc).__name__}: {exc}"
        print(f"[ERROR][revert_patch] Revert rejected: {message}", flush=True)
        return {"ok": False, "error": message}
    print(f"[DEBUG][revert_patch] Saved revert proposal: {record.patch_id}", flush=True)
    return {
        "ok": True, "patch_id": record.patch_id, "reverts_patch_id": patch_id, "reused_open_proposal": reused,
        "session_id": record.session_id, "repo_name": record.repo_name, "file_path": record.file_path,
        "parent_patch_id": record.parent_patch_id, "status": record.status.value,
        "diff": record.diff, "operation": record.operation,
        "note": "Nothing was written. Applying this Patch ID needs the user's request and CLI Y/N confirmation.",
    }
