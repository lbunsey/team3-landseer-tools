"""list_patches: lists this session's patch proposals and their status. Read-only; records are process-local."""

from langchain_core.tools import tool

from . import patch_store

Status = patch_store.PatchStatus
MAX_LIMIT = 100


def _changes(diff):
    """'+3/-1': lines added/removed in a unified diff."""
    lines = diff.splitlines()
    added = sum(1 for l in lines if l.startswith("+") and not l.startswith("+++"))
    removed = sum(1 for l in lines if l.startswith("-") and not l.startswith("---"))
    return f"+{added}/-{removed}"


def _iso(moment):
    return moment.isoformat(timespec="seconds") if moment else None


@tool
def list_patches(session_id: str, repo_name: str = "", status: str = "", limit: int = 30) -> dict:
    """List this session's patch proposals (newest last) without their contents or diffs.

    session_id is filled in by the application. Optional filters: repo_name (a folder in
    repos/) and status (pending_review, approved, applied, superseded or rejected).
    limit (1-100, default 30) keeps only the most recent entries. Each entry has the full
    patch_id, repo_name, file_path, operation, status, parent_patch_id, changes ("+added/-removed"),
    created_at and applied_at. An applied patch that a later applied patch undid carries
    reverted_by. Read-only: no file is written and no approval is changed.
    """
    print(f"[DEBUG][list_patches] Listing patches: repo={repo_name or '*'} status={status or '*'}", flush=True)
    try:
        if status and status not in {s.value for s in Status}:
            raise ValueError(f"status must be one of: {', '.join(s.value for s in Status)}")
        records = patch_store.list_session_patches(session_id)
        undone = {}   # applied patch id -> id of the applied patch that restored its original contents
        for p in records:
            parent = next((r for r in records if r.patch_id == p.parent_patch_id), None)
            if parent and p.status == Status.APPLIED and parent.status == Status.APPLIED \
                    and p.updated_content == parent.original_content:
                undone[parent.patch_id] = p.patch_id
        shown = [p for p in records
                 if (not repo_name or p.repo_name == repo_name) and (not status or p.status.value == status)]
        limit = max(1, min(int(limit), MAX_LIMIT))
        entries = [{
            "patch_id": p.patch_id, "repo_name": p.repo_name, "file_path": p.file_path,
            "operation": p.operation, "status": p.status.value, "parent_patch_id": p.parent_patch_id,
            "changes": _changes(p.diff), "created_at": _iso(p.created_at), "applied_at": _iso(p.applied_at),
            **({"reverted_by": undone[p.patch_id]} if p.patch_id in undone else {}),
        } for p in shown[-limit:]]
    except (ValueError, TypeError) as exc:
        message = f"{type(exc).__name__}: {exc}"
        print(f"[ERROR][list_patches] List rejected: {message}", flush=True)
        return {"ok": False, "error": message}
    return {"ok": True, "session_id": session_id, "total_matching": len(shown),
            "shown": len(entries), "patches": entries}
