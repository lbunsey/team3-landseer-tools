"""Terminal review of pending patches: the human Y/N step of the patch workflow.

Trusted application code, never an agent tool: approval only comes from a
person answering here. All pending patches are reviewed together because one
tool call (e.g. add_model) can propose several files that only work together.
"""

from . import patch_apply, patch_store


def review_pending_patches(ask=input) -> str:
    """Show every pending diff, ask once, then apply all or reject all.

Applies in creation order and stops at the first failure, rejecting the
rest, so a half-applied change set is reported rather than silently left.
Returns a one-line summary ('' if nothing was pending).
"""
    pending = patch_store.list_pending_patches()
    if not pending:
        return ""

    print(f"\n{len(pending)} proposed change(s) waiting for your review:\n", flush=True)
    for record in pending:
        kind = "new file" if record.creates_file else "edit"
        print(f"=== {record.repo_name}/{record.file_path} ({kind}, patch {record.patch_id[:8]})")
        print(record.diff, flush=True)

    answer = ask(f"Apply all {len(pending)} change(s)? [y/N]: ").strip().lower()
    results = []
    if answer in ("y", "yes"):
        for i, record in enumerate(pending):
            label = f"{record.repo_name}/{record.file_path}"
            try:
                patch_store.approve_patch(record.patch_id)
                patch_apply.apply_saved_patch(record.patch_id)
                results.append(f"applied {label}")
            except (ValueError, TypeError, KeyError, OSError, patch_apply.ApplyStatusError) as exc:
                print(f"[ERROR][patch_review] {label}: {type(exc).__name__}: {exc}", flush=True)
                results.append(f"FAILED {label} ({exc})")
                for rest in pending[i:]:
                    if patch_store.get_patch(rest.patch_id).status in (
                        patch_store.PatchStatus.PENDING_REVIEW, patch_store.PatchStatus.APPROVED,
                    ):
                        patch_store.reject_patch(rest.patch_id)
                results += [f"not applied {r.repo_name}/{r.file_path}" for r in pending[i + 1:]]
                break
    else:
        for record in pending:
            patch_store.reject_patch(record.patch_id)
            results.append(f"rejected {record.repo_name}/{record.file_path}")

    summary = "Patch review: " + "; ".join(results)
    print(summary + "\n", flush=True)
    return summary
