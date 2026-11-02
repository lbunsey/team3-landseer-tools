"""Dataset-integration task planning, patch batches and honest manual status.

The model analyzes repository-specific semantics and supplies proposed code.
This tool validates/stores that proposal; it neither invents a loader nor
runs code, downloads data, approves patches or applies them.
"""

from dataclasses import dataclass, replace
from threading import RLock
from typing import Literal
from uuid import uuid4

from langchain_core.tools import tool
from pydantic import BaseModel

from . import patch_store
from .inspect_repo import get_repo_root, inspect_repo
from .patch_preview import prepare_patch


class DatasetChange(BaseModel):
    file_path: str
    updated_content: str
    reason: str
    operation: Literal["modify", "create"] = "modify"
    expected_hash: str = ""
    parent_patch_id: str | None = None


@dataclass(frozen=True)
class DatasetTask:
    task_id: str
    session_id: str
    repo_name: str
    dataset_name: str
    task_description: str
    dataset_source: str
    integration_summary: str = ""
    validation_steps: tuple[str, ...] = ()
    patch_batches: tuple[tuple[str, ...], ...] = ()
    open_questions: tuple[str, ...] = ()


_tasks: dict[str, DatasetTask] = {}
_lock = RLock()


def _result(task: DatasetTask) -> dict:
    latest = task.patch_batches[-1] if task.patch_batches else ()
    records = [patch_store.get_patch(patch_id) for patch_id in latest]
    if task.open_questions:
        status = "needs_clarification"
    elif not records:
        status = "planning"
    elif all(p.status == patch_store.PatchStatus.APPLIED for p in records):
        status = "code_applied_unverified"
    elif any(p.status == patch_store.PatchStatus.APPLIED for p in records):
        status = "partially_applied_unverified"
    elif any(p.status in {patch_store.PatchStatus.REJECTED, patch_store.PatchStatus.SUPERSEDED} for p in records):
        status = "needs_revision"
    else:
        status = "pending_review"
    return {
        "ok": True, "task_id": task.task_id, "session_id": task.session_id,
        "repo_name": task.repo_name, "dataset_name": task.dataset_name,
        "task_description": task.task_description, "dataset_source": task.dataset_source,
        "status": status, "integration_summary": task.integration_summary,
        "validation_status": "not_run_user_verification_required",
        "validation_steps": list(task.validation_steps),
        "open_questions": list(task.open_questions),
        "patch_batches": [list(batch) for batch in task.patch_batches],
        "batch_revision": len(task.patch_batches),
        "patches": [{
            "patch_id": p.patch_id, "file_path": p.file_path,
            "operation": p.operation, "status": p.status.value, "diff": p.diff,
        } for p in records],
        "note": "This is the complete latest proposal. Review and apply pending IDs separately through CLI Y/N; do not reapply retained applied IDs. No code or data was executed/downloaded by this tool. Multi-file application is not atomic.",
    }


@tool
def add_dataset(
    repo_name: str,
    dataset_name: str,
    session_id: str,
    action: Literal["plan", "preview", "status"] = "plan",
    task_id: str = "",
    task_description: str = "",
    dataset_source: str = "",
    integration_summary: str = "",
    changes: list[DatasetChange] | None = None,
    validation_steps: list[str] | None = None,
    open_questions: list[str] | None = None,
) -> dict:
    """Plan a dataset integration, preview code changes, or report task status.

plan: start a task using the user's high-level goal; returns repository
overview and an analysis checklist. Do NOT repeatedly plan the same task.
Use inspection tools to understand existing loaders, models and metrics.

preview: after discussing the plan, supply task_id, integration_summary,
manual validation_steps and complete per-file changes with reasons. Modify
requires the inspected base hash; create requires expected_hash='' and an
existing parent directory. Nonempty open_questions block patch creation:
ask plain-language research questions, recommend technical choices yourself.
Questions remain pending across calls; pass open_questions=[] after the user
resolves them and describe the agreed choices in integration_summary.
Submit only files being revised or added. Omitted files retain their current
Patch IDs, including already-applied files. Identical active proposals reuse
their IDs and approval. A new batch is a complete task snapshot, not just this
call's changed files. No content changes means no new batch. Removal is not
supported. All changes are prepared before any records are saved.

status: retrieve the latest patch states, including partially applied work.
No LLM, shell, downloads, approval or execution is performed by this tool.
Code application always remains UNVERIFIED until the user tests it; this
tool cannot certify dataset compatibility or scientific correctness.
"""
    print(f"[DEBUG][add_dataset] {action}: {repo_name}/{dataset_name}", flush=True)
    try:
        if not session_id.strip() or not dataset_name.strip():
            raise ValueError("session_id and dataset_name must be non-empty")
        with _lock:
            if action == "plan":
                if task_id or changes:
                    raise ValueError("plan starts a task without task_id or changes")
                if not task_description.strip():
                    raise ValueError("Provide the user's high-level task_description")
                get_repo_root(repo_name)
                overview = inspect_repo.invoke({"repo_name": repo_name})
                if overview.startswith("Error"):
                    raise ValueError(overview)
                task = DatasetTask(
                    task_id=uuid4().hex, session_id=session_id, repo_name=repo_name,
                    dataset_name=dataset_name, task_description=task_description,
                    dataset_source=dataset_source,
                    open_questions=tuple(open_questions or ()),
                )
                _tasks[task.task_id] = task
                result = _result(task)
                result.update({
                    "repository_overview": overview,
                    "analysis_checklist": [
                        "Inspect complete files before edits; identify loader, preprocessing, split, labels, model output, loss, CLI and evaluation contracts.",
                        "Check source/format and whether the dataset already exists. Explain target and group attributes; never invent them or silently change the research task.",
                        "Recommend technical defaults in plain language. Resolve research-scope questions before generating patches.",
                        "Preserve existing datasets and defaults. Separate optional new branches from unrelated refactoring.",
                        "State downloads/dependencies and prepare manual commands, expected results and regression checks; do not execute them.",
                    ],
                })
                return result
            task = _tasks[task_id]
            if (task.session_id, task.repo_name, task.dataset_name) != (session_id, repo_name, dataset_name):
                raise ValueError("Task does not belong to this session/repository/dataset")
            if action == "status":
                return _result(task)
            if action != "preview":
                raise ValueError("Unknown action")
            questions = tuple(open_questions) if open_questions is not None else task.open_questions
            if questions:
                task = replace(task, open_questions=questions)
                _tasks[task.task_id] = task
                return _result(task)
            if not integration_summary.strip():
                raise ValueError("integration_summary must describe the agreed integration plan")
            if not validation_steps or any(not step.strip() for step in validation_steps):
                raise ValueError("Provide non-empty manual validation_steps")
            if not changes or len(changes) > 20:
                raise ValueError("Provide between 1 and 20 file changes")
            prepared = []
            previous = {
                patch_store.get_patch(pid).file_path.casefold(): patch_store.get_patch(pid)
                for pid in (task.patch_batches[-1] if task.patch_batches else ())
            }
            # Preserve order and all unchanged/applied proposals. Replace only
            # the slot for a revised file; append genuinely new target files.
            complete_batch = {key: record.patch_id for key, record in previous.items()}
            reasons = {}
            paths = set()
            for change in changes:
                if not change.reason.strip():
                    raise ValueError("Each change needs a reason")
                proposal = prepare_patch(repo_name, change.file_path, change.updated_content,
                                         change.expected_hash, change.operation)
                # Windows aliases must not create two competing patches in a batch.
                key = proposal.file_path.casefold()
                if key in paths:
                    raise ValueError("Duplicate target path in change batch")
                paths.add(key)
                reasons[key] = change.reason
                parent_id = change.parent_patch_id
                old = previous.get(key)
                if old and old.status in {
                    patch_store.PatchStatus.PENDING_REVIEW, patch_store.PatchStatus.APPROVED,
                }:
                    if parent_id is not None and parent_id != old.patch_id:
                        raise ValueError("Revise the current proposal for this file, not an older parent")
                    if (
                        old.operation == proposal.operation
                        and old.original_hash == proposal.original_hash
                        and old.updated_content == proposal.updated_content
                    ):
                        continue
                if parent_id is None and old and old.status in {
                    patch_store.PatchStatus.PENDING_REVIEW, patch_store.PatchStatus.APPROVED,
                }:
                    parent_id = old.patch_id
                if parent_id is not None:
                    parent = patch_store.get_patch(parent_id)
                    if parent.original_hash != proposal.original_hash:
                        raise ValueError("Parent base changed; start a new proposal from inspected contents")
                prepared.append(dict(
                    session_id=session_id, repo_name=repo_name, file_path=proposal.file_path,
                    original_hash=proposal.original_hash, original_content=proposal.original_content,
                    updated_content=proposal.updated_content, diff=proposal.diff,
                    parent_patch_id=parent_id, operation=proposal.operation,
                ))
            records = patch_store.create_patch_batch(prepared)
            for record in records:
                complete_batch[record.file_path.casefold()] = record.patch_id
            batches = task.patch_batches
            if records:
                batches += (tuple(complete_batch.values()),)
            task = replace(
                task, integration_summary=integration_summary,
                validation_steps=tuple(validation_steps),
                open_questions=(),
                patch_batches=batches,
            )
            _tasks[task.task_id] = task
            result = _result(task)
            result["batch_created"] = bool(records)
            result["new_patch_ids"] = [record.patch_id for record in records]
            for item in result["patches"]:
                reason = reasons.get(item["file_path"].casefold())
                if reason is not None:
                    item["reason"] = reason
            return result
    except (ValueError, TypeError, KeyError, OSError) as exc:
        message = f"{type(exc).__name__}: {exc}"
        print(f"[ERROR][add_dataset] {message}", flush=True)
        return {"ok": False, "error": message}
