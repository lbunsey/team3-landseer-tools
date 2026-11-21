"""Dataset-integration task planning, patch batches and honest manual status.

The model analyzes repository-specific semantics and supplies proposed code.
This tool validates/stores proposals and can run explicitly approved smoke checks
in temporary source copies. It never approves or applies repository patches.
"""

from dataclasses import dataclass, replace
import re
from datetime import datetime, timezone
from threading import RLock
from typing import Literal
from uuid import uuid4

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from . import patch_store
from .inspect_repo import get_repo_root, inspect_repo
from .patch_preview import prepare_patch
from .patch_utils import resolve_target, compute_file_hash
from .dataset_testing import run_checks, invalidate, resolve_profile, profile_description, inspect_local_data, IntegrationContract, validate_contract, contract_requirements, inspect_contract_parameters
from .dataset_diagnostics import save_rejected_candidate, save_rejected_validation


ANALYSIS_AREAS = (
    "data_source", "sample_contract", "transforms", "internal_augmentations",
    "splits", "model_and_loss", "training_entry", "evaluation",
)


class CodeEvidence(BaseModel):
    file_path: str
    line: int | None = None


class IntegrationAnalysis(BaseModel):
    area: Literal["data_source", "sample_contract", "transforms", "internal_augmentations",
                  "splits", "model_and_loss", "training_entry", "evaluation"]
    findings: str = ''
    evidence: list[CodeEvidence] = Field(default_factory=list)
    decision: Literal["change", "keep", "not_applicable"] | None = None
    rationale: str = ''
    planned_changes: str = ''


def _validate_analysis_item(repo_name, item):
    if not all(text.strip() for text in (item.findings, item.rationale, item.planned_changes)):
        raise ValueError(f"{item.area}: findings, rationale and planned_changes must be non-empty")
    if item.decision is None:
        raise ValueError(f'{item.area}: provide decision (change/keep/not_applicable)')
    if not item.evidence:
        raise ValueError(f"{item.area}: provide repository code evidence, including for keep/not_applicable")
    for evidence in item.evidence:
        if evidence.line is None:
            raise ValueError(f'{item.area}: evidence for {evidence.file_path} needs a source line; draft retained')
        target = resolve_target(repo_name, evidence.file_path)
        with target.open('r', encoding='utf-8-sig') as stream:
            lines = stream.read().splitlines()
        if evidence.line < 1 or evidence.line > len(lines):
            raise ValueError(f"{item.area}: evidence line is outside {evidence.file_path}")


def _analysis_progress(task):
    hints = {
        'splits': 'Find official train/test or split arguments in loader calls, or explicit split code. No separate split helper is required.',
        'internal_augmentations': 'Read the actual wrapper and augmentation helpers; cite their real file/line locations.',
    }
    items = {item.area: item for item in task.integration_analysis}
    completed, pending, issues = [], [], []
    for area in ANALYSIS_AREAS:
        try:
            if area not in items:
                raise ValueError(f'{area}: not yet recorded')
            _validate_analysis_item(task.repo_name, items[area])
            completed.append(area)
        except (ValueError, OSError, UnicodeError) as exc:
            pending.append(area)
            issues.append(dict(area=area, reason=str(exc),
                               next_step=hints.get(area, 'Complete this analysis item with real source evidence; do not invent findings or line numbers.')))
    return dict(analysis_complete=not pending and bool(task.integration_summary.strip()),
                completed_analysis_areas=completed, pending_analysis_areas=pending,
                analysis_issues=issues,
                summary_required=not bool(task.integration_summary.strip()))


VALIDATION_STAGES = ("data_loading", "transformation", "dataloader", "pipeline", "regression")


class ValidationCheck(BaseModel):
    stage: Literal["data_loading", "transformation", "dataloader", "pipeline", "regression"]
    command: str
    expected_result: str


def _validation_result(task, records):
    latest = task.patch_batches[-1] if task.patch_batches else ()
    current = {}
    for report in task.validation_reports:
        if report["plan_id"] == task.validation_plan_id and tuple(report["patch_ids"]) == latest:
            current[report["stage"]] = report
    files_current = bool(records) and all(p.status == patch_store.PatchStatus.APPLIED for p in records)
    if files_current and current:
        try:
            files_current = all(
                compute_file_hash(resolve_target(p.repo_name, p.file_path).read_bytes())
                == compute_file_hash(p.updated_content) for p in records
            )
        except (ValueError, OSError):
            files_current = False
    stages = []
    for stage in VALIDATION_STAGES:
        check = next((c for c in task.validation_plan if c.stage == stage), None)
        report = current.get(stage)
        state = report["outcome"] if report and files_current else "not_run"
        if not report or not files_current:
            if any(r["stage"] == stage for r in task.validation_reports):
                state = "stale"
        stages.append({"stage": stage, "status": state,
                       "command": check.command if check else "",
                       "expected_result": check.expected_result if check else "",
                       "user_report": report["user_report"] if report else ""})
    states = {item["status"] for item in stages}
    if "user_reported_failed" in states:
        status = "user_reported_failed"
    elif states == {"user_reported_passed"}:
        status = "user_reported_passed"
    elif states == {"not_run"}:
        status = "not_run_user_verification_required"
    elif "stale" in states:
        status = "stale_user_verification_required"
    else:
        status = "incomplete_user_verification_required"
    return status, stages


class DatasetChange(BaseModel):
    file_path: str
    updated_content: str
    reason: str
    operation: Literal["modify", "create"] = "modify"
    expected_hash: str = Field(default="", description="For modify in the official agent CLI, omit this field: runtime binds the hash of this conversation's complete inspect_repo read. Direct tool calls must supply the exact inspected SHA-256. Never invent a hash. Create requires an empty string.")
    parent_patch_id: str | None = Field(
        default=None,
        description="Normally omit: the current file proposal is selected automatically. This is a Patch ID, never a Task ID.",
    )


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
    integration_analysis: tuple[IntegrationAnalysis, ...] = ()
    validation_plan: tuple[ValidationCheck, ...] = ()
    validation_plan_id: str = ""
    validation_reports: tuple[dict, ...] = ()
    automatic_test_history: tuple[dict, ...] = ()
    integration_contract: dict | None = None
    preflight_history: tuple[dict, ...] = ()
    preflight_pending: bool = False
    preflight_checked: bool = False
    rejected_diagnostics: tuple[dict, ...] = ()
    static_reviews: tuple[dict, ...] = ()


_tasks: dict[str, DatasetTask] = {}
_lock = RLock()
_test_failures: dict[tuple[str, str], int] = {}
MAX_TEST_FAILURES = 4  # Initial failure plus up to three automatic repair attempts.


def record_configuration_rejection(session_id, task_id, repo_name, contract, error, guidance, fields=()):
    """Internal workflow evidence hook; never registered as an agent tool."""
    with _lock:
        task = _tasks[task_id]
        if (task.session_id, task.repo_name.casefold()) != (session_id, repo_name.casefold()):
            raise ValueError('Configuration diagnostic does not belong to this conversation/repository')
        evidence = save_rejected_validation(session_id, task_id, repo_name, contract, error, list(fields), guidance)
        _tasks[task_id] = replace(task, rejected_diagnostics=task.rejected_diagnostics + (evidence,))
        return evidence


def runtime_test_snapshot(session_id):
    """Trusted CLI bookkeeping; never a model tool or approval mechanism."""
    with _lock:
        return {task.task_id: dict(automatic=len(task.automatic_test_history), preflight=len(task.preflight_history))
                for task in _tasks.values() if task.session_id == session_id}


def prepare_natural_dataset_request(text, session_id, context_repo=None):
    """Route explicit dataset requests without requiring tool names.

    Ambiguous or unsupported wording is left to the conversational agent.
    This routing is intentionally conservative, not universal intent recognition.
    """
    if re.search(r"(?i)(?:do not|don't)\s+(?:add|modify|integrate)|(?:\u4e0d\u8981|\u5148\u5225)\s*(?:\u65b0\u589e|\u52a0\u5165|\u4fee\u6539|\u6574\u5408|\u652f\u63f4)", text):
        return None
    if not re.search(r"(?i)add|support|integrate|enable|extend|\u65b0\u589e|\u52a0\u5165|\u652f\u63f4|\u6574\u5408", text):
        return None
    if re.search(r"(?i)explain|how to|\u89e3\u91cb|\u8aaa\u660e|\u5982\u4f55|\u600e\u9ebc", text) and not re.search(r"(?i)please|want|\u5e6b\u6211|\u6211\u60f3|\u6211\u8981", text):
        return None
    if re.search(r"(?i)^(?:does|can|is)\b|\u662f\u5426|\u6709\u6c92\u6709", text) and not re.search(r"(?i)please|want|\u5e6b\u6211|\u6211\u60f3|\u6211\u8981", text):
        return None
    if re.search(r"[?\uff1f]", text) and not re.search(r"(?i)please|want|need|\u5e6b\u6211|\u6211\u60f3|\u6211\u8981|\u8acb", text):
        return None
    target_text = re.split(r"(?i)\b(?:preserve|keep|retain)\b|\u4fdd\u7559|\u7dad\u6301", text)[0]
    known = re.findall(r"(?i)(?<![a-z0-9])(?:fashion[-_ ]?mnist|cifar[-_ ]?100|cifar[-_ ]?10|svhn|mnist|stl[-_ ]?10|celeba)(?![a-z0-9])", target_text)
    if len(set(n.lower() for n in known)) == 1:
        dataset_name = re.sub(r"[-_ ]", "", known[0]).lower()
    else:
        custom = re.search(r"(?i)\bdataset\s+[\"'`]?([a-z][a-z0-9_-]+)", target_text)
        if (not custom or custom.group(1).lower() in {"to", "integration", "support", "and", "model"}) and re.search(r"(?i)\bdataset\b|\u8cc7\u6599\u96c6", target_text):
            custom = re.search(r"(?i)(?:\bsupport|\badd|\bintegrate|\u652f\u63f4|\u52a0\u5165|\u65b0\u589e)\s+[\"'`]?([a-z][a-z0-9_-]+)", target_text)
        if not custom or custom.group(1).lower() in {"to", "integration", "support", "and", "model"}:
            return None
        dataset_name = custom.group(1)
    from . import patch_utils
    repos = patch_utils.REPOS_DIR
    matches = [p.name for p in repos.iterdir() if p.is_dir() and not p.is_symlink()
               and re.search(r"(?i)(?<![a-z0-9_])" + re.escape(p.name) + r"(?![a-z0-9_])", text)] if repos.is_dir() else []
    if not matches and context_repo:
        matches = [p.name for p in repos.iterdir() if p.is_dir() and not p.is_symlink()
                   and p.name.casefold() == context_repo.casefold()] if repos.is_dir() else []
    if len(matches) != 1:
        return None
    repo_name = matches[0]
    plan = add_dataset.invoke(dict(repo_name=repo_name, dataset_name=dataset_name, session_id=session_id,
                                  task_description=text, action="plan"))
    if not plan.get("ok"):
        return {"task": plan}
    return {"task": plan, "profile": None, "local_data": None}



def runtime_task_summaries(session_id):
    """Authoritative store-derived CLI summaries, separate from model prose."""
    with _lock:
        summaries = []
        for task in _tasks.values():
            if task.session_id != session_id:
                continue
            result = _result(task)
            summaries.append({
                "task_id": task.task_id, "repo_name": task.repo_name,
                "dataset_name": task.dataset_name, "batch_revision": result["batch_revision"],
                "status": result["status"], "automatic_test_status": result["automatic_test_status"],
                "preflight_status": result['preflight_status'],
                "analysis_complete": result['analysis_complete'],
                "completed_analysis_areas": result['completed_analysis_areas'],
                "pending_analysis_areas": result['pending_analysis_areas'],
                "summary_required": result['summary_required'],
                "open_questions": list(task.open_questions),
                "latest_cpu_result": task.automatic_test_history[-1] if task.automatic_test_history else None,
                "saved_patches": len(result["patches"]),
                "patch_ids": [p['patch_id'] for p in result['patches']],
                "static_review": result['static_review'],
                "applied_records": sum(p["status"] == "applied" for p in result["patches"]),
            })
        return summaries


def runtime_repair_request(session_id, before):
    """Continue a failed check from this turn, but never a blocked/cancelled run."""
    with _lock:
        for task in reversed(tuple(_tasks.values())):
            if task.session_id != session_id or task.open_questions:
                continue
            seen = before.get(task.task_id, dict(automatic=0, preflight=0))
            if task.preflight_pending and len(task.preflight_history) > seen['preflight']:
                if _test_failures.get((session_id, task.repo_name.casefold()), 0) >= MAX_TEST_FAILURES:
                    continue
                failure = task.preflight_history[-1]
                return (
                    f"[Runtime dataset repair] Task {task.task_id}, repository {task.repo_name}, dataset {task.dataset_name}. "
                    "The candidate failed static preflight in this turn; no new patch was saved and no runtime test ran. "
                    "Do not stop at a data check. Correct every listed issue in the proposed code and contract, "
                    "then resubmit the complete analysis and revised files through preview_patch. "
                    "Do not weaken checks or silently change the intended training method. "
                    "After a valid preview inspect data; if real inputs are missing, stop and report exact paths. "
                    "Do not download, install, execute or apply against user limits. Issues: "
                    + str(failure['issues'])
                )
            if task.preflight_pending or len(task.automatic_test_history) <= seen['automatic']:
                continue
            if task.automatic_test_history[-1]["status"] != "failed":
                continue
            if _test_failures.get((session_id, task.repo_name.casefold()), 0) >= MAX_TEST_FAILURES:
                continue
            return (
                f"[Runtime dataset repair] Task {task.task_id}, repository {task.repo_name}, "
                f"dataset {task.dataset_name}, current batch_revision {len(task.patch_batches)}. "
                "The smoke test failed in this turn. Read the recorded logs, repair the implementation "
                "or explain and correct a faulty test, update the saved analysis only if the plan changed, "
                "submit revised files via preview_patch and rerun validate_dataset. Do not weaken assertions to force success. "
                "If dependencies/data are missing, report blocked and stop. Respect user limits; "
                "do not apply files without their request and CLI Y/N."
            )
    return None


def _result(task: DatasetTask) -> dict:
    latest = task.patch_batches[-1] if task.patch_batches else ()
    records = [patch_store.get_patch(patch_id) for patch_id in latest]
    try:
        profile = resolve_profile(get_repo_root(task.repo_name), task.dataset_name, task.integration_contract)
    except (OSError, ValueError):
        profile = None
    if task.open_questions:
        status = "needs_clarification"
    elif task.preflight_pending:
        status = "needs_revision"
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
    validation_status, validation_stages = _validation_result(task, records)
    automatic_status = "not_run"
    if task.automatic_test_history and tuple(task.automatic_test_history[-1]["patch_ids"]) == latest:
        automatic_status = task.automatic_test_history[-1]["status"]
        if automatic_status == "passed":
            from .dataset_testing import assert_tested
            try:
                assert_tested(latest[0])
            except (ValueError, OSError, KeyError):
                automatic_status = "stale"
    return {
        "ok": True, "task_id": task.task_id, "session_id": task.session_id,
        "repo_name": task.repo_name, "dataset_name": task.dataset_name,
        "task_description": task.task_description, "dataset_source": task.dataset_source,
        "status": status, "integration_summary": task.integration_summary,
        "validation_status": validation_status,
        "validation_plan_id": task.validation_plan_id,
        "validation_stages": validation_stages,
        "validation_history": list(task.validation_reports),
        "automatic_test_history": list(task.automatic_test_history),
        "automatic_test_status": automatic_status,
        "preflight_status": 'failed' if task.preflight_pending else ('passed' if task.preflight_checked else 'not_run'),
        "preflight_history": list(task.preflight_history),
        "rejected_diagnostics": list(task.rejected_diagnostics),
        "static_review": list(task.static_reviews),
        "automatic_failures": _test_failures.get((task.session_id, task.repo_name.casefold()), 0),
        "automatic_failure_limit": MAX_TEST_FAILURES,
        "test_contract": profile_description(profile),
        "integration_contract": task.integration_contract,
        "workflow_next_action": (
            "Resolve research questions with the user before proposing changes." if task.open_questions else
            "Repair all preflight issues and resubmit preview. Data inspection cannot resolve candidate code errors." if task.preflight_pending else
            "Inspect the actual workflow and save minimal proposals first. CPU validation is optional, using supported existing interfaces only." if not latest else
            "Stop: automatic repair budget exhausted; report logs and ask for human investigation." if
            _test_failures.get((task.session_id, task.repo_name.casefold()), 0) >= MAX_TEST_FAILURES and automatic_status != "passed" else
            "Stop: report the blocking reason; do not install/download or weaken assertions to force a pass." if automatic_status == "blocked" else
            "Read framework failure logs, revise the implementation and rerun smoke_test; do not write replacement tests." if automatic_status == "failed" else
            "Show the tested diff and results; apply only on user request with per-file Y/N." if automatic_status == "passed" else
            "Show the saved syntax-checked proposal. CPU validation is optional; only attempt requested verification with a supported existing interface and real local inputs."
        ),
        "validation_note": "Reports are user-reported, not independently executed or verified. Bound to exact patch IDs, check plan and current applied file hashes; unrelated files/environment are not tracked.",
        "validation_steps": list(task.validation_steps),
        "required_analysis_areas": list(ANALYSIS_AREAS),
        **_analysis_progress(task),
        "integration_analysis": [item.model_dump() for item in task.integration_analysis],
        "analysis_note": "Coverage and existing evidence locations are checked; semantic correctness and actual file-reading by the model are not verified.",
        "open_questions": list(task.open_questions),
        "patch_batches": [list(batch) for batch in task.patch_batches],
        "batch_revision": len(task.patch_batches),
        "patches": [{
            "patch_id": p.patch_id, "file_path": p.file_path,
            "operation": p.operation, "status": p.status.value, "diff": p.diff,
        } for p in records],
        "note": "This is the complete latest proposal. Review and apply pending IDs separately through CLI Y/N; do not reapply retained applied IDs. Preview does not execute code. smoke_test executes approved Python checks in a temporary source copy; inspect automatic_test_history for results. Multi-file application is not atomic.",
    }


@tool
def add_dataset(
    repo_name: str,
    dataset_name: str,
    session_id: str,
    action: Literal["plan", "preview", "status", "record_analysis", "inspect_data", "inspect_contract", "configure_contract", "validation_plan", "report_validation", "smoke_test", "reset_smoke_tests"] = "plan",
    task_id: str = "",
    task_description: str = "",
    dataset_source: str = "",
    integration_summary: str = "",
    changes: list[DatasetChange] | None = None,
    validation_steps: list[str] | None = None,
    open_questions: list[str] | None = None,
    integration_analysis: list[IntegrationAnalysis] | None = None,
    validation_checks: list[ValidationCheck] | None = None,
    validation_plan_id: str = "",
    batch_revision: int = 0,
    validation_stage: Literal["data_loading", "transformation", "dataloader", "pipeline", "regression"] | None = None,
    validation_outcome: Literal["user_reported_passed", "user_reported_failed"] | None = None,
    user_report: str = "",
    timeout_seconds: int = 60,
    dataset_paths: list[str] | None = None,
    integration_contract: IntegrationContract | None = None,
    entry_file: str = "",
) -> dict:
    """Plan a dataset integration, preview code changes, or report task status.

plan: start or safely retrieve a matching task using the user's high-level goal; returns repository
overview and an analysis checklist. Do NOT repeatedly plan the same task.
Use inspection tools to understand existing loaders, models and metrics.
For other actions an empty task_id is safely filled only when exactly one
task matches this session/repository/dataset. Explicit invalid or foreign IDs
are never replaced. Errors list only matching tasks, not other sessions.

preview: after discussing the plan, supply integration_summary and complete
per-file changes with reasons. validation_steps are optional documentation;
framework-owned defaults are supplied when omitted.
Supply integration_analysis covering every required_analysis_areas entry:
findings, existing repository file/line evidence, decision (change/keep/not_applicable),
rationale and planned_changes (explain no change where appropriate). Analyze
internal augmentations separately from outer transforms, and all evaluation
entry points. Locations are checked, not the truth of findings. Revisions must
resubmit the complete analysis to avoid silently reusing stale decisions. Modify
requires the inspected base hash; create requires expected_hash='' and an
existing parent directory. Nonempty open_questions block patch creation:
ask plain-language research questions, recommend technical choices yourself.
Questions remain pending across calls; pass open_questions=[] after the user
resolves them and describe the agreed choices in integration_summary.
Submit only files being revised or added. Omitted files retain their current
Patch IDs, including already-applied files. Identical active proposals reuse
their IDs and approval. A new batch is a complete task snapshot, not just this
call's changed files. No content changes means no new batch. Removal is not
supported. Omit per-file parent_patch_id: current parents are selected by the
tool. Empty parents or this task's own ID are treated as omitted metadata;
other explicit parents must be valid Patch IDs for the same target and session.
All changes are prepared before any records are saved.

inspect_data: check local input paths without executing code or downloading.
smoke_test: test using FRAMEWORK-OWNED checks, never agent-authored test scripts.
Do not supply smoke_checks. First call inspect_contract with the existing
entry_file to obtain original dataset cases, wrapper bindings and path rules.
configure_contract records repository interfaces,
real input paths and target/original cases using IntegrationContract. The
target name comes from this task: put its settings in target_case without a name.
For original datasets use the required_case_names from inspect_contract. Do not
create separate train/test cases. Identical duplicates are merged; conflicting
settings or missing num_classes/input_shape are reported explicitly, never guessed.
real entry point must use the same import-safe components factory that is tested.
Inspect test_contract for the schema and supported task types. Unsupported tasks
are blocked. Provide timeout_seconds; data files come from the saved contract.
The framework exercises the actual wrapper/model and real data. Direct CLI
execution consent is required, independently of file application approval.
An omitted batch_revision (0) uses this task's latest batch; an explicit
nonzero revision must match it. The test result records exact Patch IDs.
Code executes with account permissions, NOT an OS sandbox. dataset_paths can
select existing files/directories inside repo data/ or datasets/ for copying,
up to 1 GB. Use relative paths in checks; download=False; no installs or full
training. Missing dependencies/data block testing: stop and explain them.
On failure read the logs, revise using preview, rerun all five checks. Never
weaken assertions or use fake samples as evidence of real data integration.
The budget is four failed runs across tasks for this session/repo (initial plus
three repair attempts). Only a passing current complete batch can be applied.
reset_smoke_tests: only on user request, asks direct CLI Y/N to reset that budget;
invalidates prior certificates. It never marks a test passed.
validation_plan: after a preview, provide all five validation_checks (stage,
command, expected_result) and current batch_revision. Commands are saved, never run.
Identical plans retain their ID; changed plans invalidate previous results.
report_validation: only after the user reports running a check, provide the
current batch_revision, validation_plan_id, validation_stage, validation_outcome
(user_reported_passed/failed), and user_report quoting their result. Requires all
latest patches applied and matching saved hashes. Never invent user reports.
A revision invalidates old reports; history remains. These are self-reports,
not proof of execution. No automatic stage ordering is enforced.
status: retrieve patch states and version-bound manual validation records.
No LLM is called. Only smoke_test executes Python after direct terminal consent;
it does not grant patch approval. Tests are not an OS sandbox.
Code application always remains UNVERIFIED until the user tests it; this
tool cannot certify dataset compatibility or scientific correctness.
"""
    print(f"[DEBUG][add_dataset] {action}: {repo_name}/{dataset_name}", flush=True)
    try:
        if not session_id.strip() or not dataset_name.strip():
            raise ValueError("session_id and dataset_name must be non-empty")
        with _lock:
            if action == "plan":
                if changes:
                    raise ValueError("plan does not save changes; use preview with analysis and integration_summary")
                get_repo_root(repo_name)
                if task_id:
                    existing = _tasks.get(task_id)
                    if existing is None or (existing.session_id, existing.repo_name.casefold(), existing.dataset_name.strip().casefold()) != (session_id, repo_name.casefold(), dataset_name.strip().casefold()):
                        raise ValueError("Explicit Task ID does not belong to this session/repository/dataset")
                    return {**_result(existing), "task_reused": True}
                for existing in _tasks.values():
                    if (existing.session_id, existing.repo_name.casefold(), existing.dataset_name.strip().casefold()) == (
                        session_id, repo_name.casefold(), dataset_name.strip().casefold()
                    ):
                        result = _result(existing)
                        result["task_reused"] = True
                        return result
                if not task_description.strip():
                    raise ValueError("Provide the user's high-level task_description")
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
                patch_store.require_dataset_workflow(session_id, repo_name)
                result = _result(task)
                result.update({
                    "repository_overview": overview,
                    "analysis_checklist": [
                        "Inspect complete files before edits; identify loader, preprocessing, split, labels, model output, loss, CLI and evaluation contracts.",
                        "Check source/format and whether the dataset already exists. Explain target and group attributes; never invent them or silently change the research task.",
                        "Recommend technical defaults in plain language. Resolve research-scope questions before generating patches.",
                        "Preserve existing datasets and defaults. Separate optional new branches from unrelated refactoring.",
                        "Read test_contract, use inspect_data, and run smoke_test with the fixed framework. No agent-written tests, installs, downloads or full training.",
                    ],
                })
                return result
            matching = [t for t in _tasks.values() if
                        (t.session_id, t.repo_name.casefold(), t.dataset_name.strip().casefold()) ==
                        (session_id, repo_name.casefold(), dataset_name.strip().casefold())]
            if not task_id.strip():
                if len(matching) == 1:
                    task_id = matching[0].task_id
                    print(f"[DEBUG][add_dataset] Using current Task ID: {task_id}", flush=True)
                else:
                    return {
                        "ok": False, "error_code": "task_not_resolved",
                        "error": "No unique current task. Start plan if none exists; otherwise use an available Task ID.",
                        "available_tasks": [{"task_id": t.task_id, "repo_name": t.repo_name,
                                             "dataset_name": t.dataset_name} for t in matching],
                    }
            if task_id not in _tasks:
                return {
                    "ok": False, "error_code": "unknown_task_id",
                    "error": "Unknown Task ID. Use the saved ID returned by plan; explicit IDs are not guessed.",
                    "available_tasks": [{"task_id": t.task_id, "repo_name": t.repo_name,
                                         "dataset_name": t.dataset_name} for t in matching],
                }
            task = _tasks[task_id]
            if (task.session_id, task.repo_name.casefold(), task.dataset_name.strip().casefold()) != (
                session_id, repo_name.casefold(), dataset_name.strip().casefold()
            ):
                raise ValueError("Task does not belong to this session/repository/dataset")
            if action == "status":
                return _result(task)
            if action == 'record_analysis':
                questions = tuple(open_questions) if open_questions is not None else task.open_questions
                merged = {item.area: item for item in task.integration_analysis}
                for item in integration_analysis or []:
                    values = merged[item.area].model_dump() if item.area in merged else {}
                    values.update(item.model_dump(exclude_unset=True))
                    merged[item.area] = IntegrationAnalysis.model_validate(values)
                task = replace(task, integration_summary=integration_summary if integration_summary.strip() else task.integration_summary,
                               integration_analysis=tuple(merged[area] for area in ANALYSIS_AREAS if area in merged),
                               open_questions=questions)
                _tasks[task_id] = task
                result = _result(task)
                result['updated_analysis_areas'] = list(dict.fromkeys(item.area for item in integration_analysis or []))
                return result
            if action == "inspect_contract":
                if not entry_file:
                    raise ValueError("Provide the existing repository-relative entry_file, for example train.py or src/train.py")
                return {**_result(task), "repository_requirements": contract_requirements(get_repo_root(repo_name), entry_file, task.dataset_name)}
            if action == "configure_contract":
                if integration_contract is None:
                    raise ValueError("Provide integration_contract with real repository interfaces and data paths")
                configured = validate_contract(get_repo_root(repo_name), integration_contract, dataset_name)
                if configured != task.integration_contract:
                    invalidate(pid for batch in task.patch_batches for pid in batch)
                    task = replace(task, integration_contract=configured, preflight_checked=False)
                    _tasks[task.task_id] = task
                result = _result(task)
                result['parameter_guidance'] = inspect_contract_parameters(get_repo_root(repo_name), configured,
                    {patch_store.get_patch(pid).file_path: patch_store.get_patch(pid).updated_content
                     for pid in (task.patch_batches[-1] if task.patch_batches else ())})
                return result
            if action == "inspect_data":
                result = _result(task)
                result["local_data"] = inspect_local_data(get_repo_root(repo_name), task.integration_contract)
                return result
            if action == "reset_smoke_tests":
                try:
                    answer = input("Reset dataset smoke-test repair budget for this conversation/repo? [Y/N]: ").strip().upper()
                except (EOFError, KeyboardInterrupt):
                    answer = "N"
                if answer != "Y":
                    return {**_result(task), "budget_reset": False}
                _test_failures[(session_id, repo_name.casefold())] = 0
                for other in _tasks.values():
                    if (other.session_id, other.repo_name.casefold()) == (session_id, repo_name.casefold()):
                        invalidate(pid for batch in other.patch_batches for pid in batch)
                return {**_result(task), "budget_reset": True}
            if action == "smoke_test":
                if task.preflight_pending:
                    raise ValueError('Repair the rejected candidate and resubmit preview before smoke testing')
                if batch_revision == 0:
                    batch_revision = len(task.patch_batches)
                if not task.patch_batches or batch_revision != len(task.patch_batches):
                    raise ValueError("Provide the current batch_revision for smoke testing")
                if task.open_questions:
                    raise ValueError("Resolve open research questions before testing")
                scope = (session_id, repo_name.casefold())
                if _test_failures.get(scope, 0) >= MAX_TEST_FAILURES:
                    return {**_result(task), "ok": False, "error": "Automatic repair limit reached; stop and report errors. Only human-confirmed reset_smoke_tests can reset it."}
                result = run_checks(get_repo_root(repo_name), task.patch_batches[-1], timeout_seconds, dataset_paths or (), dataset_name, contract=task.integration_contract)
                if result["status"] == "failed":
                    _test_failures[scope] = _test_failures.get(scope, 0) + 1
                task = replace(task, automatic_test_history=task.automatic_test_history + (result,),
                               preflight_checked=result.get('candidate_preflight', {}).get('status') == 'passed')
                _tasks[task.task_id] = task
                return _result(task)
            if action in {"validation_plan", "report_validation"}:
                if not task.patch_batches or batch_revision != len(task.patch_batches):
                    raise ValueError("Provide the current batch_revision after creating a preview")
                if action == "validation_plan":
                    checks = tuple(validation_checks or ())
                    if len(checks) != len(VALIDATION_STAGES) or {c.stage for c in checks} != set(VALIDATION_STAGES):
                        raise ValueError("Provide one check for each of: " + ", ".join(VALIDATION_STAGES))
                    if any(not c.command.strip() or not c.expected_result.strip() for c in checks):
                        raise ValueError("Each check requires a command and expected_result")
                    checks = tuple(sorted(checks, key=lambda c: VALIDATION_STAGES.index(c.stage)))
                    if checks != task.validation_plan:
                        task = replace(task, validation_plan=checks, validation_plan_id=uuid4().hex)
                else:
                    if not task.validation_plan_id or validation_plan_id != task.validation_plan_id:
                        raise ValueError("Provide the current validation_plan_id")
                    if validation_stage is None or validation_outcome is None or not user_report.strip():
                        raise ValueError("Provide stage, outcome and a non-empty user_report")
                    ids = task.patch_batches[-1]
                    for pid in ids:
                        record = patch_store.get_patch(pid)
                        if record.status != patch_store.PatchStatus.APPLIED:
                            raise ValueError("Apply all latest patches before reporting validation")
                        current = resolve_target(repo_name, record.file_path).read_bytes()
                        if compute_file_hash(current) != compute_file_hash(record.updated_content):
                            raise ValueError("Applied file changed; create a new preview before validation")
                    report = dict(stage=validation_stage, outcome=validation_outcome,
                                  user_report=user_report, plan_id=validation_plan_id,
                                  batch_revision=batch_revision, patch_ids=list(ids),
                                  reported_at=datetime.now(timezone.utc).isoformat())
                    task = replace(task, validation_reports=task.validation_reports + (report,))
                _tasks[task.task_id] = task
                return _result(task)
            if action != "preview":
                raise ValueError("Unknown action")
            questions = tuple(open_questions) if open_questions is not None else task.open_questions
            if questions:
                task = replace(task, open_questions=questions)
                _tasks[task.task_id] = task
                return _result(task)
            if not validation_steps:
                validation_steps = [f"Framework stage: {stage}" for stage in VALIDATION_STAGES]
            if any(not step.strip() for step in validation_steps):
                raise ValueError("validation_steps entries must be non-empty")
            if not changes or len(changes) > 20:
                raise ValueError("Provide between 1 and 20 file changes")
            # Analysis is advisory for saving source, not an execution certificate.
            # Merge drafts without discarding earlier evidence or inventing it.
            merged = {item.area: item for item in task.integration_analysis}
            for item in integration_analysis or []:
                values = merged[item.area].model_dump() if item.area in merged else {}
                values.update(item.model_dump(exclude_unset=True))
                merged[item.area] = IntegrationAnalysis(**values)
            analysis = tuple(merged.values())
            integration_summary = integration_summary.strip() or task.integration_summary
            prepared = []
            previous = {
                patch_store.get_patch(pid).file_path.casefold(): patch_store.get_patch(pid)
                for pid in (task.patch_batches[-1] if task.patch_batches else ())
            }
            # Preserve order and all unchanged/applied proposals. Replace only
            # the slot for a revised file; append genuinely new target files.
            complete_batch = {key: record.patch_id for key, record in previous.items()}
            reasons = {}
            reviews = []
            paths = set()
            for change in changes:
                if not change.reason.strip():
                    raise ValueError("Each change needs a reason")
                proposal = prepare_patch(repo_name, change.file_path, change.updated_content,
                                         change.expected_hash, change.operation)
                if proposal.file_path.endswith('.py'):
                    from .dataset_review import review_dataset_change
                    review = review_dataset_change(proposal.original_content.decode('utf-8-sig'),
                                                   proposal.updated_content.decode('utf-8-sig'))
                    reviews.append(dict(file_path=proposal.file_path, **review))
                # Windows aliases must not create two competing patches in a batch.
                key = proposal.file_path.casefold()
                if key in paths:
                    raise ValueError("Duplicate target path in change batch")
                paths.add(key)
                reasons[key] = change.reason
                parent_id = (change.parent_patch_id or "").strip() or None
                # LLMs sometimes put the task ID in every optional ID field.
                # Dataset revisions already select their current parent below;
                # the known current task ID is not a request for another branch.
                if parent_id == task.task_id:
                    parent_id = None
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
                    try:
                        parent = patch_store.get_patch(parent_id)
                    except KeyError:
                        raise ValueError(
                            f"{proposal.file_path}: unknown parent_patch_id. Omit this field; "
                            "add_dataset selects the current file proposal automatically. "
                            "A Task ID is not a Patch ID."
                        ) from None
                    if (parent.session_id, parent.repo_name, parent.file_path, parent.operation) != (
                        session_id, repo_name, proposal.file_path, proposal.operation
                    ):
                        raise ValueError("Parent patch must belong to the same session, repository, file and operation")
                    if parent.original_hash != proposal.original_hash:
                        raise ValueError("Parent base changed; start a new proposal from inspected contents")
                prepared.append(dict(
                    testing_required=True, session_id=session_id, repo_name=repo_name, file_path=proposal.file_path,
                    original_hash=proposal.original_hash, original_content=proposal.original_content,
                    updated_content=proposal.updated_content, diff=proposal.diff,
                    parent_patch_id=parent_id, operation=proposal.operation,
                ))
            records = patch_store.create_patch_batch(prepared)
            for record in records:
                complete_batch[record.file_path.casefold()] = record.patch_id
            batches = task.patch_batches
            if records:
                invalidate(complete_batch.values())
                batches += (tuple(complete_batch.values()),)
            task = replace(
                task, integration_summary=integration_summary,
                static_reviews=tuple({r['file_path']: r for r in (*task.static_reviews, *reviews)}.values()),
                integration_analysis=analysis,
                validation_steps=tuple(validation_steps),
                open_questions=(),
                patch_batches=batches,
                preflight_pending=False,
                preflight_checked=False,
            )
            _tasks[task.task_id] = task
            result = _result(task)
            result["batch_created"] = bool(records)
            result["new_patch_ids"] = [record.patch_id for record in records]
            result['proposal_validation'] = 'syntax_and_file_integrity_only'
            result['warnings'] = ([] if result.get('analysis_complete') and not result.get('summary_required') else
                ['Analysis or integration summary remains pending. Proposal storage checks syntax and file integrity only; CPU validation and application requirements still apply.'])
            for item in result["patches"]:
                reason = reasons.get(item["file_path"].casefold())
                if reason is not None:
                    item["reason"] = reason
            return result
    except (ValueError, TypeError, KeyError, OSError) as exc:
        message = f"{type(exc).__name__}: {exc}"
        print(f"[ERROR][add_dataset] {message}", flush=True)
        result = {"ok": False, "error": message, "failed_action": action}
        if hasattr(exc, 'syntax_error'):
            result['syntax_error'] = exc.syntax_error
        if action == 'configure_contract':
            with _lock:
                current = _tasks.get(task_id)
                if current and (current.session_id, current.repo_name.casefold(), current.dataset_name.casefold()) == (session_id, repo_name.casefold(), dataset_name.casefold()):
                    try:
                        contract = integration_contract.model_dump() if integration_contract is not None else {}
                        candidates = {patch_store.get_patch(pid).file_path: patch_store.get_patch(pid).updated_content
                                      for pid in (current.patch_batches[-1] if current.patch_batches else ())}
                        guidance = inspect_contract_parameters(get_repo_root(repo_name), contract, candidates) if contract else {}
                        fields = getattr(exc, 'mapping_issues', [])
                        result.update(field_errors=fields[:20], field_error_count=len(fields), parameter_guidance=guidance,
                                      configuration_saved=False, previous_settings_retained=current.integration_contract is not None)
                        evidence = save_rejected_validation(session_id, task_id, repo_name, contract, message, fields, guidance)
                        _tasks[task_id] = replace(current, rejected_diagnostics=current.rejected_diagnostics + (evidence,))
                        result['rejected_diagnostic'] = evidence
                    except (OSError, ValueError, KeyError, TypeError) as diagnostic_error:
                        result['diagnostic_error'] = type(diagnostic_error).__name__
        if action == 'preview':
            with _lock:
                current = _tasks.get(task_id)
                if current and (current.session_id, current.repo_name.casefold()) == (session_id, repo_name.casefold()):
                    try:
                        evidence = save_rejected_candidate(session_id, task_id, repo_name, changes,
                            current.integration_contract, message, getattr(exc, 'preflight_issues', []))
                        current = replace(current, rejected_diagnostics=current.rejected_diagnostics + (evidence,))
                        _tasks[task_id] = current
                        result['rejected_diagnostic'] = evidence
                    except (OSError, ValueError) as diagnostic_error:
                        result['diagnostic_error'] = type(diagnostic_error).__name__
        if hasattr(exc, "issues"):
            result.update(required_corrections=exc.issues, repository_requirements=exc.requirements)
        if hasattr(exc, 'preflight_issues'):
            result['preflight_issues'] = exc.preflight_issues
            if action == 'preview':
                with _lock:
                    current = _tasks.get(task_id)
                    if current and (current.session_id, current.repo_name.casefold()) == (session_id, repo_name.casefold()):
                        failure = dict(issues=exc.preflight_issues, reason=message,
                                       recorded_at=datetime.now(timezone.utc).isoformat())
                        current = replace(current, preflight_pending=True, preflight_checked=False,
                                          preflight_history=current.preflight_history + (failure,))
                        _tasks[task_id] = current
                        scope = (session_id, repo_name.casefold())
                        _test_failures[scope] = _test_failures.get(scope, 0) + 1
        return result
