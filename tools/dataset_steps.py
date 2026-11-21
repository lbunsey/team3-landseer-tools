"""Small agent-facing steps backed by the existing dataset task/patch service.

No step calls the model or advances another phase automatically. Session IDs
are supplied by trusted CLI middleware. The legacy service remains available
to programmatic callers, but its large schema is not registered for the model.
"""
from typing import Literal

from langchain_core.tools import tool

from .add_dataset import add_dataset as service, IntegrationAnalysis, record_configuration_rejection
from .dataset_testing import IntegrationContract, inspect_contract_parameters, assess_component_interface
from . import patch_utils, patch_store
from .dataset_diagnostics import save_rejected_edits
from .dataset_discovery import discover_workflow
from .inspect_repo import get_repo_root


from .patch_preview import ExactReplacement, LineRangeEdit, assemble_replacements


DATASET_TOOL_NAMES = frozenset({
    'add_dataset', 'record_dataset_analysis', 'configure_dataset_validation', 'validate_dataset',
})


def dataset_stage(name, arguments):
    return {'record_dataset_analysis': 'record_analysis',
            'configure_dataset_validation': 'configure_contract', 'preview_patch': 'preview',
            'preview_dataset_file': 'preview', 'validate_dataset': 'smoke_test'}.get(
                name, arguments.get('action', 'plan') if name == 'add_dataset' else None)


def _compact(result, *, show_diff=False):
    if not result.get('ok'):
        return result
    keys = ('ok', 'task_id', 'repo_name', 'dataset_name', 'status', 'batch_revision',
            'open_questions', 'integration_summary', 'automatic_test_status',
            'automatic_failures', 'automatic_failure_limit', 'task_reused',
            'batch_created', 'new_patch_ids', 'proposal_validation', 'budget_reset', 'warnings', 'static_review',
            'repository_requirements', 'local_data', 'repository_overview',
            'required_analysis_areas', 'analysis_complete', 'completed_analysis_areas',
            'pending_analysis_areas', 'analysis_issues', 'summary_required', 'updated_analysis_areas', 'parameter_guidance')
    compact = {key: result[key] for key in keys if key in result}
    compact['patches'] = [{k: v for k, v in patch.items() if show_diff or k != 'diff'}
                          for patch in result.get('patches', [])]
    history = result.get('automatic_test_history') or []
    if history:
        compact['latest_cpu_result'] = history[-1]
    if result.get('open_questions'):
        next_action = 'Discuss research questions with the user; record the resolution before proposing code.'
    elif not result.get('analysis_complete'):
        next_action = 'Analysis is pending but does not block preview_patch. Read the real source and save requested minimal proposals now; report unknowns without inventing evidence. Do not ask again for permission already given to save.'
    elif not result.get('patches'):
        next_action = 'Complete-read the actual source, then preview_patch with full content or range edits. A complete read covers range edits without a second read_lines call. Create uses full updated_content.'
    elif not result.get('integration_contract'):
        next_action = 'Show saved syntax-checked proposals and pending limitations. CPU validation is optional: only when requested, configure an existing supported interface. Do not refactor merely for the runner.'
    elif result.get('automatic_test_status') == 'passed':
        next_action = 'Show the tested result; apply only on user request and separate Y/N approval.'
    elif result.get('automatic_test_status') == 'blocked':
        next_action = 'Stop and explain the recorded blocker. Do not install/download or fabricate inputs.'
    elif result.get('automatic_test_status') == 'failed':
        next_action = 'Read latest_cpu_result, revise the relevant files, and rerun validate_dataset within the repair budget.'
    else:
        next_action = 'Show saved proposals. If execution verification is requested and supported, inspect real inputs then validate_dataset. Otherwise report execution not verified.'
    compact['workflow_next_action'] = next_action
    return compact


@tool('add_dataset')
def dataset_task(
    repo_name: str,
    dataset_name: str,
    action: Literal['plan', 'status', 'read_lines', 'inspect_workflow', 'inspect_contract', 'inspect_data', 'reset_smoke_tests'] = 'plan',
    task_description: str = '',
    entry_file: str = '',
    session_id: str = '',
    start_line: int = 1,
    end_line: int = 200,
) -> dict:
    """Start/query a dataset task or inspect its original interfaces/local inputs.

    plan: provide the user's task_description. status: read progress.
    plan includes static workflow discovery from README/scripts and Python code.
    inspect_workflow: refresh entry candidates and source call evidence (read-only).
    read_lines: entry_file is the repository-relative text path. Returns numbered
    original lines and exact hash; select start_line/end_line (inclusive).
    Long requests return a page of at most 200 lines/20000 characters plus next_call.
    Follow next_call until has_more=false; only returned lines count as read.
    Use these actual line numbers for start_line/end_line replacement edits.
    inspect_contract: read-only analysis of an existing evidence-backed entry_file.
    No complete-read prerequisite. Missing/omitted entries return selection hints,
    not a workflow failure. Complete-read is still required before proposing edits.
    Never guess main.py or choose the first filename in the overview.
    inspect_data: report configured missing files without executing code.
    reset_smoke_tests: only on user request; requires terminal consent.
    This tool does not accept code or validation settings. Use the separate
    record_dataset_analysis, configure_dataset_validation, preview_patch
    and validate_dataset steps. The CLI supplies session_id; do not invent IDs.
    """
    selection_needed = False
    if action == 'inspect_contract':
        if not entry_file:
            selection_needed = True
        else:
            try:
                patch_utils.resolve_target(repo_name, entry_file)
            except FileNotFoundError:
                selection_needed = True
            except (OSError, ValueError) as exc:
                return dict(ok=False, error=str(exc))
    result = _compact(service.invoke(dict(repo_name=repo_name, dataset_name=dataset_name,
        session_id=session_id, action='status' if selection_needed or action in {'inspect_workflow', 'read_lines'} else action,
        task_description=task_description, entry_file=entry_file)))
    if result.get('ok') and (selection_needed or action in {'plan', 'inspect_workflow'}):
        result['repository_workflow'] = discover_workflow(repo_name)
    if result.get('ok') and selection_needed:
        result.update(entry_inspection_status='needs_entry_selection', requested_entry=entry_file,
                      note='No entry was inspected. Select an existing entry from command/source evidence; no code was executed.',
                      workflow_next_action='Choose a relevant existing entry using repository_workflow evidence, then inspect_contract. No preliminary complete-read is required.')
    elif result.get('ok') and action == 'inspect_contract':
        result['entry_inspection_status'] = 'inspected'
    if result.get('ok') and action == 'read_lines':
        try:
            if not 1 <= start_line <= end_line:
                raise ValueError('Select an inclusive 1-based range with end_line >= start_line')
            raw = patch_utils.resolve_target(repo_name, entry_file).read_bytes()
            lines = raw.decode('utf-8').replace('\r\n', '\n').split('\n')
            if lines and not lines[-1]:
                lines.pop()
            if start_line > len(lines):
                raise ValueError(f'start_line exceeds file line count {len(lines)}')
            excerpt, chars = [], 0
            target_end = min(end_line, len(lines))
            for i in range(start_line - 1, min(start_line + 199, target_end)):
                if chars + len(lines[i]) > 20000:
                    break
                excerpt.append(dict(line=i + 1, text=lines[i]))
                chars += len(lines[i])
            if not excerpt:
                result.update(file_path=entry_file, numbered_lines=[], read_status='line_too_long', has_more=True,
                              note='One source line exceeds the excerpt budget; use complete inspect_repo and full-content proposals rather than truncated line edits.')
                return result
            has_more = excerpt[-1]['line'] < target_end
            result.update(file_path=entry_file, original_hash=patch_utils.compute_file_hash(raw),
                          total_lines=len(lines), numbered_lines=excerpt, has_more=has_more,
                          requested_start_line=start_line, requested_end_line=end_line,
                          note='Original source, not the previous proposal; all edit ranges refer to this original.')
            if has_more:
                result['next_call'] = dict(name='add_dataset', args=dict(repo_name=repo_name, dataset_name=dataset_name,
                    action='read_lines', entry_file=entry_file, start_line=excerpt[-1]['line'] + 1, end_line=end_line))
                result['workflow_next_action'] = 'Continue reading with next_call; this page is successful, not a workflow failure.'
        except (ValueError, OSError) as exc:
            return dict(ok=False, error=str(exc))
    return result


@tool
def record_dataset_analysis(
    repo_name: str,
    dataset_name: str,
    integration_analysis: list[IntegrationAnalysis],
    integration_summary: str = '',
    open_questions: list[str] | None = None,
    session_id: str = '',
) -> dict:
    """Save only the agreed analysis; do not submit source code here.

    Save one or more analysis items at a time. Omitted areas and the existing
    summary are retained; supplying the same area updates only explicitly supplied
    fields in that item. An explicit empty evidence list clears its evidence.
    Drafts with missing text/evidence are saved and marked pending, not complete.
    Read analysis_issues and update only pending items. No need to repeat all eight.
    Eventually cover data_source, sample_contract, transforms, internal_augmentations,
    splits, model_and_loss, training_entry and evaluation. Each item uses area,
    findings (text), evidence (objects with file_path and integer line), decision
    (change/keep/not_applicable), rationale (text), planned_changes (text).
    Research questions can be recorded with open_questions; no code may be
    proposed while questions remain. Pass [] once resolved. No code runs/writes.
    """
    return _compact(service.invoke(dict(repo_name=repo_name, dataset_name=dataset_name,
        session_id=session_id, action='record_analysis', integration_summary=integration_summary,
        integration_analysis=[item.model_dump(exclude_unset=True) for item in integration_analysis], open_questions=open_questions)))


@tool
def configure_dataset_validation(
    repo_name: str,
    dataset_name: str,
    integration_contract: IntegrationContract,
    session_id: str = '',
) -> dict:
    """Save CPU verification settings only; never submit code here.

    First inspect_contract for the real entry and original datasets. Supply
    target_case settings without a name; the program owns the target identity.
    Include real local input paths and original dataset cases. The existing
    runner supports tensor classification with single-view CE or three-view JSD.
    Prefer interface_mode=component_adapter for an existing callable: factory_file
    and factory_name identify it; argument_map maps dataset_name/data_root to its
    actual parameter names. Map device/download if accepted; set remaining options
    via factory_kwargs. Return train_dataset/test_dataset/model as dict or tuple.
    inspect_contract returns callable_signatures from real source. On rejection,
    read field_errors and parameter_guidance, including the declared callable and
    its actual parameters. Map controlled name -> formal parameter identifier,
    never args.dataset, a dataset value such as svhn, or a filesystem path.
    No main() factory call is required in this mode, and success ONLY certifies
    component data flow, NOT entry/CLI integration. Import must still be safe.
    This phase validates metadata only, without importing repository code.
    """
    task = service.invoke(dict(repo_name=repo_name, dataset_name=dataset_name, session_id=session_id, action='status'))
    if not task.get('ok'):
        return task
    candidates = {p['file_path']: patch_store.get_patch(p['patch_id']).updated_content for p in task['patches']}
    guidance = inspect_contract_parameters(get_repo_root(repo_name), integration_contract, candidates)
    assessment = assess_component_interface(guidance)
    if assessment['status'] in {'needs_component_adapter', 'needs_component_selection'}:
        guidance['interface_assessment'] = assessment
        selection = assessment['status'] == 'needs_component_selection'
        error = ('The selected callable is not a usable component builder for this CPU interface. '
                 + '; '.join(issue['reason'] for issue in assessment['issues']))
        result = dict(ok=False, error_code=assessment['status'], recoverable_precondition=True,
            task_id=task['task_id'], repo_name=repo_name, dataset_name=dataset_name,
            error=error, configuration_saved=False, previous_settings_retained=task['integration_contract'] is not None,
            parameter_guidance=guidance, interface_assessment=assessment,
            saved_patch_count=len(task['patches']), validation_optional=True,
            proposal_creation_blocked=False,
            next_tool='configure_dataset_validation' if selection else 'preview_patch',
            workflow_next_action=(
                'Inspect the existing component leads and select a suitable real callable, then configure its actual parameters. Do not generate an adapter if an existing builder is usable.' if selection else
                'This interface is unsupported by the optional CPU runner. Do not invent a callable or refactor solely for verification. Save/show the minimal dataset proposal via preview_patch using the actual repository workflow; report execution not verified. Reuse a supported existing builder only if one is found. Do not apply, download or install.'))
        if not selection and not task.get('analysis_complete'):
            result.update(analysis_issues=task['analysis_issues'], pending_analysis_areas=task['pending_analysis_areas'],
                workflow_next_action=result['workflow_next_action'] + ' Analysis remains pending; it does not block saving source proposals.')
        try:
            result['rejected_diagnostic'] = record_configuration_rejection(session_id, task['task_id'], repo_name,
                integration_contract.model_dump(), error, guidance)
        except (ValueError, OSError) as exc:
            result['diagnostic_error'] = type(exc).__name__
        print(f"[DEBUG][configure_dataset_validation] Interface: {assessment['status']}", flush=True)
        return result
    result = service.invoke(dict(repo_name=repo_name, dataset_name=dataset_name,
        session_id=session_id, action='configure_contract', integration_contract=integration_contract.model_dump()))
    compact = _compact(result)
    if result.get('ok'):
        compact['saved_validation_settings'] = result['integration_contract']
        compact['interface_assessment'] = assessment
    return compact


@tool
def save_dataset_proposal(
    repo_name: str,
    dataset_name: str,
    file_path: str,
    reason: str,
    updated_content: str | None = None,
    operation: Literal['modify', 'create'] = 'modify',
    expected_hash: str = '',
    session_id: str = '',
    replacements: list[LineRangeEdit] | None = None,
    task_id: str = '',
    parent_patch_id: str | None = None,
) -> dict:
    """Internal task association helper; NOT registered as an agent tool.

    For modify use replacements [{start_line: N, end_line: M, new_lines: [...]}]
    or complete updated_content. Complete inspect_repo reads cover ranges;
    read_lines is optional for locating boundaries. Ranges are inclusive
    and always refer to the hash-bound ORIGINAL, not previous edits/proposals.
    The tool obtains old source itself; do not submit old_lines or old_text.
    All ranges must be disjoint. [] deletes a range. This tool accepts only range
    replacements, not old_text/old_lines. The internal assembly helper retains
    compatibility with older programmatic edit objects.
    Each array item may be a source line or a multiline source block, preserving
    indentation. Real LF/CRLF is accepted; literal backslash escapes are unchanged.
    Do not encode line separators as literal backslash-n. Text pairs are legacy.
    Each original fragment must
    match exactly once in the ORIGINAL inspected file; include enough context.
    Send all edits for this file together; each submission starts from the original,
    not the previous proposal. Use real source newlines, never manually escape JSON.
    Do not supply updated_content together with replacements. Create requires
    updated_content. Full-content modify remains available for compatibility.
    updated_content MUST be the complete real source text, never a description,
    list of instructions, JSON-encoded code or diff. Saved analysis is reused;
    incomplete analysis remains pending and does not block proposal storage.
    For modify first complete-read this exact file with inspect_repo; the CLI
    supplies expected_hash from that read. Direct callers supply the real hash.
    For create the file must be absent, expected_hash empty, parent already present.
    Other task files are retained in the batch. No semantic checks, repository
    imports, execution, approval or target-file writes occur here. Use
    validate_dataset after ALL required files are saved.
    """
    task = service.invoke(dict(repo_name=repo_name, dataset_name=dataset_name,
                              session_id=session_id, action='status', task_id=task_id))
    if not task.get('ok'):
        return task
    if task['open_questions']:
        return dict(ok=False, error_code='open_research_questions',
                    error='Resolve the recorded research questions before proposing code.', open_questions=task['open_questions'])
    original = None
    try:
        if (updated_content is None) == (replacements is None):
            raise ValueError('Provide exactly one of replacements or updated_content; never a plan')
        if replacements is not None:
            if operation != 'modify':
                raise ValueError('Create requires full updated_content, not replacements')
            raw = patch_utils.resolve_target(repo_name, file_path).read_bytes()
            original = raw.decode('utf-8')
            if patch_utils.compute_file_hash(raw) != expected_hash:
                raise ValueError('Original hash changed; complete-read the current file before proposing edits')
            updated_content = assemble_replacements(original, replacements)
    except (ValueError, OSError) as exc:
        result = dict(ok=False, error_code='invalid_source_edits', error=str(exc))
        print(f"[ERROR][dataset_proposal] {str(exc)[:500]}", flush=True)
        if hasattr(exc, 'match_details'):
            result['match_details'] = exc.match_details
        try:
            result['rejected_diagnostic'] = save_rejected_edits(session_id, task['task_id'],
                repo_name, file_path, replacements, expected_hash, original, exc)
        except (ValueError, OSError) as diagnostic_error:
            result['diagnostic_error'] = type(diagnostic_error).__name__
        return result
    result = service.invoke(dict(repo_name=repo_name, dataset_name=dataset_name,
        session_id=session_id, task_id=task['task_id'], action='preview', integration_summary=task['integration_summary'],
        integration_analysis=task['integration_analysis'],
        changes=[dict(file_path=file_path, updated_content=updated_content, reason=reason,
                      operation=operation, expected_hash=expected_hash, parent_patch_id=parent_patch_id)]))
    compact = _compact(result, show_diff=True)
    if any(getattr(edit, 'text', None) is not None for edit in (replacements or [])):
        compact.setdefault('warnings', []).append('Ignored redundant text metadata; replacement code comes only from new_lines.')
    if result.get('ok'):
        saved = next(p for p in result['patches'] if p['file_path'] == file_path.replace('\\', '/'))
        compact.update({key: saved[key] for key in ('patch_id', 'file_path', 'operation', 'diff', 'status') if key in saved})
        record = patch_store.get_patch(saved['patch_id'])
        compact.update(original_hash=record.original_hash, parent_patch_id=record.parent_patch_id)
        compact['session_id'] = session_id
    return compact


# Compatibility for local programmatic callers; absent from TOOLS.
preview_dataset_file = save_dataset_proposal


@tool
def validate_dataset(
    repo_name: str,
    dataset_name: str,
    timeout_seconds: int = 60,
    session_id: str = '',
) -> dict:
    """Validate the task's latest complete candidate batch on CPU.

    First configure_dataset_validation and save all actual file proposals.
    The framework owns checks: data, transforms/wrapper, batch, forward/loss/
    backward/update, and original dataset regressions. It tests a temporary copy,
    never the live repo, after terminal execution consent. No downloads or installs.
    Missing real inputs/dependencies are blocked, never passed. This does not
    apply files or verify full training/scientific performance.
    """
    task = service.invoke(dict(repo_name=repo_name, dataset_name=dataset_name,
                              session_id=session_id, action='status'))
    if not task.get('ok'):
        return task
    if not task['integration_contract']:
        return dict(ok=False, error_code='validation_settings_required',
                    error='Use configure_dataset_validation before CPU execution. No code was run.')
    return _compact(service.invoke(dict(repo_name=repo_name, dataset_name=dataset_name,
        session_id=session_id, action='smoke_test', timeout_seconds=timeout_seconds)))


DATASET_STEP_TOOLS = [dataset_task, record_dataset_analysis, configure_dataset_validation, validate_dataset]
