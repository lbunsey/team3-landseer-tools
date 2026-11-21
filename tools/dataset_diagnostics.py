"""Local rejected-candidate evidence. Never an approved or applicable patch."""
import hashlib
import json
from pathlib import Path
from uuid import uuid4

from .patch_utils import _reject_link

DIAGNOSTIC_ROOT = Path(__file__).resolve().parent.parent / '.dataset_diagnostics'


def _write_evidence(session_id, task_id, payload):
    directory = DIAGNOSTIC_ROOT
    for part in (None, hashlib.sha256(session_id.encode()).hexdigest()[:24],
                 hashlib.sha256(task_id.encode()).hexdigest()[:24]):
        if part is not None:
            directory = directory / part
        if directory.exists() or directory.is_symlink():
            _reject_link(directory)
        else:
            directory.mkdir()
    diagnostic_id = uuid4().hex
    path = directory / (diagnostic_id + '.json')
    payload.update(diagnostic_id=diagnostic_id, applicable=False)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    return dict(diagnostic_id=diagnostic_id, path=str(path), applicable=False,
                note='Diagnostic evidence only, not an applicable patch.')


def save_rejected_candidate(session_id, task_id, repo_name, changes, contract, error, issues):
    proposals = []
    for change in (changes or [])[:20]:
        item = change.model_dump() if hasattr(change, 'model_dump') else dict(change)
        content = item.get('updated_content', '')
        if len(content.encode('utf-8')) > 500000:
            item['updated_content'] = content.encode('utf-8')[:500000].decode('utf-8', errors='ignore')
            item['diagnostic_truncated'] = True
        proposals.append(item)
    payload = dict(task_id=task_id, repo_name=repo_name,
                   rejected=True, applicable=False, error=error, issues=issues,
                   integration_contract=contract, changes=proposals)
    return _write_evidence(session_id, task_id, payload)


def save_invalid_tool_call(session_id, tool_name, arguments, errors):
    """Capture failures that occur before the typed service body can run."""
    raw = json.dumps(arguments, ensure_ascii=False)
    payload = dict(kind='invalid_tool_arguments', tool=tool_name, errors=errors,
                   rejected=True, arguments_truncated=len(raw) > 2_000_000)
    if len(raw) <= 2_000_000:
        payload['arguments'] = arguments
    else:
        payload['arguments_excerpt'] = raw[:2_000_000]
    return _write_evidence(session_id, 'unvalidated-tool-call', payload)


def save_rejected_edits(session_id, task_id, repo_name, file_path, replacements,
                        expected_hash, original, error):
    """Keep received edits and the exact base for diagnosing failed matching."""
    arguments = [item.model_dump(exclude_none=True) for item in (replacements or [])]
    raw = json.dumps(arguments, ensure_ascii=False)
    payload = dict(kind='rejected_source_edits', task_id=task_id, repo_name=repo_name,
                   file_path=file_path, expected_hash=expected_hash, error=str(error),
                   original_content=original, replacements_truncated=len(raw) > 2_000_000)
    if len(raw) <= 2_000_000:
        payload['replacements'] = arguments
    else:
        payload['replacements_excerpt'] = raw[:2_000_000]
    return _write_evidence(session_id, task_id, payload)


def save_rejected_validation(session_id, task_id, repo_name, contract, error, field_errors, guidance):
    raw = json.dumps(contract, ensure_ascii=False)
    payload = dict(kind='rejected_validation_settings', task_id=task_id, repo_name=repo_name,
                   error=error, field_errors=field_errors, parameter_guidance=guidance,
                   configuration_saved=False, contract_truncated=len(raw) > 2_000_000)
    if len(raw) <= 2_000_000:
        payload['integration_contract'] = contract
    else:
        payload['contract_excerpt'] = raw[:2_000_000]
    return _write_evidence(session_id, task_id, payload)
