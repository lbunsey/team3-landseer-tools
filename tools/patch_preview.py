"""Read-only previews of single-file UTF-8 patches; no repository writes.

Callers supply the FULL proposed file and the hash of the base they inspected.
Proposed line endings follow the original file; BOM is not normalized.
Records are process-local.
CLI confirmation is required before the saved proposal can be applied.
"""

from dataclasses import dataclass
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, BeforeValidator
import difflib
import os
import re

from langchain_core.tools import tool

from . import patch_store, patch_utils


def _normalize_operation(value):
    return 'modify' if value == 'replace' else value


PatchOperation = Annotated[Literal['modify', 'create'], BeforeValidator(_normalize_operation)]


class ExactReplacement(BaseModel):
    """Literal source fragments, not JSON-encoded strings or regular expressions."""
    model_config = ConfigDict(extra='forbid')
    old_lines: list[str] | None = None
    new_lines: list[str] | None = None
    old_text: str | None = None
    new_text: str | None = None
    start_line: int | None = None
    end_line: int | None = None

    def source_pair(self):
        if self.old_lines is not None or self.new_lines is not None:
            if self.old_text is not None or self.new_text is not None:
                raise ValueError('Use old_lines/new_lines OR old_text/new_text, never both')
            if not self.old_lines or self.new_lines is None:
                raise ValueError('Provide nonempty old_lines and new_lines (empty list deletes)')
            if any('\n' in line or '\r' in line for line in self.old_lines + self.new_lines):
                raise ValueError('Each lines item must be one source line; use separate items for newlines')
            return '\n'.join(self.old_lines), '\n'.join(self.new_lines)
        if self.old_text is None or self.new_text is None:
            raise ValueError('Provide old_lines and new_lines')
        return self.old_text, self.new_text


class LineRangeEdit(BaseModel):
    """Agent-facing edits: tool reads old source, model supplies only new lines."""
    model_config = ConfigDict(extra='forbid')
    start_line: int = Field(ge=1, strict=True)
    end_line: int = Field(ge=1, strict=True)
    new_lines: list[str] = Field(description='Replacement source lines or multiline source blocks. Real LF/CRLF is accepted; literal backslash escapes are preserved.')

    text: str | None = Field(default=None, description='Legacy source metadata, ignored. Only new_lines supplies replacement code.')

    @field_validator('start_line', 'end_line', mode='before')
    @classmethod
    def normalize_numeric_string(cls, value):
        # Preserve strict rejection of bools, floats and ambiguous expressions.
        if isinstance(value, str) and re.fullmatch(r'[0-9]{1,9}', value):
            return int(value)
        return value


def assemble_replacements(original, replacements):
    """Match literal source with LF/CRLF equivalence; never decode escapes."""
    if not replacements or len(replacements) > 50:
        raise ValueError('Provide 1-50 exact replacements')
    base = original.replace('\r\n', '\n')
    if '\r' in base or ('\r\n' in original and '\n' in original.replace('\r\n', '')):
        raise ValueError('Mixed or bare-CR original line endings are not supported')
    newline = '\r\n' if '\r\n' in original else '\n'
    spans = []
    for index, edit in enumerate(replacements, 1):
        if edit.start_line is not None or edit.end_line is not None:
            if any(getattr(edit, name, None) is not None for name in ('old_lines', 'old_text', 'new_text')):
                raise ValueError('Line-range edits use start_line/end_line/new_lines only; do not copy old source')
            lines = base.split('\n')
            physical = [line + '\n' for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
            if (type(edit.start_line) is not int or type(edit.end_line) is not int
                    or not 1 <= edit.start_line <= edit.end_line <= len(physical)):
                raise ValueError(f'Replacement {index}: invalid inclusive 1-based original line range')
            if edit.new_lines is None:
                raise ValueError('Provide new_lines as source lines or blocks; [] deletes the selected lines')
            start = sum(map(len, physical[:edit.start_line - 1]))
            end = start + sum(map(len, physical[edit.start_line - 1:edit.end_line]))
            multiline = any('\n' in line or '\r' in line for line in edit.new_lines)
            new = '\n'.join(edit.new_lines).replace('\r\n', '\n')
            if '\r' in new:
                raise ValueError('Bare CR source is not supported; use LF or CRLF')
            if edit.new_lines and base[start:end].endswith('\n') and (not multiline or not new.endswith('\n')):
                new += '\n'
            if new == base[start:end]:
                continue  # Harmless unchanged range in a larger edit batch.
            if any(start < b and a < end for a, b, _ in spans):
                raise ValueError('Replacement ranges overlap; submit disjoint original ranges')
            spans.append((start, end, new))
            continue
        old, new = edit.source_pair()
        old = old.replace('\r\n', '\n')
        new = new.replace('\r\n', '\n')
        if '\r' in old or '\r' in new:
            raise ValueError(f'Replacement {index}: bare CR is not supported')
        if not old or old == new:
            raise ValueError('old_text must be nonempty and differ from new_text')
        start = base.find(old)
        if start < 0:
            failure = ValueError(f'Replacement {index}: old_text has 0 matches after LF/CRLF normalization; copy the inspected original fragment')
            old_lines, base_lines = old.splitlines(), base.splitlines()
            # A suggestion only: never use a similar fragment to apply an edit.
            anchor = old_lines[0] if old_lines else old
            candidates = difflib.get_close_matches(anchor, base_lines, n=1, cutoff=0.5)
            detail = dict(replacement_index=index, submitted_excerpt=repr(old[:500]),
                          literal_backslash_n=old.count('\\n'), real_newlines=old.count('\n'))
            if candidates:
                line = base_lines.index(candidates[0])
                context = '\n'.join(base_lines[line:line + min(len(old_lines), 8)])
                detail.update(possible_original_line=line + 1, possible_original_excerpt=repr(context[:700]),
                              suggestion_only=True)
                mismatch = next((i for i, (a, b) in enumerate(zip(old, context)) if a != b),
                                min(len(old), len(context)))
                detail['first_difference_in_suggestion'] = dict(offset=mismatch,
                    submitted=repr(old[max(0, mismatch - 30):mismatch + 60]),
                    original=repr(context[max(0, mismatch - 30):mismatch + 60]))
            failure.match_details = detail
            raise failure
        if base.find(old, start + 1) >= 0:
            raise ValueError(f'Replacement {index}: old_text has multiple matches after LF/CRLF normalization; include more context')
        end = start + len(old)
        if any(start < b and a < end for a, b, _ in spans):
            raise ValueError('Replacement ranges overlap; submit disjoint original fragments')
        spans.append((start, end, new))
    result = base
    for start, end, new in sorted(spans, reverse=True):
        result = result[:start] + new + result[end:]
    return result.replace('\n', newline)


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
reader. Raises on stale bases, no-op changes, binary/non-UTF-8, oversized
contents or Python compilation errors. Compilation never executes code and
uses the running interpreter's Python version, not the repository's runtime.
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
    if target.suffix.lower() == ".py":
        try:
            # Decode policy is already strict UTF-8. Strip a leading BOM only
            # for compilation; preserve it in the actual saved bytes/diff.
            # dont_inherit prevents this module's future flags affecting code.
            compile(updated_content.removeprefix("\ufeff"), relative, "exec", dont_inherit=True)
        except SyntaxError as exc:
            failure = ValueError(
                f"Python syntax check failed: {relative}, line {exc.lineno}, "
                f"column {exc.offset} — {exc.msg}"
            )
            def excerpt(text, line):
                lines = text.splitlines()
                return [{'line': n, 'text': lines[n - 1][:1000]}
                        for n in range(max(1, line - 4), min(len(lines), line + 6) + 1)]
            failure.syntax_error = dict(file_path=relative, line=exc.lineno, column=exc.offset,
                message=exc.msg, candidate_excerpt=excerpt(updated_content, exc.lineno or 1),
                original_excerpt=excerpt(original_text, exc.lineno or 1),
                note='Candidate line numbers can shift. Replacement ranges always refer to the inspected ORIGINAL. Check complete statement boundaries; do not guess missing lines.')
            raise failure from exc
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
        return {"ok": False, "error": message, **({'syntax_error': exc.syntax_error} if hasattr(exc, 'syntax_error') else {})}
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
    updated_content: str | None = None,
    expected_hash: str = '',
    session_id: str = '',
    parent_patch_id: str | None = None,
    operation: PatchOperation = "modify",
    replacements: list[LineRangeEdit] | None = None,
    reason: str = '',
    task_id: str = '',
) -> dict:
    """Preview modifying or creating one UTF-8 file without writing it.

For a new file use operation='create', expected_hash=''; the parent directory
must exist. Existing paths are never overwritten by create proposals.

    Unified proposal tool for dataset, model and general changes. Supply either
    FULL updated_content, or replacements [{start_line, end_line, new_lines}] for
    existing files. Ranges are inclusive 1-based ORIGINAL lines; read numbered
    source covering the range first, or complete-read the file. The CLI checks source visibility
    against the base hash. Never retype old source. New files require full updated_content.
    The official CLI binds a unique active task and base hash; task_id is needed
    only to disambiguate multiple dataset tasks in that session/repo. reason
    describes the change. Workflow association is handled by runtime, not here.
    Direct callers must provide the exact inspected base hash for modifications.
Proposed LF/CRLF endings are converted to the original file's uniform style
before diff generation and storage. Mixed-ending originals are rejected.
Use the same session_id and parent_patch_id for a revision. Every successful
call saves a new pending-review ID and returns the full, untruncated diff.
No approval or application occurs. Errors do not save or supersede records.
Python proposals must compile with the running Python version; no code is
executed/imported. This does not check dependencies or program correctness.
"""
    print(f"[DEBUG][patch_preview] Preparing proposal: {repo_name}/{file_path}", flush=True)
    try:
        operation = _normalize_operation(operation)
        if (updated_content is None) == (replacements is None):
            raise ValueError('Provide exactly one of updated_content or replacements')
        if replacements is not None:
            if operation != 'modify':
                raise ValueError('Create requires complete updated_content')
            raw = patch_utils.resolve_target(repo_name, file_path).read_bytes()
            if patch_utils.compute_file_hash(raw) != expected_hash:
                raise ValueError('Stale base: file changed; inspect it again before previewing')
            updated_content = assemble_replacements(raw.decode('utf-8'), replacements)
        prepared = prepare_patch(repo_name, file_path, updated_content, expected_hash, operation)
        record = _save_prepared(prepared, session_id, parent_patch_id)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        message = f"{type(exc).__name__}: {exc}"
        print(f"[ERROR][patch_preview] Preview rejected: {message}", flush=True)
        return {"ok": False, "error": message, **({'syntax_error': exc.syntax_error} if hasattr(exc, 'syntax_error') else {})}
    print(f"[DEBUG][patch_preview] Saved patch: {record.patch_id}", flush=True)
    return {
        "ok": True, "patch_id": record.patch_id,
        "session_id": record.session_id, "repo_name": record.repo_name,
        "file_path": record.file_path, "original_hash": record.original_hash,
        "parent_patch_id": record.parent_patch_id, "status": record.status.value,
        "diff": record.diff,
        "operation": record.operation,
        "warnings": (["Ignored redundant text metadata; replacement code comes only from new_lines."]
                     if any(getattr(edit, 'text', None) is not None for edit in (replacements or [])) else []),
    }
