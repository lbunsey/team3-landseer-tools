"""Advisory source review for dataset proposals; never imports repository code.

Checks detect narrow, explainable risks, not semantic correctness. Findings do
not approve, rewrite or block a patch. Moving code across files may need review.
"""
import ast
import difflib
from pathlib import Path
import sysconfig


def _dataset_aliases(tree):
    modules, classes = set(), {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == 'torchvision':
                modules.update(a.asname or a.name for a in node.names if a.name == 'datasets')
            elif node.module == 'torchvision.datasets':
                classes.update((a.asname or a.name, a.name) for a in node.names)
        elif isinstance(node, ast.Import):
            modules.update(a.asname or a.name for a in node.names if a.name == 'torchvision.datasets')
    return modules, classes


def _loader_calls(tree):
    modules, classes = _dataset_aliases(tree)
    result = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = None
        if isinstance(node.func, ast.Name):
            name = classes.get(node.func.id)
        elif isinstance(node.func, ast.Attribute) and ast.unparse(node.func.value) in modules:
            name = node.func.attr
        if name:
            result.append((name, node))
    return result


def _installed_signatures():
    """Read installed torchvision constructor declarations without importing it."""
    root = Path(sysconfig.get_path('purelib')) / 'torchvision' / 'datasets'
    signatures = {}
    if not root.is_dir():
        return signatures
    for path in root.glob('*.py'):
        try:
            tree = ast.parse(path.read_text(encoding='utf-8'))
        except (OSError, UnicodeError, SyntaxError):
            continue
        for cls in tree.body:
            if not isinstance(cls, ast.ClassDef):
                continue
            ctor = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__'), None)
            if ctor and ctor.args.kwarg is None:
                signatures[cls.name] = (set(a.arg for a in ctor.args.args + ctor.args.kwonlyargs), str(path))
    return signatures


def _overwrites(tree):
    result = []
    for block in ast.walk(tree):
        for field in ('body', 'orelse', 'finalbody'):
            statements = getattr(block, field, None)
            if not isinstance(statements, list):
                continue
            previous = {}
            for node in statements:
                if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Constant):
                    continue
                for target in node.targets:
                    if not isinstance(target, ast.Name) or target.id not in {'num_classes', 'n_classes', 'NUM_CLASSES'}:
                        continue
                    if target.id in previous and previous[target.id][0] != node.value.value:
                        result.append(dict(code='class_count_overwritten', line=node.lineno,
                            message=f'{target.id} is assigned {previous[target.id][0]!r} at line {previous[target.id][1]} and then {node.value.value!r} in the same block. Check the dataset branch boundaries.'))
                    previous[target.id] = (node.value.value, node.lineno)
    return result


def _escape_changes(old, new):
    """Compare parsed string values only; never decode or rewrite source text."""
    def literals(node, scope=(), calls=()):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            scope = scope + (node.name,)
            calls = ()
        if isinstance(node, ast.Call):
            func = node.func
            name = ('<string>.' + func.attr if isinstance(func, ast.Attribute)
                    and isinstance(func.value, ast.Constant) and isinstance(func.value.value, str)
                    else ast.unparse(func))
            calls = calls + (name,)
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield (scope, calls), node
        for child in ast.iter_child_nodes(node):
            yield from literals(child, scope, calls)

    original = list(literals(old))
    existing = {(context, node.value) for context, node in original}
    escaped = {}
    controls = (('\n', '\\n'), ('\t', '\\t'), ('\r', '\\r'))
    for context, node in original:
        for mask in range(1, 8):
            value = node.value
            for index, (control, literal) in enumerate(controls):
                if mask & (1 << index):
                    value = value.replace(control, literal)
            if value != node.value:
                escaped.setdefault((context, value), []).append(node)
    issues = []
    for context, node in literals(new):
        matches = escaped.get((context, node.value), [])
        if matches and (context, node.value) not in existing:
            previous = min(matches, key=lambda item: abs(item.lineno - node.lineno))
            issues.append(dict(code='string_escape_semantics_changed', line=node.lineno,
                original_line=previous.lineno,
                original_value=repr(previous.value)[:300], candidate_value=repr(node.value)[:300],
                message='An existing string newline/tab/carriage return became literal backslash text in the same scope/call context. This can change console output or CSV row separators. Preserve the original string unless this behavior change is intentional; do not globally unescape source.'))
    return issues


def _source_excerpt(source, start, end, limit=90):
    lines = source.splitlines()
    start, end = max(1, start), min(len(lines), end)
    stop = min(end, start + limit - 1)
    rows, remaining = [], 6000
    for n in range(start, stop + 1):
        text = lines[n - 1]
        if len(text) > remaining:
            break
        rows.append(dict(line=n, text=text))
        remaining -= len(text)
    actual_end = rows[-1]['line'] if rows else start - 1
    return dict(start_line=start, end_line=actual_end, complete=actual_end == end,
        requested_end_line=end, numbered_lines=rows)


def _repair_context(original, candidate, old, new, issues):
    """Provide actual source, never a generated repair or guessed edit range."""
    old_lines, new_lines = original.splitlines(), candidate.splitlines()
    mapping = difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False).get_opcodes()
    contexts = []
    for issue in issues[:4]:
        line = issue['line']
        if issue['code'] == 'existing_loader_removed':
            containers = [n for n in ast.walk(old) if isinstance(n, ast.If)
                          and n.lineno <= line <= n.end_lineno]
            block = min(containers, key=lambda n: n.end_lineno - n.lineno) if containers else None
            start, end = (block.lineno, block.end_lineno) if block else (line - 5, line + 7)
            target = next((j + min(max(0, line - 1 - i), max(0, je - j - 1)) + 1
                           for _, i, ie, j, je in mapping if i <= line - 1 < ie), line)
        else:
            original_line = issue.get('original_line') or next((i + min(max(0, line - 1 - j), max(0, ie - i - 1)) + 1
                                  for _, i, ie, j, je in mapping if j <= line - 1 < je), line)
            start, end, target = original_line - 6, original_line + 6, line
        contexts.append(dict(code=issue['code'],
            original_branch=_source_excerpt(original, start, end),
            candidate_context=_source_excerpt(candidate, target - 8, target + 12),
            note='Original and candidate line numbers are separate. Use ORIGINAL coordinates for replacement edits. If an excerpt is incomplete, read the remaining original lines before revising.'))
    # Related local evaluation calls are evidence only. Empty paths do not imply
    # that execution is skipped; the model must inspect the real call guards.
    evaluation_names = {n.name for n in ast.walk(new) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and any(word in n.name.lower() for word in ('test', 'eval', 'corrupt'))}
    calls = [n for n in ast.walk(new) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id in evaluation_names]
    evaluations = [_source_excerpt(candidate, n.lineno - 2, n.end_lineno + 2, limit=8) for n in calls[:8]]
    return dict(findings=contexts, candidate_evaluation_calls=evaluations,
        instruction='Preserve existing loader branches, class counts and unrelated string values. Restore accidental double-escaping at the specific reported literals; do not globally unescape code. Submit the COMPLETE set of intended edits against the original file, not only the last rejected change. If evaluation is unavailable, inspect and guard the actual calls and result logging; an empty path alone does not skip evaluation. Do not invent paths or benchmarks. Source excerpts are evidence, not automatic code changes.')


def dataset_edit_context(source, dataset_name):
    """Source/API evidence before editing, using the same review helpers."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return dict(status='unavailable', note='Original source does not parse; inspect it before editing.')
    calls = _loader_calls(tree)
    blocks, seen = [], set()
    for _, call in calls:
        enclosing = [n for n in ast.walk(tree) if isinstance(n, ast.If) and n.lineno <= call.lineno <= n.end_lineno]
        if enclosing:
            block = min(enclosing, key=lambda n: n.end_lineno - n.lineno)
            span = block.lineno, block.end_lineno
        else:
            span = max(1, call.lineno - 2), call.end_lineno + 2
        if span not in seen:
            blocks.append(_source_excerpt(source, *span))
            seen.add(span)
        if len(blocks) == 3:
            break
    canonical = lambda name: ''.join(c for c in name.lower() if c.isalnum())
    signatures = _installed_signatures()
    api = [dict(loader=name, accepted_keywords=sorted(params - {'self'}), source_file=path)
           for name, (params, path) in signatures.items() if canonical(name) == canonical(dataset_name)]
    return dict(existing_loaders=sorted({name for name, _ in calls}),
        original_loader_blocks=blocks, installed_target_api=api,
        instruction='These are ORIGINAL coordinates. Preserve all existing loader bodies and class counts. Prefer inserting a new elif BEFORE an existing else; retain that else and its body, rather than rewriting unrelated branches. Use the installed API evidence when available; matching input shape does not prove augmentation/evaluation compatibility. This is source evidence, not a generated patch or runtime test.')


def review_dataset_change(original, candidate, signatures=None):
    """Return evidence-backed warnings. An empty report is not a runtime pass."""
    try:
        old, new = ast.parse(original), ast.parse(candidate)
    except SyntaxError:
        return dict(status='unavailable', issues=[],
            scope='Source comparison could not parse both versions; no logic or runtime certification.')
    old_calls, new_calls = _loader_calls(old), _loader_calls(new)
    issues = []
    for name in sorted({n for n, _ in old_calls} - {n for n, _ in new_calls}):
        issues.append(dict(code='existing_loader_removed',
            line=next(n.lineno for label, n in old_calls if label == name),
            message=f'Original torchvision loader {name} no longer appears in this file. Restore existing support or explain and verify a move to another file.'))
    baseline = {v['message'].split(' at line ')[0] for v in _overwrites(old)}
    issues.extend(v for v in _overwrites(new) if v['message'].split(' at line ')[0] not in baseline)
    signatures = _installed_signatures() if signatures is None else signatures
    old_bad = set()
    for calls, report in ((old_calls, False), (new_calls, True)):
        for name, call in calls:
            if name not in signatures:
                continue
            accepted, source = signatures[name]
            for keyword in call.keywords:
                if keyword.arg is None or keyword.arg in accepted:
                    continue
                key = (name, keyword.arg)
                if not report:
                    old_bad.add(key)
                elif key not in old_bad:
                    issues.append(dict(code='unsupported_loader_keyword', line=call.lineno,
                        message=f'{name} does not accept keyword {keyword.arg!r} in the installed constructor. Accepted keywords: {", ".join(sorted(accepted - {"self"}))}.',
                        evidence=source))
    issues.extend(_escape_changes(old, new))
    return dict(status='issues_found' if issues else 'no_findings', issues=issues,
        repair_context=_repair_context(original, candidate, old, new, issues) if issues else None,
        scope='Static source comparison and available installed constructor signatures only; runtime, augmentations and evaluation compatibility remain unverified.')
