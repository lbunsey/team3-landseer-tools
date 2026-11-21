"""Bounded, read-only workflow evidence; candidates are not proven entry points."""
import ast
import os
import re

from .inspect_repo import get_repo_root
from . import patch_utils


def discover_workflow(repo_name):
    root = get_repo_root(repo_name)
    patch_utils._reject_link(root)
    files, documents, python = [], [], {}
    total = 0
    truncated = False
    def safe_directory(path):
        try:
            patch_utils._reject_link(path)
            return True
        except (OSError, ValueError):
            return False
    for folder, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not d.startswith('.') and d not in
                         {'data', 'datasets', 'venv', 'node_modules', '__pycache__', 'result', 'results'}
                         and safe_directory(root / os.path.relpath(folder, root) / d))
        for name in sorted(names):
            path = root / os.path.relpath(folder, root) / name
            relative = path.relative_to(root).as_posix()
            if name.startswith('.') or path.suffix.lower() not in {'.py', '.sh', '.bash', '.ps1', '.md', '.rst'}:
                continue
            try:
                patch_utils._reject_link(path)
                if path.stat().st_size > 500_000:
                    continue
                if len(files) >= 1000 or total + path.stat().st_size > 2_000_000:
                    truncated = True
                    continue
                text = path.read_text(encoding='utf-8-sig')
            except (OSError, ValueError, UnicodeError):
                continue
            files.append(relative)
            total += len(text.encode('utf-8'))
            if path.suffix == '.py':
                try:
                    python[relative] = ast.parse(text)
                except SyntaxError:
                    continue
            else:
                documents.append((relative, text))
    commands = []
    for relative, text in documents:
        for number, line in enumerate(text.splitlines(), 1):
            if re.search(r'\b(python[\d.]*|torchrun|accelerate|bash|sh)\b', line):
                commands.append(dict(file_path=relative, line=number, text=line[:350]))
    candidates = []
    for relative, tree in python.items():
        evidence = []
        for node in tree.body:
            if isinstance(node, ast.If) and '__name__' in ast.unparse(node.test) and '__main__' in ast.unparse(node.test):
                evidence.append(dict(line=node.lineno, kind='execution_guard', text=ast.unparse(node.test)))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                call = ast.unparse(node.func)
                if call.endswith(('.parse_args', '.parse_known_args')):
                    evidence.append(dict(line=node.lineno, kind='cli_parser', text=call))
        refs = [c for c in commands if relative in c['text'] or
                (relative.rsplit('/', 1)[-1] in c['text'])]
        if not evidence and not refs:
            continue
        calls = [dict(line=n.lineno, call=ast.unparse(n.func)) for n in ast.walk(tree)
                 if isinstance(n, ast.Call) and any(word in ast.unparse(n.func).lower()
                 for word in ('dataset', 'loader', 'transform', 'train', 'test', 'eval', 'model', 'loss'))]
        candidates.append(dict(file_path=relative, evidence=evidence[:8],
                               command_references=refs[:5], relevant_calls=calls[:16],
                               imports=[dict(line=n.lineno, text=ast.unparse(n)) for n in tree.body
                                        if isinstance(n, (ast.Import, ast.ImportFrom))][:20]))
    candidates.sort(key=lambda c: (-len(c['command_references']), -len(c['evidence']), c['file_path']))
    return dict(candidates=candidates[:8], execution_examples=commands[:20],
                files_scanned=len(files), truncated=truncated,
                note='Static entry candidates, not proof of execution. No code imported or executed.',
                next_steps=['Read relevant README/scripts and complete candidate source with inspect_repo.',
                            'Choose the entry for the requested task using command/call evidence, never a guessed filename.',
                            'Trace entry -> loader/split -> transforms/wrapper -> model/loss -> evaluation; follow imports into helper modules.',
                            'Save known analysis incrementally; pending items do not block requested proposals. Report unknown links instead of inventing evidence.'])
