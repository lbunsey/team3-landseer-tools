import json
import os
import re
import sys
from pathlib import Path

from langchain_core.tools import tool

# Same folder that clone_repo.py clones into.
REPOS_DIR = Path(__file__).resolve().parent.parent / "repos"

SKIP_DIRS = {".git", "node_modules", "venv", ".venv", "__pycache__", ".idea", ".vscode"}
SKIP_FILES = {".env"}
SKIP_SUFFIXES = {".pem", ".key"}

MAX_FILES = 300          # max files shown by list_repo_files
MAX_MATCHES = 50         # max hits returned by search_repo
MAX_CHARS = 20000        # max characters returned by read_repo_file
MAX_FILE_BYTES = 1_000_000


def _repo_path(repo, rel=""):
    """Return (repo_root, target). Raises ValueError if the repo is missing or the path escapes it."""
    root = (REPOS_DIR / repo).resolve()
    if root.parent != REPOS_DIR.resolve() or not root.is_dir():
        raise ValueError(f"Repo '{repo}' not found. Clone it first.")
    target = (root / rel).resolve()
    if target != root and root not in target.parents:
        raise ValueError("Path is outside the repo.")
    return root, target


def _skipped(path, root):
    rel = path.relative_to(root)
    return (
        any(part in SKIP_DIRS for part in rel.parts)
        or rel.name in SKIP_FILES
        or rel.suffix in SKIP_SUFFIXES
    )


def _iter_files(root, max_depth=None):
    """Yield readable files in the repo, skipping junk folders and secret files."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        depth = len(Path(dirpath).relative_to(root).parts)
        if max_depth is not None and depth >= max_depth:
            dirnames[:] = []
            continue
        for name in sorted(filenames):
            path = Path(dirpath) / name
            if not _skipped(path, root):
                yield path


def _read_text(path):
    """Return the file's text, or None if it looks binary. Notebooks become plain code/markdown."""
    with open(path, "rb") as f:
        data = f.read(50_000_000 if path.suffix == ".ipynb" else MAX_FILE_BYTES)
    if b"\0" in data[:1024]:
        return None
    text = data.decode("utf-8", errors="replace")
    if path.suffix == ".ipynb":
        try:
            cells = json.loads(text)["cells"]
            return "\n\n".join(
                f"# [{c.get('cell_type')}]\n" + "".join(c.get("source", [])) for c in cells
            )
        except (ValueError, KeyError, TypeError):
            pass  # not valid notebook JSON, fall back to raw text
    return text


@tool
def list_repo_files(repo: str, max_depth: int = 3) -> str:
    """List the files in a cloned repository, skipping .git, venv and similar folders.

    Input: the folder name returned by clone_repo (for example 'my-repo'),
    and optionally max_depth (how many folder levels to show, default 3).
    Returns one relative file path per line.
    """
    try:
        root, _ = _repo_path(repo)
    except ValueError as e:
        return f"Error: {e}"

    files = [p.relative_to(root).as_posix() for p in _iter_files(root, max_depth)]
    if not files:
        return "No files found."
    shown = files[:MAX_FILES]
    note = f"\n(showing {MAX_FILES} of {len(files)} files; use a smaller max_depth or search_repo)" if len(files) > MAX_FILES else ""
    return "\n".join(shown) + note


@tool
def read_repo_file(repo: str, path: str, start_line: int = 1, end_line: int = 300) -> str:
    """Read a text file from a cloned repository, with line numbers.

    Input: the repo folder name, the file path inside the repo (for example 'models/__init__.py'),
    and optionally start_line and end_line to read a section of a long file (default lines 1-300).
    Jupyter notebooks are shown as their code and markdown cells.
    """
    try:
        root, target = _repo_path(repo, path)
    except ValueError as e:
        return f"Error: {e}"
    if not target.is_file():
        return f"Error: file '{path}' not found in {repo}."
    if _skipped(target, root):
        return "Error: this file is not allowed to be read."

    text = _read_text(target)
    if text is None:
        return "This is a binary file, so it is not shown."

    lines = text.splitlines()
    start = max(start_line, 1)
    chunk = lines[start - 1:end_line]
    if not chunk:
        return f"No lines in that range. The file has {len(lines)} lines."

    out = "\n".join(f"{start + i}: {line}" for i, line in enumerate(chunk))
    header = f"{path} (lines {start}-{start + len(chunk) - 1} of {len(lines)})\n"
    if len(out) > MAX_CHARS:
        out = out[:MAX_CHARS] + "\n... (cut off; read a smaller line range)"
    return header + out


@tool
def search_repo(repo: str, pattern: str) -> str:
    """Search every text file in a cloned repository for a regular expression (case-insensitive).

    Input: the repo folder name and a pattern such as 'get_model|--model|resnet\\d+'.
    Returns matching lines as 'path:line_number: text'. Use read_repo_file to see more around a match.
    """
    try:
        root, _ = _repo_path(repo)
        regex = re.compile(pattern, re.IGNORECASE)
    except ValueError as e:
        return f"Error: {e}"
    except re.error as e:
        return f"Error: invalid pattern: {e}"

    hits = []
    for path in _iter_files(root):
        text = _read_text(path)
        if text is None:
            continue
        rel = path.relative_to(root).as_posix()
        for number, line in enumerate(text.splitlines(), 1):
            if regex.search(line):
                hits.append(f"{rel}:{number}: {line.strip()[:200]}")
                if len(hits) >= MAX_MATCHES:
                    return "\n".join(hits) + f"\n(stopped at {MAX_MATCHES} matches; use a more specific pattern)"
    return "\n".join(hits) or "No matches."


READ_TOOLS = [list_repo_files, read_repo_file, search_repo]


# Quick test:  python read_repo.py NIPS2022_EP_BNP "get_model|--model"
if __name__ == "__main__":
    repo = sys.argv[1]
    print(list_repo_files.invoke({"repo": repo}))
    if len(sys.argv) > 2:
        print("\n--- search ---")
        print(search_repo.invoke({"repo": repo, "pattern": sys.argv[2]}))
