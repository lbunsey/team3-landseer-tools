"""list_arguments: lists the command-line arguments (argparse) of a cloned repo's scripts.
Terminal: python -m tools.list_arguments <repo_name> [script]"""

import ast
import os
import sys
import warnings
from pathlib import Path

from langchain_core.tools import tool

from .list_datasets import strings, words   # same helpers as the other list_* tools

REPOS_DIR = Path(__file__).resolve().parent.parent / "repos"
SKIP_DIRS = {"venv", "env", "node_modules", "__pycache__", "site-packages", "build", "dist"}
MAX_CHARS = 6000


# ---------- scanning ----------

def short(text, limit):
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit - 3] + "..."


def show(node, limit):
    """A literal as Python would print it, or the source text for anything computed."""
    try:
        return short(repr(ast.literal_eval(node)), limit)
    except (ValueError, SyntaxError):
        return short(ast.unparse(node), limit)


def describe(call):
    """One add_argument(...) call as a line: flags, type/default/choices/..., help."""
    flags = [f for a in call.args for f in strings(a)]
    if not flags:
        return None                                        # add_argument(*names): nothing static to show
    kw = {k.arg: k.value for k in call.keywords if k.arg}
    parts = []
    for key in ("action", "type", "nargs", "default", "choices"):
        if key in kw:
            parts.append(f"{key}={show(kw[key], 60)}")
    if isinstance(kw.get("required"), ast.Constant) and kw["required"].value is True:
        parts.append("required")
    line = f"  {', '.join(flags)}" + (f"  [{', '.join(parts)}]" if parts else "")
    if isinstance(kw.get("help"), ast.Constant) and isinstance(kw["help"].value, str):
        line += f"  - {short(kw['help'].value, 70)}"
    return line


def scan_python(path):
    """The described add_argument lines of one Python file, in source order."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # the scanned repo's own warnings are not ours
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except (SyntaxError, ValueError):
        return []                             # e.g. Python 2 code
    calls = sorted((n for n in ast.walk(tree) if isinstance(n, ast.Call) and words(n.func)[-1:] == ["add_argument"]),
                   key=lambda n: n.lineno)
    return [line for line in map(describe, calls) if line]


def scan_repo(root, script=""):
    """Returns ({file: [argument lines]}, files scanned), limited to `script` when one is given."""
    wanted = script.strip().replace("\\", "/").removeprefix("./")
    found, count = {}, 0
    for folder, subfolders, files in os.walk(root):   # os.walk never follows symlinked folders
        subfolders[:] = sorted(d for d in subfolders if d not in SKIP_DIRS and not d.startswith("."))
        for name in sorted(files):
            path = Path(folder) / name
            if not name.endswith(".py") or path.is_symlink() or path.stat().st_size > 1_000_000:
                continue
            rel = path.relative_to(root).as_posix()
            if wanted and wanted not in (rel, name, name.removesuffix(".py")):
                continue
            count += 1
            if lines := scan_python(path):
                found[rel] = lines
            if count >= 3000:                         # safety limit for huge repos
                return found, count
    return found, count


# ---------- report ----------

def make_report(repo, found, count, script=""):
    scope = f" for '{script}'" if script else ""
    if not found:
        return (f"Repo '{repo}' ({count} files scanned): no argparse arguments found{scope}. The repo may use "
                "click/typer, Hydra or yaml configs, sys.argv, or add flags in a loop (add_argument(f'--{name}')), "
                "which this scan cannot see.")
    total = sum(len(v) for v in found.values())
    out = [f"Repo '{repo}': {total} arguments in {len(found)} script(s){scope} ({count} files scanned)"]
    hidden = []
    budget = MAX_CHARS - 700                  # leave room for the NOT SHOWN note, which must never be cut off
    for rel, lines in found.items():
        block = "\n".join([f"\n{rel}"] + lines)
        if len("\n".join(out)) + len(block) <= budget:
            out.append(block)
        elif len(out) == 1:                   # a single script that is too long on its own: show what fits
            out.append(block[:budget - len(out[0])].rsplit("\n", 1)[0] + "\n  ... (cut)")   # cut on a whole line
            hidden.append(f"{rel} (rest of {len(lines)})")
        else:
            hidden.append(f"{rel} ({len(lines)})")
    if hidden:
        named = []                            # name as many hidden scripts as fit in ~500 characters
        for h in hidden:
            if sum(len(x) + 2 for x in named) + len(h) <= 500:
                named.append(h)
        more = f", and {len(hidden) - len(named)} more" if len(hidden) > len(named) else ""
        out.append(f"\nNOT SHOWN (output limit): {', '.join(named)}{more}. Call again with script=<file> to see one.")
    return "\n".join(out)


# ---------- the tool ----------

@tool
def list_arguments(repo_name: str, script: str = "") -> str:
    """Lists the command-line arguments (argparse) that a cloned repository's scripts accept.

    Input: repo_name, the name of a folder inside the local 'repos' folder. script (optional): one
    script, as a repo-relative path ('tools/run.py') or a file name ('train.py' or 'train'); leave
    it empty to list every script that defines arguments.
    Returns: for each script, one line per argument with its flags and, where given, action, type,
    nargs, default, choices, required and a short help text. Output is capped at 6,000 characters;
    scripts that do not fit are named so you can ask for them one at a time with script=.
    Read-only: it never runs the repo's code. It cannot see click/typer, Hydra/yaml configs,
    sys.argv or flags added in a loop.
    Returns an error message if the repo is not found.
    """
    name = repo_name.strip().rstrip("/")
    print(f"[DEBUG][list_arguments] list_arguments started: {name} {script}", flush=True)

    if not name or Path(name).name != name or name in (".", "..") or len(name) > 255:   # a folder name, not a path
        return "Error: repo_name must be a folder name inside 'repos', not a path."
    root = REPOS_DIR / name
    if root.is_symlink() or not root.is_dir():
        available = sorted(p.name for p in REPOS_DIR.iterdir() if p.is_dir()) if REPOS_DIR.is_dir() else []
        return f"Error: repo '{name}' not found in repos/. Available repos: {', '.join(available) or 'none'}."

    try:
        found, count = scan_repo(root, script)
        report = make_report(name, found, count, script.strip())
    except Exception as e:
        print(f"[ERROR][list_arguments] list_arguments failed: {type(e).__name__}: {e}", flush=True)
        raise
    print(f"[DEBUG][list_arguments] list_arguments finished: {count} files scanned.", flush=True)
    return report


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit("usage: python -m tools.list_arguments <repo_name> [script]")
    print(list_arguments.invoke({"repo_name": sys.argv[1], "script": sys.argv[2] if len(sys.argv) == 3 else ""}))
