"""list_datasets: lists the datasets a cloned repo supports (details in the tool docstring below).
Terminal: python list_datasets.py <repo_name>"""

import ast
import os
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

from langchain_core.tools import tool

REPOS_DIR = Path(__file__).resolve().parent / "repos"
SKIP_DIRS = {"venv", "env", "node_modules", "__pycache__", "site-packages", "build", "dist"}

# Kinds of evidence, strongest first, with the label shown to the user.
# The first three mean "supported"; the others are weaker.
KINDS = {"loaded": "loaded", "choices": "argparse choices", "list": "dataset list",
         "branch": "if-check", "default": "argparse default"}
STRONG = ("loaded", "choices", "list")

NAME_WORDS = {"dataset", "dset", "dataset_name", "dataset_type", "data_name"}   # args.dataset, ...
LIST_VAR = re.compile(r"(^|_)datasets?(_names?|_list|_choices)?$")               # DATASETS = [...]
NOT_DATASETS = {"ImageFolder", "DatasetFolder", "VisionDataset", "FakeData", "Dataset", "DatasetDict"}
NOT_NAMES = {"train", "test", "val", "valid", "validation", "eval", "cpu", "cuda"}
README_NAMES = re.compile(   # well-known names, longest first
    r"(?<![a-z])(fashion[-_ ]?mnist|mnist|cifar[-_ ]?100|cifar[-_ ]?10|svhn|stl[-_ ]?10|"
    r"tiny[-_ ]?imagenet|imagenet|gtsrb|celeba)", re.I)


# ---------- helpers ----------

def normalize(name):
    """CIFAR-10, CIFAR10 and cifar_10 all become 'cifar10'."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def words(node):
    """An expression as words: args.dataset.lower() -> ['args', 'dataset']."""
    return re.findall(r"\w+", re.sub(r"\.(lower|upper|strip)\(\)", "", ast.unparse(node)))


def strings(node):
    """The string constants in a literal: 'a', ['a', 'b'] or {'a': ...}."""
    items = node.elts if isinstance(node, (ast.List, ast.Tuple, ast.Set)) else \
        node.keys if isinstance(node, ast.Dict) else [node]
    return [i.value for i in items if isinstance(i, ast.Constant) and isinstance(i.value, str)]


def is_dataset_var(node):
    """True for args.dataset, opt.dataset.lower(), cfg['dataset']['name'], dataset_name, ..."""
    w = [] if isinstance(node, ast.Constant) else [x.lower() for x in words(node)]
    return bool(w) and bool({w[-1], "_".join(w[-2:])} & NAME_WORDS)


# ---------- scanning ----------

def scan_python(path, rel, found):
    """Record every dataset name found in one Python file."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # the scanned repo's own warnings are not ours
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except (SyntaxError, ValueError):
        return                                # e.g. Python 2 code

    def add(name, kind, node):
        if re.fullmatch(r"[A-Za-z][\w\-.]{1,30}", name) and normalize(name) not in NOT_NAMES:
            found[normalize(name)][kind].append((rel, node.lineno))

    # names bound to torchvision.datasets or its classes (aliases included)
    imported = {a.asname or a.name.split(".")[-1]
                for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
                for a in n.names if "datasets" in f"{getattr(n, 'module', '')}.{a.name}"}

    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):                      # if args.dataset == "cifar10"
            sides = [node.left] + node.comparators
            for op, a, b in zip(node.ops, sides, sides[1:]):
                # `'x' in args.dataset` is a substring test, so only `args.dataset in [...]` counts
                for var, values in [(a, b)] if isinstance(op, (ast.In, ast.NotIn)) else [(a, b), (b, a)]:
                    if is_dataset_var(var):
                        for name in strings(values):
                            add(name, "branch", node)

        elif isinstance(node, (ast.Assign, ast.AnnAssign)):    # DATASETS = ["cifar10", "stl10"]
            targets = getattr(node, "targets", None) or [node.target]
            if isinstance(node.value, (ast.List, ast.Tuple, ast.Set, ast.Dict)) \
                    and any(LIST_VAR.search((words(t) or [""])[-1].lower()) for t in targets):
                for name in strings(node.value):
                    add(name, "list", node)

        elif isinstance(node, ast.Dict):                       # {"dataset": ["cifar10", "gtsrb"]}
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and isinstance(key.value, str) \
                        and LIST_VAR.search(key.value.lower()) and isinstance(value, (ast.List, ast.Tuple, ast.Set)):
                    for name in strings(value):
                        add(name, "list", node)

        elif isinstance(node, ast.Call):
            w = words(node.func)
            if w[-1:] == ["load_dataset"] and node.args:       # HuggingFace
                for name in strings(node.args[0]):
                    add(name, "loaded", node)
            elif w and w[-1][:1].isupper() and w[-1] not in NOT_DATASETS and (w[0] in imported or "datasets" in w):
                add(w[-1], "loaded", node)                     # datasets.CIFAR10(...)
            elif w[-1:] == ["add_argument"]:                   # parser.add_argument("--dataset", ...)
                flags = {f.lstrip("-").replace("-", "_").lower() for a in node.args for f in strings(a)}
                if flags & {"dataset", "dataset_name", "data_name"}:
                    for kw in node.keywords:
                        if kw.arg in ("choices", "default"):
                            for name in strings(kw.value):
                                add(name, kw.arg, node)


def scan_repo(root):
    """Returns (dataset evidence found in code, dataset names mentioned in READMEs, files scanned)."""
    found = defaultdict(lambda: defaultdict(list))   # dataset -> kind -> [(file, line)]
    mentioned = defaultdict(set)                      # dataset -> {README files}
    count = 0
    for folder, subfolders, files in os.walk(root):   # os.walk never follows symlinked folders
        subfolders[:] = sorted(d for d in subfolders if d not in SKIP_DIRS and not d.startswith("."))
        for name in sorted(files):
            path = Path(folder) / name
            if path.is_symlink() or path.stat().st_size > 1_000_000:
                continue
            rel = path.relative_to(root).as_posix()
            if name.endswith(".py"):
                scan_python(path, rel, found)
            elif name.lower().startswith("readme"):
                for m in README_NAMES.finditer(path.read_text(encoding="utf-8", errors="ignore")):
                    mentioned[normalize(m.group(1))].add(rel)
            else:
                continue
            count += 1
            if count >= 3000:                         # safety limit for huge repos
                return found, mentioned, count
    return found, mentioned, count


# ---------- report ----------

def where(evidence):
    """One example location per kind of evidence, strongest first (at most three)."""
    return ", ".join([f"{evidence[k][0][0]}:{evidence[k][0][1]} ({KINDS[k]})" for k in KINDS if k in evidence][:3])


def make_report(repo, found, mentioned, count):
    supported = {d: e for d, e in found.items() if any(k in e for k in STRONG)}
    unsure = {d: e for d, e in found.items() if d not in supported}
    readme_only = {d: files for d, files in mentioned.items() if d not in found}

    lines = [f"Repo '{repo}' ({count} files scanned)", "", "SUPPORTED (declared or loaded in code):"]
    lines += [f"  - {d}: {where(e)}" for d, e in sorted(supported.items())] or ["  (none found)"]
    if unsure:
        lines += ["", "NAMED IN CODE BUT NO LOADER FOUND (check by hand):"]
        lines += [f"  - {d}: {where(e)}" for d, e in sorted(unsure.items())]
    if readme_only:
        lines += ["", "ONLY MENTIONED IN README (not found in code, so treat as unsupported):"]
        lines += [f"  - {d}: {', '.join(sorted(f)[:2])}" for d, f in sorted(readme_only.items())]
    if not found:
        lines += ["", "Note: no dataset code was recognised. The repo may load data with custom "
                      "code, so read its data files directly."]
    return "\n".join(lines)


# ---------- the tool ----------

@tool
def list_datasets(repo_name: str) -> str:
    """Lists the datasets a cloned repository supports.

    Input: repo_name, the name of a folder inside the local 'repos' folder.
    Returns: the datasets the repo's code declares or loads (supported), datasets that are
    only named in code with no loader found, and datasets only mentioned in the README.
    Read-only: it never runs the repo's code. Returns an error message if the repo is not found.
    """
    name = repo_name.strip().rstrip("/")
    if not name or Path(name).name != name or name in (".", ".."):   # a folder name, not a path
        return "Error: repo_name must be a folder name inside 'repos', not a path."
    root = REPOS_DIR / name
    if root.is_symlink() or not root.is_dir():
        available = sorted(p.name for p in REPOS_DIR.iterdir() if p.is_dir()) if REPOS_DIR.is_dir() else []
        return f"Error: repo '{name}' not found in repos/. Available repos: {', '.join(available) or 'none'}."
    return make_report(name, *scan_repo(root))[:6000]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python list_datasets.py <repo_name>")
    print(list_datasets.invoke({"repo_name": sys.argv[1]}))
