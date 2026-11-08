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
 
REPOS_DIR = Path(__file__).resolve().parent.parent / "repos"
SKIP_DIRS = {"venv", "env", "node_modules", "__pycache__", "site-packages", "build", "dist"}
 
# Kinds of evidence, strongest first, with the label shown to the user.
# The first three mean "supported"; the others are weaker.
KINDS = {"loaded": "loaded", "choices": "argparse choices", "list": "dataset list",
         "branch": "if-check", "default": "argparse default", "path": "named in an argparse flag or path"}
STRONG = ("loaded", "choices", "list")
 
NAME_WORDS = {"dataset", "dset", "dataset_name", "dataset_type", "data_name"}   # args.dataset, ...
LIST_VAR = re.compile(r"(^|_)datasets?(_names?|_list|_choices)?$")               # DATASETS = [...]
NOT_DATASETS = {"ImageFolder", "DatasetFolder", "VisionDataset", "FakeData", "Dataset", "DatasetDict"}
GENERIC_LOADERS = {"ImageFolder", "DatasetFolder"}   # read any image folder: the dataset depends on the path
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
 
def scan_python(path, rel, found, generic):
    """Record every dataset name found in one Python file (and any generic folder loader in `generic`)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # the scanned repo's own warnings are not ours
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except (SyntaxError, ValueError):
        return                                # e.g. Python 2 code
 
    def add(name, kind, node):
        if re.fullmatch(r"[A-Za-z][\w\-.]{1,30}", name) and normalize(name) not in NOT_NAMES:
            found[normalize(name)][kind].append((rel, node.lineno))
 
    # names bound to torchvision.datasets or its classes -> the real name (aliases included: IF -> ImageFolder)
    imported = {a.asname or a.name.split(".")[-1]: a.name.split(".")[-1]
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
 
        elif isinstance(node, ast.ClassDef):                   # class MySet(datasets.ImageFolder):
            for base in node.bases:
                b = words(base)
                if b and imported.get(b[-1], b[-1]) in GENERIC_LOADERS and (b[0] in imported or "datasets" in b):
                    generic[imported.get(b[-1], b[-1])].append((rel, node.lineno))
 
        elif isinstance(node, ast.Call):
            w = words(node.func)
            cls = imported.get(w[0], w[0]) if len(w) == 1 else (w[-1] if w else "")   # IF(...) means ImageFolder(...)
            if w[-1:] == ["load_dataset"] and node.args:       # HuggingFace
                for name in strings(node.args[0]):
                    add(name, "loaded", node)
            elif w and cls in GENERIC_LOADERS and (w[0] in imported or "datasets" in w):
                generic[cls].append((rel, node.lineno))        # datasets.ImageFolder(path): which dataset?
            elif w and cls[:1].isupper() and cls not in NOT_DATASETS and (w[0] in imported or "datasets" in w):
                add(cls, "loaded", node)                       # datasets.CIFAR10(...)
            elif w[-1:] == ["add_argument"]:                   # parser.add_argument("--dataset", ...)
                flags = {f.lstrip("-").replace("-", "_").lower() for a in node.args for f in strings(a)}
                if flags & {"dataset", "dataset_name", "data_name"}:
                    for kw in node.keywords:
                        if kw.arg in ("choices", "default"):
                            for name in strings(kw.value):
                                add(name, kw.arg, node)
                # a dataset name inside any flag or default path: --imagenet-r-dir, default="data/imagenet/train/"
                texts = [f for a in node.args for f in strings(a)] + \
                        [t for kw in node.keywords if kw.arg == "default" for t in strings(kw.value)]
                for m in (m for t in texts for m in README_NAMES.finditer(t)):
                    found[normalize(m.group(1))]["path"].append((rel, node.lineno))
 
 
def scan_repo(root):
    """Returns (dataset evidence found in code, dataset names mentioned in READMEs, generic loaders, files scanned)."""
    found = defaultdict(lambda: defaultdict(list))   # dataset -> kind -> [(file, line)]
    mentioned = defaultdict(set)                      # dataset -> {README files}
    generic = defaultdict(list)                       # ImageFolder -> [(file, line)]
    count = 0
    for folder, subfolders, files in os.walk(root):   # os.walk never follows symlinked folders
        subfolders[:] = sorted(d for d in subfolders if d not in SKIP_DIRS and not d.startswith("."))
        for name in sorted(files):
            path = Path(folder) / name
            if path.is_symlink() or path.stat().st_size > 1_000_000:
                continue
            rel = path.relative_to(root).as_posix()
            if name.endswith(".py"):
                scan_python(path, rel, found, generic)
            elif name.lower().startswith("readme"):
                for m in README_NAMES.finditer(path.read_text(encoding="utf-8", errors="ignore")):
                    mentioned[normalize(m.group(1))].add(rel)
            else:
                continue
            count += 1
            if count >= 3000:                         # safety limit for huge repos
                return found, mentioned, generic, count
    return found, mentioned, generic, count
 
 
# ---------- report ----------
 
def where(evidence):
    """One example location per kind of evidence, strongest first (at most three)."""
    return ", ".join([f"{evidence[k][0][0]}:{evidence[k][0][1]} ({KINDS[k]})" for k in KINDS if k in evidence][:3])
 
 
def make_report(repo, found, mentioned, generic, count):
    supported = {d: {k: v for k, v in e.items() if k != "path"}   # a path name adds nothing to real support
                 for d, e in found.items() if any(k in e for k in STRONG)}
    unsure = {d: e for d, e in found.items() if d not in supported}
    readme_only = {d: files for d, files in mentioned.items() if d not in found}
 
    lines = [f"Repo '{repo}' ({count} files scanned)", "", "SUPPORTED (declared or loaded in code):"]
    lines += [f"  - {d}: {where(e)}" for d, e in sorted(supported.items())] or ["  (none found)"]
    if unsure:
        lines += ["", "NAMED IN CODE BUT NO LOADER FOUND (check by hand):"]
        lines += [f"  - {d}: {where(e)}" for d, e in sorted(unsure.items())]
    if generic:
        lines += ["", "GENERIC FOLDER LOADERS (they read any image folder, so the dataset depends on the path; check by hand):"]
        lines += [f"  - {k}: {', '.join(f'{f}:{n}' for f, n in sorted(set(v))[:3])}" for k, v in sorted(generic.items())]
    if readme_only:
        lines += ["", "ONLY MENTIONED IN README (no matching name found in code; it may be loaded in a way this scan "
                      "cannot see, check by hand):"]
        lines += [f"  - {d}: {', '.join(sorted(f)[:2])}" for d, f in sorted(readme_only.items())]
    if not found and not generic:
        lines += ["", "Note: no dataset code was recognised. The repo may load data with custom "
                      "code, so read its data files directly."]
    return "\n".join(lines)
 
 
# ---------- the tool ----------
 
@tool
def list_datasets(repo_name: str) -> str:
    """Lists the datasets a cloned repository supports.
 
    Input: repo_name, the name of a folder inside the local 'repos' folder.
    Returns: the datasets the repo's code declares or loads (supported), datasets that are
    only named in code (an if-check, or inside a flag or default path) with no loader found,
    generic folder loaders such as ImageFolder (which dataset they read depends on the path),
    and datasets only mentioned in the README. Output is capped at 6,000 characters.
    Read-only: it never runs the repo's code. Returns an error message if the repo is not found.
    """
    name = repo_name.strip().rstrip("/")
    
    print(f"[DEBUG][list_datasets] list_datasets started: {name}", flush=True)
 
    if not name or Path(name).name != name or name in (".", "..") or len(name) > 255:   # a folder name, not a path
        return "Error: repo_name must be a folder name inside 'repos', not a path."
    root = REPOS_DIR / name
    if root.is_symlink() or not root.is_dir():
        available = sorted(p.name for p in REPOS_DIR.iterdir() if p.is_dir()) if REPOS_DIR.is_dir() else []
        return f"Error: repo '{name}' not found in repos/. Available repos: {', '.join(available) or 'none'}."
    
    print(f"[DEBUG][list_datasets] Scanning repository: {root}", flush=True)
    
    try:
        found, mentioned, generic, count = scan_repo(root)
 
        print(
            f"[DEBUG][list_datasets] Scan completed: {count} files, ",
            flush=True,
        )
 
        report = make_report(name, found, mentioned, generic, count)
        if len(report) > 6000:                            # cut on a whole line and say so
            report = report[:5920].rsplit("\n", 1)[0] + "\n... (output cut at 6000 characters)"
 
        print("[DEBUG][list_datasets] list_datasets finished.", flush=True)
        return report
 
    except Exception as e:
        print(
            f"[ERROR][list_datasets] list_datasets failed: "
            f"{type(e).__name__}: {e}",
            flush=True,
        )
        raise
 
 
if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python list_datasets.py <repo_name>")
    print(list_datasets.invoke({"repo_name": sys.argv[1]}))
