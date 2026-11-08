"""list_attacks: lists the attacks (backdoor or adversarial) a cloned repo supports.
Terminal: python -m tools.list_attacks <repo_name>"""
 
import ast
import os
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path
 
from langchain_core.tools import tool
 
from .list_datasets import normalize, strings, words   # same name-matching helpers, same conventions
 
REPOS_DIR = Path(__file__).resolve().parent.parent / "repos"
SKIP_DIRS = {"venv", "env", "node_modules", "__pycache__", "site-packages", "build", "dist"}
 
# Kinds of evidence, strongest first. Unlike datasets there is no loader call to look for:
# the if/elif chain on the attack name usually IS the implementation, so it counts as support.
KINDS = {"choices": "argparse choices", "list": "attack list", "branch": "if-check", "default": "argparse default"}
STRONG = ("choices", "list", "branch")
 
# Variables, flags and dict keys that hold an attack name: args.attack_type, --poison-type, triggerType, ...
NAME_WORDS = {"attack", "attack_type", "attack_name", "attack_method", "attack_mode", "poison_type",
              "poison_method", "backdoor_type", "trigger_type", "target_type"}
LIST_VAR = re.compile(r"(^|_)(attacks?|poison(er|ers)?(_types?|_methods?)?|backdoors?|triggers?)"
                      r"(_names?|_list|_choices|_dict|_registry|_map)?$")   # ATTACKS = [...], poison_types = {...}
NOT_NAMES = {"none", "benign", "clean", "no", "off", "false", "true", "default", "all", "train", "test", "eval",
             "targeted", "untargeted", "onlyoriginal", "original", "standard"}   # modes and values, not attacks
NOT_CLASSES = {"attack", "attacks", "attacker"}
SKIP_CLASS = re.compile(r"(base|abstract|wrapper|mixin|errors?|exceptions?|utils?|helpers?|config|dataset|loader|model|net|"
                        r"module|layer|block|logger|meter|manager)s?$", re.I)
MAX_LEADS = 40
MAX_CHARS = 6000
README_NAMES = re.compile(   # well-known attacks, only used to spot README mentions
    r"(?<![a-z])(badnets|blended|wanet|issba|refool|clean[-_ ]?label|input[-_ ]?aware|narcissus|sleeper[-_ ]?agent|"
    r"trojannn|fgsm|pgd|deepfool|autoattack)(?![a-z])", re.I)
 
 
# ---------- helpers ----------
 
def snake(word):
    """triggerType -> trigger_type"""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", word).lower()
 
 
def attack_var(node):
    """The selector's name if node is args.attack_type, triggerType, cfg['attack']['name'], ... else None."""
    if isinstance(node, ast.Constant):
        return None
    w = [snake(x) for x in words(node)]
    candidates = [w[-1], "_".join(w[-2:])] if w else []
    return next((c for c in candidates if c in NAME_WORDS), None)
 
 
def in_attack_path(rel):
    """core/attacks/ISSBA.py, torchattacks/attack.py: a path with 'attack' in it, outside tests and examples."""
    parts = rel.lower().split("/")
    return any("attack" in p for p in parts) and not {"test", "tests", "example", "examples", "docs"} & set(parts)
 
 
# ---------- scanning ----------
 
def scan_python(path, rel, found, defs):
    """Record every attack name found in one Python file, and its top-level classes and functions in defs."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # the scanned repo's own warnings are not ours
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except (SyntaxError, ValueError):
        return                                # e.g. Python 2 code
 
    for node in tree.body:                                     # class FGSM(Attack): / def deepfool(...):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and not node.name.startswith("_"):
            bases = " ".join(ast.unparse(b) for b in getattr(node, "bases", []))
            defs.append((node.name, rel, node.lineno, "class" if isinstance(node, ast.ClassDef) else "function", bases))
 
    def add(name, kind, node, via):
        if re.fullmatch(r"[A-Za-z][\w\-.]{1,30}", name) and normalize(name) not in NOT_NAMES:
            found[normalize(name)][kind].append((rel, node.lineno, via))
 
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):                      # if args.attack_type == "badnets"
            sides = [node.left] + node.comparators
            for op, a, b in zip(node.ops, sides, sides[1:]):
                # `'x' in attack_type` is a substring test, so only `attack_type in [...]` counts
                for var, values in [(a, b)] if isinstance(op, (ast.In, ast.NotIn)) else [(a, b), (b, a)]:
                    if attack_var(var):
                        for name in strings(values):
                            add(name, "branch", node, attack_var(var))
 
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):    # ATTACKS = ["badnets", "blended"]
            targets = getattr(node, "targets", None) or [node.target]
            names = [snake((words(t) or [""])[-1]) for t in targets]
            # A dict only counts as a registry ({"badnets": BadNets}); {"cifar10": [files...]} maps something else.
            nested = isinstance(node.value, ast.Dict) and any(
                isinstance(v, (ast.List, ast.Tuple, ast.Set, ast.Dict)) for v in node.value.values)
            if isinstance(node.value, (ast.List, ast.Tuple, ast.Set, ast.Dict)) and not nested:
                for var in (n for n in names if LIST_VAR.search(n)):
                    for name in strings(node.value):
                        add(name, "list", node, var)
 
        elif isinstance(node, ast.Dict):                       # {"poison_type": ["badnet", "blend"]}
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and isinstance(key.value, str) and isinstance(value, (ast.List, ast.Tuple, ast.Set)) \
                        and (snake(key.value) in NAME_WORDS or LIST_VAR.search(snake(key.value))):
                    for name in strings(value):
                        add(name, "list", node, key.value)
 
        elif isinstance(node, ast.Call) and words(node.func)[-1:] == ["add_argument"]:   # parser.add_argument("--attack-type", ...)
            flags = {snake(f.lstrip("-").replace("-", "_")) for a in node.args for f in strings(a)}
            for flag in flags & NAME_WORDS:
                for kw in node.keywords:
                    if kw.arg in ("choices", "default"):
                        for name in strings(kw.value):
                            add(name, kw.arg, node, "--" + flag.replace("_", "-"))
 
 
def scan_repo(root):
    """Returns (attack evidence found in code, attack names mentioned in READMEs, definitions, files scanned)."""
    found = defaultdict(lambda: defaultdict(list))   # attack -> kind -> [(file, line, selector)]
    defs = []                                         # (name, file, line, class|function|file, base classes)
    mentioned = defaultdict(set)                      # attack -> {README files}
    count = 0
    for folder, subfolders, files in os.walk(root):   # os.walk never follows symlinked folders
        subfolders[:] = sorted(d for d in subfolders if d not in SKIP_DIRS and not d.startswith("."))
        for name in sorted(files):
            path = Path(folder) / name
            if path.is_symlink() or path.stat().st_size > 1_000_000:
                continue
            rel = path.relative_to(root).as_posix()
            if name.endswith(".py"):
                scan_python(path, rel, found, defs)
                if not name.startswith(("__", "test")):
                    defs.append((path.stem, rel, 1, "file", ""))
            elif name.lower().startswith("readme"):
                for m in README_NAMES.finditer(path.read_text(encoding="utf-8", errors="ignore")):
                    mentioned[normalize(m.group(1))].add(rel)
            else:
                continue
            count += 1
            if count >= 3000:                         # safety limit for huge repos
                return found, mentioned, defs, count
    return found, mentioned, defs, count
 
 
# ---------- report ----------
 
def where(evidence):
    """One example per kind of evidence, strongest first (at most three): file:line (kind on selector)."""
    return ", ".join([f"{evidence[k][0][0]}:{evidence[k][0][1]} ({KINDS[k]} on {evidence[k][0][2]})"
                      for k in KINDS if k in evidence][:3])
 
 
def attack_class(name, rel, kind, bases):
    """A public class in an 'attack' path whose name matches its file (fgsm.py: FGSM) or extends an ...Attack base."""
    stem, key = normalize(Path(rel).stem), normalize(name)
    return (kind == "class" and in_attack_path(rel) and not Path(rel).name.startswith("_") and len(key) > 1
            and not SKIP_CLASS.search(name) and not re.match(r"(base|abstract)", name, re.I)
            and (len(stem) > 2 and (stem in key or key in stem) or "attack" in bases.lower()))
 
 
def find_leads(defs, found, mentioned):
    """One attack-like class per file, plus any class, function or file named like an attack the README mentions."""
    leads, files = {}, set()
    for name, rel, line, kind, bases in defs:
        key = normalize(name)
        if key in found or key in leads or key in NOT_CLASSES:
            continue
        if key in mentioned or (rel not in files and attack_class(name, rel, kind, bases)):
            leads[key] = (name, rel, line, kind)
            files.add(rel)
    return leads
 
 
def make_report(repo, found, mentioned, defs, count):
    supported = {a: e for a, e in found.items() if any(k in e for k in STRONG)}
    default_only = {a: e for a, e in found.items() if a not in supported}
    leads = find_leads(defs, found, mentioned)
    readme_only = {a: files for a, files in mentioned.items() if a not in found and a not in leads}
 
    lines = [f"Repo '{repo}' ({count} files scanned)", "", "SUPPORTED (declared or branched on in code):"]
    lines += [f"  - {a}: {where(e)}" for a, e in sorted(supported.items())] or ["  (none found)"]
    if default_only:
        lines += ["", "ONLY AN ARGPARSE DEFAULT (no code branches on it; check by hand):"]
        lines += [f"  - {a}: {where(e)}" for a, e in sorted(default_only.items())]
    if leads:
        lines += ["", "DEFINED IN CODE AS A CLASS, FUNCTION OR FILE "
                      "(leads; not shown to be wired to a selector, check by hand):"]
        lines += [f"  - {n}: {rel}:{line} ({kind})" for n, rel, line, kind in sorted(leads.values())[:MAX_LEADS]]
        if len(leads) > MAX_LEADS:
            lines += [f"  (+{len(leads) - MAX_LEADS} more not shown; use search_repo or list_repo_files)"]
    if readme_only:
        lines += ["", "ONLY MENTIONED IN README (no matching code found; "
                      "it may still be implemented in a way this scan cannot see):"]
        lines += [f"  - {a}: {', '.join(sorted(f)[:2])}" for a, f in sorted(readme_only.items())]
    if not found:
        lines += ["", "Note: no attack selector was recognised. The repo may not implement attacks, or may "
                      "choose them in a way this scan cannot see (classes, decorators, config files)."]
    report = "\n".join(lines)
    if len(report) > MAX_CHARS:                   # cut on a whole line and say so
        report = report[:MAX_CHARS - 80].rsplit("\n", 1)[0] + f"\n... (output cut at {MAX_CHARS} characters)"
    return report
 
 
# ---------- the tool ----------
 
@tool
def list_attacks(repo_name: str) -> str:
    """Lists the attacks (backdoor or adversarial) a cloned repository supports.
 
    Input: repo_name, the name of a folder inside the local 'repos' folder.
    Returns: the attack names the repo's code declares (argparse choices, attack lists) or branches
    on (if-checks), each with file:line evidence and the variable or flag it was found under
    (attack_type, poison_type, trigger_type, ...); names found only as an argparse default;
    classes, functions or files that look like attacks (leads for libraries that define one class
    per attack, with no selector); and names only mentioned in the README. Output is capped at
    6,000 characters. Read-only: it never runs the repo's code.
    Returns an error message if the repo is not found.
    """
    name = repo_name.strip().rstrip("/")
    print(f"[DEBUG][list_attacks] list_attacks started: {name}", flush=True)
 
    if not name or Path(name).name != name or name in (".", "..") or len(name) > 255:   # a folder name, not a path
        return "Error: repo_name must be a folder name inside 'repos', not a path."
    root = REPOS_DIR / name
    if root.is_symlink() or not root.is_dir():
        available = sorted(p.name for p in REPOS_DIR.iterdir() if p.is_dir()) if REPOS_DIR.is_dir() else []
        return f"Error: repo '{name}' not found in repos/. Available repos: {', '.join(available) or 'none'}."
 
    try:
        found, mentioned, defs, count = scan_repo(root)
        report = make_report(name, found, mentioned, defs, count)
    except Exception as e:
        print(f"[ERROR][list_attacks] list_attacks failed: {type(e).__name__}: {e}", flush=True)
        raise
    print(f"[DEBUG][list_attacks] list_attacks finished: {count} files scanned.", flush=True)
    return report
 
 
if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m tools.list_attacks <repo_name>")
    print(list_attacks.invoke({"repo_name": sys.argv[1]}))
