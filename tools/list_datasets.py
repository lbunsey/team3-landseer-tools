"""list_datasets: lists the datasets a cloned repo supports (details in the tool docstring below).
Terminal: python list_datasets.py <repo_name>"""
 
import ast
import os
import re
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path
 
from langchain_core.tools import tool
 
REPOS_DIR = Path(__file__).resolve().parent.parent / "repos"
SKIP_DIRS = {"venv", "env", "node_modules", "__pycache__", "site-packages", "build", "dist"}
 
# Kinds of evidence, strongest first, with the label shown to the user.
# The kinds in STRONG mean "supported"; a dataset with only the others is not printed.
KINDS = {"loaded": "loaded", "folder": "loaded from a folder named for it", "choices": "argparse choices",
         "list": "dataset list", "branch_load": "loaded inside an if-check",
         "dir": "folder or path the code takes", "branch": "if-check", "default": "argparse default",
         "path": "named in an argparse flag or path"}
STRONG = ("loaded", "folder", "choices", "list", "branch_load", "dir")
 
NAME_WORDS = {"dataset", "dset", "dataset_name", "dataset_type", "data_name"}   # args.dataset, ...
LIST_VAR = re.compile(r"(^|_)datasets?(_names?|_list|_choices)?$")               # DATASETS = [...]
NOT_DATASETS = {"ImageFolder", "DatasetFolder", "VisionDataset", "FakeData", "Dataset", "DatasetDict"}
GENERIC_LOADERS = {"ImageFolder", "DatasetFolder"}   # read any image folder: the dataset depends on the path
NOT_NAMES = {"train", "test", "val", "valid", "validation", "eval", "cpu", "cuda"}
FLAG_KINDS = {"choices", "list", "branch", "default"}   # evidence where the code spells the value a user would pass
VARIANT_WORDS = {"target_attr", "target_attribute", "subset", "variant"}   # flags that pick a variant of a dataset
VARIANT_NAMES = re.compile(   # shifted or corrupted copies: CIFAR-10-C, ImageNet-R, imagenet_a, CIFAR-10-C-Bar
    r"(?<![a-z0-9])((?:cifar[-_ ]?(?:10|100)|tiny[-_ ]?imagenet|imagenet)[-_ ](?:c|p|r|a|o)(?:[-_ ]bar)?)(?![a-z])", re.I)
README_NAMES = re.compile(   # well-known names, longest first
    r"(?<![a-z])(fashion[-_ ]?mnist|mnist|cifar[-_ ]?100|cifar[-_ ]?10|svhn|stl[-_ ]?10|"
    r"tiny[-_ ]?imagenet|imagenet|gtsrb|celeba)", re.I)
# A call that loads data: load_acs_data(), get_dataset(), DataLoader(...), pd.read_csv(...). Not model weights or configs.
LOADER_CALL = re.compile(r"^(?:load|get|read|build|make|prepare|fetch)_.*(?:data|datasets?|set|loaders?|images|samples)$"
                         r"|dataset|dataloader|read_csv|read_parquet|read_table", re.I)
NOT_LOADER = re.compile(r"state_dict|model|weight|checkpoint|ckpt|config|param|yaml|json", re.I)
LOADER_DEF = re.compile(r"^(?:load|get|read|build|make|prepare)_(.+?)_(?:data|datasets?)$", re.I)   # def load_adult_data
NOT_DATA_WORDS = {"train", "training", "test", "testing", "val", "valid", "validation", "eval", "all", "raw", "batch",
                  "image", "images", "tabular", "text", "custom", "clean", "poisoned", "poison", "backdoor", "new",
                  "full", "small", "subset", "sample", "samples", "default", "local", "user", "my", "mixed"}
DIR_SUFFIX = re.compile(r"[-_ ](?:dir|path|root|folder)$", re.I)
OUTPUT_FLAG = re.compile(r"save|out|log|result|ckpt|checkpoint|model|cache|tmp|work", re.I)   # flags that write, not read
 
 
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
 
 
def named_in(text):
    """Dataset names inside a path or flag string: 'data/tiny-imagenet-200/' -> ['tiny-imagenet']."""
    return [m.group(1) for rx in (VARIANT_NAMES, README_NAMES) for m in rx.finditer(text)]
 
 
def exact_names(text, is_flag=False, plain=True):
    """Dataset names that a flag or a path names as a whole folder: 'data/imagenet_r/' -> ['imagenet_r'],
    '--imagenet-r-dir' -> ['imagenet-r']. plain=False keeps only shifted copies (CIFAR-10-C, ImageNet-R)."""
    parts = [DIR_SUFFIX.sub("", text.lstrip("-"))] if is_flag else re.split(r"[/\\]+", text)
    found = []
    for part in (p.strip() for p in parts):
        for cand in (part, re.sub(r"[-_]\d+$", "", part)):       # tiny-imagenet-200 -> tiny-imagenet
            if VARIANT_NAMES.fullmatch(cand) or (plain and README_NAMES.fullmatch(cand)):
                found.append(cand)
                break
    return found
 
 
def is_variant_var(node):
    """True for args.target_attr, cfg.subset, variant, ..."""
    w = [] if isinstance(node, ast.Constant) else [x.lower() for x in words(node)]
    return bool(w) and w[-1] in VARIANT_WORDS
 
 
def is_dataset_var(node):
    """True for args.dataset, opt.dataset.lower(), cfg['dataset']['name'], dataset_name, ..."""
    w = [] if isinstance(node, ast.Constant) else [x.lower() for x in words(node)]
    return bool(w) and bool({w[-1], "_".join(w[-2:])} & NAME_WORDS)
 
 
# ---------- scanning ----------
 
def scan_python(path, rel, found, generic, variants):
    """Record every dataset name found in one Python file (generic folder loaders go in `generic`,
    values of flags like target_attr in `variants`)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # the scanned repo's own warnings are not ours
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except (SyntaxError, ValueError):
        return                                # e.g. Python 2 code
 
    def add(name, kind, node):
        if re.fullmatch(r"[A-Za-z][\w\-.]{1,30}", name) and normalize(name) not in NOT_NAMES:
            found[normalize(name)][kind].append((rel, node.lineno))
            found[normalize(name)]["spelling"].append((name, kind))   # as written, e.g. bank_marketing
 
    # names bound to torchvision.datasets or its classes -> the real name (aliases included: IF -> ImageFolder)
    imported = {a.asname or a.name.split(".")[-1]: a.name.split(".")[-1]
                for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
                for a in n.names if "datasets" in f"{getattr(n, 'module', '')}.{a.name}"}
 
    func_of = {}                                               # node -> name of the function it sits in (nearest wins)
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for n in ast.walk(fn):
                func_of[id(n)] = fn.name
 
    def add_variant(flag, name, node):
        if re.fullmatch(r"[A-Za-z][\w\-.]{1,30}", name) and normalize(name) not in NOT_NAMES:
            variants[(flag, func_of.get(id(node), ""), rel)][name].append((rel, node.lineno))
 
    def is_loader_call(call):
        """True for load_acs_data(...), DataLoader(...), datasets.CelebA(...), ImageFolder(...)."""
        w = words(call.func)
        if not w:
            return False
        cls = imported.get(w[0], w[0]) if len(w) == 1 else w[-1]
        torchvision = w[0] in imported or "datasets" in w
        return (bool(LOADER_CALL.search(w[-1])) and not NOT_LOADER.search(w[-1])) or \
            (torchvision and (cls in GENERIC_LOADERS or (cls[:1].isupper() and cls not in NOT_DATASETS)))
 
    def compared(cmp):
        """The dataset names a comparison tests: args.dataset == "cifar10" -> ["cifar10"]."""
        sides, names = [cmp.left] + cmp.comparators, []
        for op, a, b in zip(cmp.ops, sides, sides[1:]):
            for var, values in [(a, b)] if isinstance(op, (ast.In, ast.NotIn)) else [(a, b), (b, a)]:
                if is_dataset_var(var):
                    names += strings(values)
        return names
 
    for node in ast.walk(tree):
        if isinstance(node, ast.If):                           # if args.dataset == "acs": return load_acs_data()
            calls = [c for s in node.body for c in ast.walk(s) if isinstance(c, ast.Call)]
            if any(is_loader_call(c) for c in calls):          # the branch itself loads data, so the dataset is supported
                for cmp in (c for c in ast.walk(node.test) if isinstance(c, ast.Compare)):
                    for name in compared(cmp):
                        add(name, "branch_load", cmp)
 
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):   # def load_census_income_kdd_data(...)
            m = LOADER_DEF.match(node.name)
            if m and not all(w in NOT_DATA_WORDS for w in m.group(1).lower().split("_")):
                add(m.group(1), "loaded", node)
 
        elif isinstance(node, ast.Compare):                    # if args.dataset == "cifar10"
            sides = [node.left] + node.comparators
            for op, a, b in zip(node.ops, sides, sides[1:]):
                # `'x' in args.dataset` is a substring test, so only `args.dataset in [...]` counts
                for var, values in [(a, b)] if isinstance(op, (ast.In, ast.NotIn)) else [(a, b), (b, a)]:
                    if is_dataset_var(var):
                        for name in strings(values):
                            add(name, "branch", node)
                    elif is_variant_var(var):                  # if target_attr == "income"
                        for name in strings(values):
                            add_variant(words(var)[-1].lower(), name, node)
 
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
                paths = [s for a in node.args[:1] for s in strings(a)] + \
                        [s for kw in node.keywords if kw.arg == "root" for s in strings(kw.value)]
                for name in (n for s in paths for n in named_in(s)):             # ImageFolder('data/tiny-imagenet-200/')
                    found[normalize(name)]["folder"].append((rel, node.lineno))
                    found[normalize(name)]["spelling"].append((name, "folder"))
            elif w and cls[:1].isupper() and cls not in NOT_DATASETS and (w[0] in imported or "datasets" in w):
                add(cls, "loaded", node)                       # datasets.CIFAR10(...)
            elif w[-1:] == ["add_argument"]:                   # parser.add_argument("--dataset", ...)
                flags = {f.lstrip("-").replace("-", "_").lower() for a in node.args for f in strings(a)}
                if flags & {"dataset", "dataset_name", "data_name"}:
                    for kw in node.keywords:
                        if kw.arg in ("choices", "default"):
                            for name in strings(kw.value):
                                add(name, kw.arg, node)
                for flag in sorted(flags & VARIANT_WORDS):                       # --target_attr, default="income"
                    for kw in node.keywords:
                        if kw.arg in ("choices", "default"):
                            for name in strings(kw.value):
                                add_variant(flag, name, node)
                # a flag or default path that is a dataset's folder (--imagenet-r-dir, default "data/imagenet_r/"):
                # the code takes that dataset as input. Flags that write output (--save_dir) do not count.
                raw_flags = [f for a in node.args for f in strings(a)]
                dirs = [n for f in raw_flags for n in exact_names(f, is_flag=True)]
                if not any(OUTPUT_FLAG.search(f) for f in raw_flags) and not flags & NAME_WORDS:   # --dataset's default is a name, not a path
                    dirs += [n for kw in node.keywords if kw.arg == "default" for t in strings(kw.value)
                             if re.search(r"[/\\]", t) for n in exact_names(t)]
                for name in dirs:
                    found[normalize(name)]["dir"].append((rel, node.lineno))
                    found[normalize(name)]["spelling"].append((name, "dir"))
                # a dataset name inside any flag or default path: --imagenet-r-dir, default="data/imagenet/train/"
                texts = [f for a in node.args for f in strings(a)] + \
                        [t for kw in node.keywords if kw.arg == "default" for t in strings(kw.value)]
                for name in (n for t in texts for n in named_in(t)):
                    found[normalize(name)]["path"].append((rel, node.lineno))
                    found[normalize(name)]["spelling"].append((name, "path"))
            elif w[-1:] in (["join"], ["Path"]):                                  # os.path.join(root, "cifar/CIFAR-10-C/")
                for name in (n for a in node.args for s in strings(a) for n in exact_names(s, plain=False)):
                    found[normalize(name)]["dir"].append((rel, node.lineno))     # a whole folder named for a shifted copy
                    found[normalize(name)]["spelling"].append((name, "dir"))
                for name in (m.group(1) for a in node.args for s in strings(a) for m in VARIANT_NAMES.finditer(s)):
                    found[normalize(name)]["path"].append((rel, node.lineno))
                    found[normalize(name)]["spelling"].append((name, "path"))
 
 
def scan_repo(root):
    """Returns (dataset evidence found in code, generic loaders, dataset variants picked by other flags,
    files scanned)."""
    found = defaultdict(lambda: defaultdict(list))   # dataset -> kind -> [(file, line)]
    generic = defaultdict(list)                       # ImageFolder -> [(file, line)]
    variants = defaultdict(lambda: defaultdict(list)) # (flag, function, file) -> value -> [(file, line)]
    count = 0
    for folder, subfolders, files in os.walk(root):   # os.walk never follows symlinked folders
        subfolders[:] = sorted(d for d in subfolders if d not in SKIP_DIRS and not d.startswith("."))
        for name in sorted(files):
            path = Path(folder) / name
            if path.is_symlink() or path.stat().st_size > 1_000_000:
                continue
            if not name.endswith(".py"):
                continue
            scan_python(path, path.relative_to(root).as_posix(), found, generic, variants)
            count += 1
            if count >= 3000:                         # safety limit for huge repos
                break
    return found, generic, variants, count
 
 
# ---------- report ----------
 
def where(evidence):
    """One example location per kind of evidence, strongest first (at most three)."""
    return ", ".join([f"{evidence[k][0][0]}:{evidence[k][0][1]} ({KINDS[k]})" for k in KINDS if k in evidence][:3])
 
 
def label(key, evidence):
    """The name as the repo spells it (bank_marketing, not bankmarketing), plus any other spellings of it."""
    spelled = evidence.get("spelling", [])
    flag = [s for s, k in spelled if k in FLAG_KINDS]      # what a user would pass, not a loader class name
    pool = flag or [s for s, k in spelled]
    if not pool:
        return key
    shown = Counter(pool).most_common(1)[0][0]
    others = sorted(set(flag) - {shown})
    return shown + (f" (also written: {', '.join(others)})" if others else "")
 
 
def variant_lines(variants, found):
    """One line per (flag, function, values): which flag picks the variant and where (files with the same values merge)."""
    groups = {}                                       # (flag, function, values) -> [(file, line)]
    for (flag, func, _file), values in variants.items():
        ordered = sorted(values, key=lambda v: min(values[v]))
        key = (flag, func, tuple(ordered))
        groups.setdefault(key, []).append(min(min(v) for v in values.values()))
    lines = []
    for (flag, func, values), spots in sorted(groups.items(), key=lambda kv: (kv[0][0], min(kv[1]))):
        spots = sorted(set(spots))
        shown = ", ".join(values[:12]) + (f" and {len(values) - 12} more" if len(values) > 12 else "")
        where_ = f"{spots[0][0]}:{spots[0][1]}" + (f" and {len(spots) - 1} more places" if len(spots) > 1 else "")
        parent = next((d for d in sorted(found) if len(d) >= 3 and d in normalize(func)), "")
        context = f", in {func}" + (f", for {label(parent, found[parent])}" if parent else "") if func else ""
        lines.append(f"  - {flag}: {shown} ({where_}{context})")
    return lines
 
 
def make_report(repo, found, generic, variants, count):
    # only datasets the code declares or loads; a name that only appears in an if-check, a default or a README is left out
    supported = {d: {k: v for k, v in e.items() if k != "path"}   # a path name adds nothing to real support
                 for d, e in found.items() if any(k in e for k in STRONG)}
 
    lines = [f"Repo '{repo}' ({count} files scanned)", "", "SUPPORTED (declared or loaded in code):"]
    lines += [f"  - {label(d, e)}: {where(e)}" for d, e in sorted(supported.items())] or ["  (none found)"]
    if variants:
        lines += ["", "VARIANTS PICKED BY ANOTHER FLAG (each value is a separate task or version of a dataset above):"]
        lines += variant_lines(variants, found)
    if generic:
        lines += ["", "GENERIC FOLDER LOADERS (they read any image folder, so the dataset depends on the path; check by hand):"]
        lines += [f"  - {k}: {', '.join(f'{f}:{n}' for f, n in sorted(set(v))[:3])}" for k, v in sorted(generic.items())]
    if not supported and not generic and not variants:
        lines += ["", "Note: no dataset code was recognised. The repo may load data with custom "
                      "code, so read its data files directly."]
    return "\n".join(lines)
 
 
# ---------- the tool ----------
 
@tool
def list_datasets(repo_name: str) -> str:
    """Lists the datasets a cloned repository supports.
 
    Input: repo_name, the name of a folder inside the local 'repos' folder.
    Returns: only the datasets the repo's code declares or loads (a name that appears only in a
    README, a default or a check with no loading code is left out), plus generic folder loaders
    such as ImageFolder (which dataset they read depends on the path). Variants picked by another
    flag (for example target_attr values such as income or Smiling) are listed in their own
    section, and shifted copies such as CIFAR-10-C or ImageNet-R are listed when the code takes
    their folder. Names are shown as the code spells them (bank_marketing, not bankmarketing).
    Output is capped at 6,000 characters.
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
        found, generic, variants, count = scan_repo(root)
 
        print(
            f"[DEBUG][list_datasets] Scan completed: {count} files, ",
            flush=True,
        )
 
        report = make_report(name, found, generic, variants, count)
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
 
