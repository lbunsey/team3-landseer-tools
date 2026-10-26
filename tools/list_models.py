"""list_models: lists the models a cloned repo supports (details in the tool docstring below).
Terminal: python tools/list_models.py <repo_name>"""

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
# The first four mean "supported"; the others are weaker.
KINDS = {"loaded": "model constructor/loader call", "choices": "argparse choices",
         "registry": "model list/registry", "selector": "model-selection function",
         "factory": "function that builds a model", "branch": "if-check",
         "default": "argparse default", "class": "nn.Module class"}
STRONG = ("loaded", "choices", "registry", "selector")

NAME_WORDS = {"model", "model_name", "model_type", "arch", "architecture",
              "backbone", "network", "net", "encoder"}
REGISTRY_VAR = re.compile(
    r"(^|_)(models?|archs?|architectures?|backbones?|networks?)"
    r"(_names?|_list|_dict|_registry|_choices|_zoo)?$")
SELECTOR_FN = re.compile(r"(get|build|create|make|load|select)_?(model|net|network|arch|backbone)s?", re.I)
LOADERS = {"create_model", "from_pretrained", "get_model", "build_model", "get_network"}
NOT_NAMES = {"train", "test", "val", "valid", "validation", "eval", "cpu", "cuda",
             "none", "true", "false", "default", "model", "module"}
MAX_CLASSES_SHOWN = 25


# ---------- helpers ----------

def normalize(name):
    """ResNet-18, resnet18 and resnet_18 all become 'resnet18'."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def words(node):
    """An expression as words: args.model.lower() -> ['args', 'model']."""
    return re.findall(r"\w+", re.sub(r"\.(lower|upper|strip)\(\)", "", ast.unparse(node)))


def strings(node):
    """The string constants in a literal: 'a', ['a', 'b'] or {'a': ...}."""
    items = node.elts if isinstance(node, (ast.List, ast.Tuple, ast.Set)) else \
        node.keys if isinstance(node, ast.Dict) else [node]
    return [i.value for i in items if isinstance(i, ast.Constant) and isinstance(i.value, str)]


def is_model_var(node):
    """True for args.model, opt.arch.lower(), cfg['model']['name'], model_name, ..."""
    w = [] if isinstance(node, ast.Constant) else [x.lower() for x in words(node)]
    return bool(w) and bool({w[-1], "_".join(w[-2:])} & NAME_WORDS)


def compared_names(node):
    """Strings compared against a model variable: args.model == 'x', model in ['x', 'y']."""
    sides = [node.left] + node.comparators
    for op, a, b in zip(node.ops, sides, sides[1:]):
        # `'x' in args.model` is a substring test, so only `args.model in [...]` counts
        pairs = [(a, b)] if isinstance(op, (ast.In, ast.NotIn)) else [(a, b), (b, a)]
        for var, values in pairs:
            if is_model_var(var):
                yield from strings(values)


def is_module_class(node):
    return isinstance(node, ast.ClassDef) and any(words(b)[-1:] == ["Module"] for b in node.bases)


# ---------- scanning ----------

def scan_python(path, rel, found, names):
    """Record every model name found in one Python file."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # the scanned repo's own warnings are not ours
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except (SyntaxError, ValueError):
        return                                # e.g. Python 2 code

    def add(name, kind, node):
        key = normalize(name)
        if re.fullmatch(r"[A-Za-z][\w\-.]{1,40}", name) and key not in NOT_NAMES:
            names.setdefault(key, name)
            found[key][kind].append((rel, node.lineno))

    # names bound to torchvision.models (module aliases) or its constructors
    tv_modules, tv_names = set(), set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module:
            if n.module == "torchvision":
                tv_modules |= {a.asname or a.name for a in n.names if a.name == "models"}
            elif n.module.startswith("torchvision.models"):
                tv_names |= {a.asname or a.name for a in n.names}
        elif isinstance(n, ast.Import):
            for a in n.names:
                if a.name.startswith("torchvision.models"):
                    tv_modules.add(a.asname or a.name.split(".")[0])

    # if-checks inside get_model / build_model style functions: the repo's real model switch
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and SELECTOR_FN.fullmatch(fn.name):
            for node in ast.walk(fn):
                if isinstance(node, ast.Compare):
                    for name in compared_names(node):
                        add(name, "selector", node)

    # top-level functions that return an nn.Module class defined in the same file
    module_classes = {n.name for n in tree.body if is_module_class(n)}
    for fn in tree.body:
        if isinstance(fn, ast.FunctionDef) and not SELECTOR_FN.fullmatch(fn.name):
            for r in ast.walk(fn):
                if isinstance(r, ast.Return) and isinstance(r.value, ast.Call) \
                        and words(r.value.func)[-1:] and words(r.value.func)[-1] in module_classes:
                    add(fn.name, "factory", fn)
                    break

    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):                      # if args.model == "resnet18"
            for name in compared_names(node):
                add(name, "branch", node)

        elif isinstance(node, (ast.Assign, ast.AnnAssign)):    # MODELS = {"resnet18": ResNet18}
            targets = getattr(node, "targets", None) or [node.target]
            if isinstance(node.value, (ast.List, ast.Tuple, ast.Set, ast.Dict)) \
                    and any(REGISTRY_VAR.search((words(t) or [""])[-1].lower()) for t in targets):
                for name in strings(node.value):
                    add(name, "registry", node)

        elif is_module_class(node):                            # class ResNet(nn.Module)
            add(node.name, "class", node)

        elif isinstance(node, ast.Call):
            w = words(node.func)
            first_str = strings(node.args[0]) if node.args else []
            if w[-1:] and w[-1] in LOADERS and first_str:      # timm.create_model("vit_base")
                for name in first_str:
                    add(name, "loaded", node)
            elif w[-2:] == ["hub", "load"] and len(node.args) > 1:   # torch.hub.load(repo, "name")
                for name in strings(node.args[1]):
                    add(name, "loaded", node)
            elif len(w) >= 2 and "models" in w[:-1] and w[0] in tv_modules:
                add(w[-1], "loaded", node)                     # models.resnet18(...)
            elif len(w) == 1 and w[0] in tv_names:
                add(w[0], "loaded", node)                      # resnet18(...) after a direct import
            elif w[-1:] == ["add_argument"]:                   # parser.add_argument("--model", ...)
                flags = {f.lstrip("-").replace("-", "_").lower() for a in node.args for f in strings(a)}
                if flags & NAME_WORDS:
                    for kw in node.keywords:
                        if kw.arg in ("choices", "default"):
                            for name in strings(kw.value):
                                add(name, kw.arg, node)


def scan_repo(root):
    """Returns (model evidence found in code, display names, files scanned)."""
    found = defaultdict(lambda: defaultdict(list))   # model -> kind -> [(file, line)]
    names = {}                                        # normalized name -> name as written
    count = 0
    for folder, subfolders, files in os.walk(root):   # os.walk never follows symlinked folders
        subfolders[:] = sorted(d for d in subfolders if d not in SKIP_DIRS and not d.startswith("."))
        for name in sorted(files):
            path = Path(folder) / name
            if not name.endswith(".py") or path.is_symlink() or path.stat().st_size > 1_000_000:
                continue
            scan_python(path, path.relative_to(root).as_posix(), found, names)
            count += 1
            if count >= 3000:                         # safety limit for huge repos
                return found, names, count
    return found, names, count


# ---------- report ----------

def where(evidence):
    """One example location per kind of evidence, strongest first (at most three)."""
    return ", ".join([f"{evidence[k][0][0]}:{evidence[k][0][1]} ({KINDS[k]})" for k in KINDS if k in evidence][:3])


def make_report(repo, found, names, count):
    supported = {m: e for m, e in found.items() if any(k in e for k in STRONG)}
    rest = {m: e for m, e in found.items() if m not in supported}
    factories = {m: e for m, e in rest.items() if "factory" in e}
    classes = {m: e for m, e in rest.items() if set(e) == {"class"}}
    weak = {m: e for m, e in rest.items() if m not in factories and m not in classes}

    lines = [f"Repo '{repo}' ({count} Python files scanned)", "",
             "SUPPORTED (selectable by name, listed in choices/registry, or built by a loader call):"]
    lines += [f"  - {names[m]}: {where(e)}" for m, e in sorted(supported.items())] or ["  (none found)"]
    if factories:
        lines += ["", "DEFINED BUT NOT SEEN IN ANY MODEL SELECTOR (may be unused, or wired up in a way "
                      "this scan can't see; verify by reading the repo's get_model-style function):"]
        lines += [f"  - {names[m]}: {where(e)}" for m, e in sorted(factories.items())]
    if weak:
        lines += ["", "NAMED IN CODE, WEAKER EVIDENCE (if-checks or defaults; verify by reading the file):"]
        lines += [f"  - {names[m]}: {where(e)}" for m, e in sorted(weak.items())]
    if classes:
        shown = sorted(classes.items())[:MAX_CLASSES_SHOWN]
        lines += ["", "nn.Module CLASSES (may be layers/blocks, not full models; verify by reading the file):"]
        lines += [f"  - {names[m]}: {where(e)}" for m, e in shown]
        if len(classes) > MAX_CLASSES_SHOWN:
            lines.append(f"  ... and {len(classes) - MAX_CLASSES_SHOWN} more")
    if not found:
        lines += ["", "Note: no model code was recognised. The repo may build models with custom "
                      "code, so use search_repo and read_repo_file to look directly."]
    lines += ["", "Next step: open the files above with read_repo_file to confirm what each hit means."]
    return "\n".join(lines)


# ---------- the tool ----------

@tool
def list_models(repo_name: str) -> str:
    """Lists the models a cloned repository supports.

    Input: repo_name, the name of a folder inside the local 'repos' folder.
    Returns: model names the repo selects by name (get_model-style functions), declares
    (argparse choices, model registries) or builds (timm/torchvision/HuggingFace loader calls),
    then models that are defined but not seen in any selector, then weaker evidence
    (if-checks, defaults, nn.Module classes), each with file:line. These are leads, not final
    answers: confirm them with read_repo_file. Read-only: it never runs the repo's code.
    Returns an error message if the repo is not found.
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
        sys.exit("usage: python tools/list_models.py <repo_name>")
    print(list_models.invoke({"repo_name": sys.argv[1]}))