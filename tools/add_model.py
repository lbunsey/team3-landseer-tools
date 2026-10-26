"""add_model: proposes the code changes that make a cloned repo support a new model
(details in the tool docstring below). It never writes the repo itself: every changed file
becomes a pending patch that a person approves or rejects in the CLI (tools/patch_review.py).
Terminal: python -m tools.add_model <repo_name> <model_name> [model_code.py]"""

import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
from collections import defaultdict
from pathlib import Path

from langchain_core.tools import tool

from . import patch_preview, patch_store, patch_utils
from .model_templates import find_template, template_names

REPOS_DIR = Path(__file__).resolve().parent.parent / "repos"
SKIP_DIRS = {"venv", "env", "node_modules", "__pycache__", "site-packages", "build", "dist"}
SESSION_ID = "add_model"

NAME_WORDS = {"model", "model_name", "model_type", "arch", "architecture",
              "backbone", "network", "net", "net_name", "arch_name"}
REGISTRY_VAR = re.compile(
    r"(^|_)(models?|archs?|architectures?|backbones?|networks?)"
    r"(_names?|_list|_dict|_map|_registry|_choices|_zoo|_fns?)?$")
CHAIN_METHODS = {"to", "cuda", "cpu", "eval", "train", "float", "half"}   # resnet18(...).to(device)
SMOKE_TIMEOUT = 180


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


def compared(test):
    """(model variable, string constant) pairs in `args.model == 'x'` / `model in ['x', 'y']`."""
    if not isinstance(test, ast.Compare):
        return []
    pairs, sides = [], [test.left] + test.comparators
    for op, a, b in zip(test.ops, sides, sides[1:]):
        if isinstance(op, ast.Eq):
            for var, value in ((a, b), (b, a)):
                if is_model_var(var) and isinstance(value, ast.Constant) and isinstance(value.value, str):
                    pairs.append((var, value))
        elif isinstance(op, ast.In) and is_model_var(a) and isinstance(b, (ast.List, ast.Tuple, ast.Set)):
            pairs += [(a, e) for e in b.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
    return pairs


def constructor(stmt):
    """The model-building call in `return f(...)` / `net = f(...).to(dev)`, else None."""
    value = getattr(stmt, "value", None)
    while isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) \
            and value.func.attr in CHAIN_METHODS:
        value = value.func.value
    return value if isinstance(value, ast.Call) and isinstance(value.func, (ast.Name, ast.Attribute)) else None


def top_level_names(tree):
    """Names a module defines or imports at top level."""
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names |= {(a.asname or a.name).split(".")[0] for a in node.names}
        elif isinstance(node, ast.Assign):
            names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
    return names


class Source:
    """One repo file: its text, syntax tree and the edits proposed for it."""

    def __init__(self, path, rel):
        self.path, self.rel = path, rel
        self.text = path.read_bytes().decode("utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # the scanned repo's own warnings are not ours
            self.tree = ast.parse(self.text)
        self.lines = self.text.splitlines(keepends=True)
        self.starts = [0]
        for line in self.lines:
            self.starts.append(self.starts[-1] + len(line))
        self.edits = []                       # (offset, text to insert / replace span)

    def pos(self, lineno, col):               # AST columns count UTF-8 bytes
        return self.starts[lineno - 1] + len(self.lines[lineno - 1].encode()[:col].decode(errors="ignore"))

    def start(self, node):
        return self.pos(node.lineno, node.col_offset)

    def end(self, node):
        return self.pos(node.end_lineno, node.end_col_offset)

    def seg(self, node):
        return self.text[self.start(node):self.end(node)]

    def indent(self, lineno):
        line = self.lines[lineno - 1]
        return line[:len(line) - len(line.lstrip())]

    def insert(self, offset, text):
        self.edits.append((offset, text))

    def updated(self):
        text = self.text
        for offset, new in sorted(self.edits, key=lambda e: e[0], reverse=True):
            text = text[:offset] + new + text[offset:]
        return text


def splice(src, outer, inner, new):
    """Source of `outer` with the `inner` node inside it replaced by `new`."""
    return src.text[src.start(outer):src.start(inner)] + new + src.text[src.end(inner):src.end(outer)]


def append_item(src, container, last, item, indent_line=None):
    """Insert `item` after `last`, the final element of a list/dict literal, keeping its layout."""
    offset = src.end(last)
    comma = re.match(r"\s*,", src.text[offset:src.end(container)])
    if last.end_lineno < container.end_lineno:            # one element per line
        indent = src.indent(indent_line or last.lineno)
        if comma:
            src.insert(offset + comma.end(), f"\n{indent}{item},")
        else:
            src.insert(offset, f",\n{indent}{item}")
    else:
        src.insert(offset + comma.end() if comma else offset, f" {item}," if comma else f", {item}")


# ---------- scanning ----------

def load_sources(root):
    """Every parsable Python file in the repo, plus top-level definitions by name."""
    sources, defs = {}, defaultdict(list)     # rel -> Source; name -> [(rel, node)]
    for folder, subfolders, files in os.walk(root):   # os.walk never follows symlinked folders
        subfolders[:] = sorted(d for d in subfolders if d not in SKIP_DIRS and not d.startswith("."))
        for name in sorted(files):
            path = Path(folder) / name
            if not name.endswith(".py") or path.is_symlink() or path.stat().st_size > patch_utils.MAX_FILE_BYTES:
                continue
            rel = path.relative_to(root).as_posix()
            try:
                src = Source(path, rel)
            except (SyntaxError, ValueError, UnicodeDecodeError):
                continue                      # e.g. Python 2 code
            sources[rel] = src
            for node in src.tree.body:
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    defs[node.name].append((rel, node))
            if len(sources) >= 3000:          # safety limit for huge repos
                return sources, defs
    return sources, defs


def find_sites(src):
    """Places in one file where the repo selects or lists models by name."""
    sites, elifs = [], set()
    for node in ast.walk(src.tree):
        if isinstance(node, ast.If) and id(node) not in elifs:        # if model == 'x': ... elif ...
            branches = [node]
            while len(branches[-1].orelse) == 1 and isinstance(branches[-1].orelse[0], ast.If) \
                    and src.lines[branches[-1].orelse[0].lineno - 1].lstrip().startswith("elif"):
                branches.append(branches[-1].orelse[0])
                elifs.add(id(branches[-1]))
            keys = [c.value for b in branches for _, c in compared(b.test)]
            if keys:
                sites.append({"kind": "if-chain", "src": src, "node": node, "branches": branches, "keys": keys})

        elif isinstance(node, (ast.Assign, ast.AnnAssign)):           # MODELS = {'x': X} / ['x']
            targets = getattr(node, "targets", None) or [node.target]
            value = node.value
            if any(REGISTRY_VAR.search((words(t) or [""])[-1].lower()) for t in targets) and \
                    isinstance(value, (ast.Dict, ast.List, ast.Tuple)) and strings(value):
                kind = "registry" if isinstance(value, ast.Dict) else "name list"
                sites.append({"kind": kind, "src": src, "node": value, "keys": strings(value)})

        elif isinstance(node, ast.Call):
            w = words(node.func)
            if w[-1:] == ["add_argument"]:                            # --model ... choices=[...]
                flags = {f.lstrip("-").replace("-", "_").lower() for a in node.args for f in strings(a)}
                if flags & NAME_WORDS:
                    for kw in node.keywords:
                        if kw.arg == "choices" and isinstance(kw.value, (ast.List, ast.Tuple)) and strings(kw.value):
                            sites.append({"kind": "argparse choices", "src": src, "node": kw.value,
                                          "keys": strings(kw.value)})
                    flag = sorted(f for a in node.args for f in strings(a) if f.startswith("--"))
                    if flag:
                        sites.append({"kind": "flag", "src": src, "node": node, "keys": [], "flag": flag[0]})
            elif w == ["getattr"] and len(node.args) >= 2 and isinstance(node.args[0], ast.Name) \
                    and is_model_var(node.args[1]):                   # getattr(models, args.arch)
                sites.append({"kind": "getattr", "src": src, "node": node, "keys": [],
                              "module": node.args[0].id})
    return sites


def package_dir(src, module_name, root):
    """The local package a name in `src` refers to (`import models` -> root/models), or None."""
    for node in src.tree.body:
        if isinstance(node, ast.Import):
            for a in node.names:
                if (a.asname or a.name) == module_name:
                    folder = root.joinpath(*a.name.split("."))
                    return folder if (folder / "__init__.py").is_file() else None
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                if (a.asname or a.name) == module_name:
                    base = src.path.parent if node.level else root
                    for _ in range(max(node.level - 1, 0)):
                        base = base.parent
                    folder = base.joinpath(*(node.module or "").split("."), a.name) if node.module \
                        else base / a.name
                    return folder if (folder / "__init__.py").is_file() else None
    return None


# ---------- planning the change ----------

def template_branch(site):
    """(branch, compared var, string const, constructor call) to copy, preferring a branch
    whose call is named after its key (resnet18 -> resnet18(...)); also says if it matched."""
    fallback = None
    for branch in site["branches"]:
        pairs = compared(branch.test)
        call = next((c for c in map(constructor, branch.body) if c is not None), None)
        if len(pairs) != 1 or call is None:
            continue
        var, const = pairs[0]
        if normalize(words(call.func)[-1]) == normalize(const.value):
            return (branch, var, const, call), True
        fallback = fallback or (branch, var, const, call)
    return fallback, False


def plan_if_chain(site, key, factory, notes):
    """Add `elif <var> == '<key>':` with a copy of an existing branch body. Returns the new call."""
    src, branches = site["src"], site["branches"]
    picked, exact = template_branch(site)
    if picked is None:
        notes.append(f"{src.rel}:{site['node'].lineno}: if-chain has no branch to copy, so it was left alone.")
        return None
    branch, var, const, call = picked
    if not exact:
        notes.append(f"{src.rel}:{branch.lineno}: copied the call arguments of the '{const.value}' branch; "
                     "check they fit the new model.")
    quote = src.seg(const)[0] if src.seg(const)[:1] in ("'", '"') else "'"
    indent = src.indent(site["node"].lineno)
    first, last_stmt = branch.body[0], branch.body[-1]
    if first.lineno == branch.test.lineno:                            # if x == 'a': return f()
        body = indent + "    " + src.text[src.start(first):src.start(call.func)] + factory + \
            src.text[src.end(call.func):src.end(last_stmt)] + "\n"
    else:
        body = src.text[src.starts[first.lineno - 1]:src.start(call.func)] + factory + \
            src.text[src.end(call.func):src.starts[last_stmt.end_lineno]]
        body += "" if body.endswith("\n") else "\n"
    new = f"{indent}elif {src.seg(var)} == {quote}{key}{quote}:\n{body}"

    last = branches[-1]
    if last.orelse:                                                   # insert before `else:`
        for lineno in range(last.orelse[0].lineno, last.body[-1].end_lineno, -1):
            if re.match(r"\s*else\s*:", src.lines[lineno - 1]):
                src.insert(src.starts[lineno - 1], new)
                break
        else:
            notes.append(f"{src.rel}:{last.lineno}: could not find the 'else:' line, so this if-chain was left alone.")
            return None
    else:
        offset = src.starts[last.body[-1].end_lineno]
        src.insert(offset, new if src.text[offset - 1:offset] == "\n" else "\n" + new)
    return splice(src, call, call.func, factory)


def plan_registry(site, key, factory, notes):
    src, node = site["src"], site["node"]
    is_list = not isinstance(node, ast.Dict)                          # name list or argparse choices
    quote = src.seg(node.elts[0] if is_list else node.keys[0])[0]
    if is_list:
        append_item(src, node, node.elts[-1], f"{quote}{key}{quote}")
        return
    for k, v in zip(node.keys, node.values):                          # copy an entry built like its key
        hit = next((n for n in ast.walk(v) if isinstance(n, (ast.Name, ast.Attribute))
                    and normalize(words(n)[-1]) == normalize(getattr(k, "value", ""))), None)
        if hit is not None:
            value = splice(src, v, hit, factory)
            append_item(src, node, node.values[-1], f"{quote}{key}{quote}: {value}", node.keys[-1].lineno)
            return
    notes.append(f"{src.rel}:{node.lineno}: could not tell how registry entries are built, so it was left alone.")


def plan_import(src, def_rel, factory, root, anchor_name=None):
    """Make `factory` importable in `src` (unless it is already defined or imported there)."""
    if def_rel == src.rel or factory in top_level_names(src.tree):
        return
    def_path = root / def_rel
    relative = any(isinstance(n, ast.ImportFrom) and n.level for n in src.tree.body)
    if relative and def_path.parent == src.path.parent:
        module = "." + def_path.stem
    else:
        module = ".".join(Path(def_rel).with_suffix("").parts)
    imports = [n for n in src.tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    after = next((n for n in imports if anchor_name in {a.asname or a.name for a in n.names}), None) \
        or (imports[-1] if imports else None)
    if after is not None:
        offset = src.starts[after.end_lineno]
    else:                                                             # after a module docstring, if any
        doc = src.tree.body[0] if src.tree.body and isinstance(getattr(src.tree.body[0], "value", None), ast.Constant) \
            and isinstance(src.tree.body[0].value.value, str) else None
        offset = src.starts[doc.end_lineno] if doc else 0
    line = f"from {module} import {factory}\n"
    src.insert(offset, line if offset == 0 or src.text[offset - 1:offset] == "\n" else "\n" + line)


def find_definition(target, defs, models_dir):
    """A top-level def/class in the repo named like the target, preferring factories in models_dir."""
    hits = [(rel, node) for name, found in defs.items() if normalize(name) == target for rel, node in found]
    hits.sort(key=lambda h: (not isinstance(h[1], ast.FunctionDef), str(Path(h[0]).parent) != models_dir))
    return hits[0] if hits else None


def signature(node):
    if isinstance(node, ast.ClassDef):
        init = next((n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None)
        return f"class {node.name}: __init__({ast.unparse(init.args) if init else ''})"
    return f"def {node.name}({ast.unparse(node.args)})"


# ---------- smoke test ----------

SMOKE_SCRIPT = r"""
import importlib, json, sys, types
sys.path.insert(0, sys.argv[1])
cfg = json.loads(sys.argv[2])
try:
    import torch, torch.nn as nn
except ImportError:
    print(json.dumps({"status": "skipped", "detail": "PyTorch is not installed in this Python"})); sys.exit()
class ProbeBN(nn.BatchNorm2d):
    pass
fn = getattr(importlib.import_module(cfg["module"]), cfg["factory"])
ns = dict(num_classes=10, n_classes=10, nclasses=10, num_class=10, classes=10, norm_layer=ProbeBN,
          args=types.SimpleNamespace(num_classes=10, n_classes=10), nn=nn, **{cfg["factory"]: fn})
how = cfg["call"]
try:
    net = eval(how, ns)
except NameError:
    how = cfg["factory"] + "(num_classes=10)"
    net = fn(num_classes=10)
net.eval()
s = cfg["size"]
with torch.no_grad():
    out = net(torch.randn(2, 3, s, s))
detail = f"built {how} with num_classes=10; forward pass on 2x3x{s}x{s} gave {tuple(out.shape)}"
if tuple(out.shape) != (2, 10):
    print(json.dumps({"status": "failed", "detail": detail + ", expected (2, 10)"})); sys.exit()
bns = [(n, m) for n, m in net.named_modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
if "norm_layer" in how:
    bad = [n for n, m in bns if not isinstance(m, ProbeBN)]
    if bad:
        print(json.dumps({"status": "failed", "detail": detail + f"; {len(bad)} of {len(bns)} BatchNorm layers "
              f"ignore norm_layer (e.g. {bad[0]}), so defenses that swap in their own BatchNorm would miss them"}))
        sys.exit()
    detail += f"; all {len(bns)} BatchNorm layers are built through norm_layer"
print(json.dumps({"status": "passed", "detail": detail}))
"""


def smoke_test(root, changes, def_rel, factory, call, size):
    """Build the model from a scratch copy of the repo's Python files with the changes applied.
    The real repo is not touched. Returns (status, detail)."""
    with tempfile.TemporaryDirectory(prefix="add_model_") as tmp:
        copy = Path(tmp) / "repo"

        def ignore(folder, names):
            return [n for n in names if n.startswith(".") or n in SKIP_DIRS or (Path(folder) / n).is_symlink()
                    or ((Path(folder) / n).is_file() and not n.endswith(".py"))]

        shutil.copytree(root, copy, ignore=ignore)
        for rel, text in changes.items():
            (copy / rel).write_text(text, encoding="utf-8")
        cfg = {"module": ".".join(Path(def_rel).with_suffix("").parts), "factory": factory,
               "call": call or f"{factory}(num_classes=10)", "size": size}
        try:
            result = subprocess.run([sys.executable, "-c", SMOKE_SCRIPT, str(copy), json.dumps(cfg)],
                                    capture_output=True, text=True, timeout=SMOKE_TIMEOUT, cwd=copy,
                                    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        except subprocess.TimeoutExpired:
            return "failed", f"timed out after {SMOKE_TIMEOUT} seconds"
    lines = result.stdout.strip().splitlines()
    if result.returncode == 0 and lines:
        try:
            report = json.loads(lines[-1])
            return report["status"], report["detail"]
        except (ValueError, KeyError):
            pass
    return "failed", (result.stderr.strip() or result.stdout.strip())[-1500:]


# ---------- the change ----------

def propose(name, root, model_name, code, input_size):
    target = normalize(model_name)
    if not target:
        return "Error: model_name must contain letters or digits."
    for record in patch_store.list_pending_patches():
        if record.repo_name == name:
            return (f"Error: repo '{name}' already has changes waiting for review ({record.file_path}, ...). "
                    "Ask the user to approve or reject them first, then call add_model again.")

    sources, defs = load_sources(root)
    sites = [s for src in sources.values() for s in find_sites(src)]
    dispatch = [s for s in sites if s["kind"] in ("if-chain", "registry", "getattr")]
    flags = {}                                                        # --model -> where it is defined
    for s in sites:
        if s["kind"] == "flag":
            flags.setdefault(s["flag"], []).append(f"{s['src'].rel}:{s['node'].lineno}")
    if not dispatch:
        return (f"Error: no model-selection code found in '{name}' (no if-chain on a model/arch variable, "
                "model registry or getattr(models, args.arch)). The repo may build one fixed model; "
                "read its training script to see how, then change it by hand.")

    for s in sites:                                                   # already selectable by name?
        if s["kind"] in ("if-chain", "registry") and target in map(normalize, s["keys"]):
            return (f"'{model_name}' is already supported: {s['src'].rel}:{s['node'].lineno} ({s['kind']}). "
                    "No changes proposed.")

    # where the repo keeps its models: next to the existing factories the dispatch calls
    known_calls = []
    for s in dispatch:
        if s["kind"] == "if-chain":
            picked, _ = template_branch(s)
            if picked:
                known_calls.append(picked[3])
    callees = [words(c.func)[-1] for c in known_calls]
    callees += [words(v)[-1] for s in dispatch if s["kind"] == "registry" for v in s["node"].values if words(v)]
    callee_defs = [defs[c][0] for c in callees if defs.get(c)]
    packages = [package_dir(s["src"], s["module"], root) for s in dispatch if s["kind"] == "getattr"]
    if callee_defs:
        models_dir = str(Path(callee_defs[0][0]).parent)
    elif any(packages):
        models_dir = next(p for p in packages if p).relative_to(root).as_posix()
    else:
        models_dir = str(Path(dispatch[0]["src"].rel).parent)

    # 1) the model's code: user-supplied, already in the repo, or a built-in template
    new_files, notes, origin = {}, [], ""
    if code.strip():
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return f"Error: the supplied code does not parse: line {e.lineno}: {e.msg}"
        found = sorted((n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                        and normalize(n.name) == target), key=lambda n: not isinstance(n, ast.FunctionDef))
        if not found:
            return f"Error: the supplied code must define a top-level function or class named like '{model_name}'."
        factory = found[0].name
        def_rel = (Path(models_dir) / f"{factory.lower()}.py").as_posix()
        new_files[def_rel] = code if code.endswith("\n") else code + "\n"
        origin = f"new file {def_rel} (code supplied by the agent)"
    elif (hit := find_definition(target, defs, models_dir)) is not None:
        def_rel, node = hit
        factory = node.name
        origin = f"already defined at {def_rel}:{node.lineno} ({signature(node)}), only registration was missing"
    elif (template := find_template(model_name)) is not None:
        factory, stem, body, note = template
        def_rel = (Path(models_dir) / f"{stem}.py").as_posix()
        if (root / def_rel).exists():
            def_rel = (Path(models_dir) / f"landseer_{stem}.py").as_posix()
        new_files[def_rel] = body
        origin = f"new file {def_rel} (built-in template)"
        notes += [note] if note else []
    else:
        example = callee_defs[0] if callee_defs else None
        lines = [f"Error: '{model_name}' is not defined anywhere in '{name}' and there is no built-in template "
                 f"for it (templates: {', '.join(template_names())}).",
                 "Write the model yourself and call add_model again with the full module source as `code`. It must:",
                 f"  - define a top-level function or class named like '{model_name}'",
                 "  - import everything it uses (e.g. torch, torch.nn as nn)"]
        if known_calls:
            c, s = known_calls[0], next(s for s in dispatch if s["kind"] == "if-chain")
            lines.append(f"  - be callable the same way the repo calls its models: {s['src'].seg(c)} "
                         f"({s['src'].rel}:{c.lineno})")
            if any(k.arg == "norm_layer" for k in c.keywords) or "norm_layer" in s["src"].seg(c):
                lines.append("  - build EVERY BatchNorm layer as norm_layer(num_features): the repo swaps in its own "
                             "BatchNorm subclass through this argument")
        if example:
            lines.append(f"  - follow the existing definition {example[0]}:{example[1].lineno}: {signature(example[1])}")
        lines.append(f"It will be saved in {models_dir or '.'}/ and registered automatically.")
        return "\n".join(lines)

    # 2) register it: same key style as the existing names
    existing = [k for s in sites for k in s["keys"]]
    key = factory.lower() if all(k == k.lower() for k in existing) else factory
    registered, calls = [], []
    for s in dispatch:
        src = s["src"]
        if s["kind"] == "if-chain":
            call = plan_if_chain(s, key, factory, notes)
            if call:
                calls.append(call)
                plan_import(src, def_rel, factory, root, words(template_branch(s)[0][3].func)[0])
                registered.append(f"{src.rel}:{s['node'].lineno} (if-chain)")
        elif s["kind"] == "registry":
            before = len(src.edits)
            plan_registry(s, key, factory, notes)
            if len(src.edits) > before:
                plan_import(src, def_rel, factory, root)
                registered.append(f"{src.rel}:{s['node'].lineno} (registry)")
        elif s["kind"] == "getattr":                                  # export it from the package
            folder = package_dir(src, s["module"], root)
            init_rel = (folder / "__init__.py").relative_to(root).as_posix() if folder else None
            if init_rel in sources:
                plan_import(sources[init_rel], def_rel, factory, root)
                registered.append(f"{init_rel} (exported for getattr at {src.rel}:{s['node'].lineno})")
            else:
                notes.append(f"{src.rel}:{s['node'].lineno}: getattr on '{s['module']}' but it is not a local "
                             "package, so it was left alone.")
    if not registered:
        return "Error: found model-selection code but could not extend it automatically.\n" + "\n".join(notes)
    for s in sites:
        if s["kind"] in ("argparse choices", "name list"):
            plan_registry(s, key, factory, notes)
            registered.append(f"{s['src'].rel}:{s['node'].lineno} ({s['kind']})")

    changes = dict(new_files)
    for src in sources.values():
        if src.edits:
            changes[src.rel] = src.updated()
    for rel, text in changes.items():
        try:
            ast.parse(text)
        except SyntaxError as e:
            return f"Error: the proposed {rel} would not parse (line {e.lineno}: {e.msg}); nothing was proposed."

    # 3) build the model once from a scratch copy, before asking anyone to approve it
    print(f"[DEBUG][add_model] Smoke-testing {factory} on a scratch copy", flush=True)
    status, detail = smoke_test(root, changes, def_rel, factory, calls[0] if calls else None, input_size)
    if status == "failed":
        return (f"Error: the model was not proposed because the smoke test failed: {detail}\n"
                "Fix the model code (pass it as `code`) and call add_model again.")

    # 4) hand every changed file to the patch workflow; the person approves in the CLI
    proposed = []
    try:
        for rel, text in changes.items():                             # new files first
            prepared = patch_preview.prepare_new_file(name, rel, text) if rel in new_files else \
                patch_preview.prepare_patch(name, rel, text, patch_utils.compute_file_hash(
                    (root / rel).read_bytes()))
            proposed.append(patch_preview._save_prepared(prepared, SESSION_ID, None))
    except (ValueError, TypeError, KeyError, OSError) as exc:
        for record in proposed:
            patch_store.reject_patch(record.patch_id)
        return f"Error: could not prepare the patches: {type(exc).__name__}: {exc}"

    out = [f"PROPOSED (not applied yet): add model '{key}' to repo '{name}'. The user will see the diffs "
           "and approve or reject them after your reply.",
           f"Model code: {origin}",
           "Registered in:", *[f"  - {r}" for r in registered],
           f"Smoke test: {status} ({detail})"]
    for flag, where in sorted(flags.items()):
        out.append(f"Once applied, select it with {flag} {key} ({', '.join(sorted(where))})")
    if notes:
        out += ["Notes:", *[f"  - {n}" for n in notes]]
    out += ["Pending patches:", *[f"  - {r.patch_id[:8]} {r.file_path}" + (" (new file)" if r.creates_file else "")
                                  for r in proposed]]
    out += ["", "Diff of edited files:", *[r.diff for r in proposed if not r.creates_file]]
    return "\n".join(out)


# ---------- the tool ----------

@tool
def add_model(repo_name: str, model_name: str, code: str = "", input_size: int = 32) -> str:
    """Proposes the code changes that make a cloned repository support a new model.

    Input: repo_name, a folder inside the local 'repos' folder; model_name, e.g. 'resnet34',
    'vgg16' or 'preactresnet18'; optionally code, the full source of a Python module defining
    the model (only needed when the tool says the model has no definition or template); and
    input_size, the image height/width used for the smoke test (32 for CIFAR, 64 for Tiny-ImageNet).
    It finds where the repo selects models by name (if-chains, registries, argparse choices),
    reuses a definition already in the repo or a built-in template (VGG11-19, PreAct ResNet18/34,
    MobileNetV2), wires it in the same way as the existing models, and builds the model once on a
    scratch copy to check it. Nothing is written: the changes become pending patches that the user
    approves or rejects after your reply. Returns the plan and diffs, or an error explaining what
    is missing.
    """
    name = repo_name.strip().rstrip("/")
    print(f"[DEBUG][add_model] add_model started: {name} {model_name}", flush=True)
    if not name or Path(name).name != name or name in (".", ".."):   # a folder name, not a path
        return "Error: repo_name must be a folder name inside 'repos', not a path."
    root = REPOS_DIR / name
    if root.is_symlink() or not root.is_dir():
        available = sorted(p.name for p in REPOS_DIR.iterdir() if p.is_dir()) if REPOS_DIR.is_dir() else []
        return f"Error: repo '{name}' not found in repos/. Available repos: {', '.join(available) or 'none'}."
    try:
        report = propose(name, root.resolve(), model_name.strip(), code, input_size)
    except Exception as e:
        print(f"[ERROR][add_model] add_model failed: {type(e).__name__}: {e}", flush=True)
        raise
    print("[DEBUG][add_model] add_model finished.", flush=True)
    return report[:8000]


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        sys.exit("usage: python -m tools.add_model <repo_name> <model_name> [model_code.py]")
    from .patch_review import review_pending_patches
    source = Path(sys.argv[3]).read_text() if len(sys.argv) == 4 else ""
    print(add_model.invoke({"repo_name": sys.argv[1], "model_name": sys.argv[2], "code": source}))
    review_pending_patches()
