"""Repository-independent, framework-owned tensor classification checks.

A repo supplies an import-safe components factory used by its real entry point.
It does not supply tests or pass/fail decisions. Not an OS security sandbox.
"""
import ast
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import traceback


class Blocked(RuntimeError):
    pass


class ContractFailure(RuntimeError):
    pass


class Checks:
    def __init__(self):
        self.count = 0
        self.details = {}
    def require(self, condition, message):
        self.count += 1
        if not condition:
            raise ContractFailure(message)


def _load(root, filename, symbol):
    path = (root / filename).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise Blocked(f"Missing repository interface: {filename}")
    parts = Path(filename).with_suffix("").parts
    if not all(p.isidentifier() for p in parts):
        raise Blocked("Factory file must be an importable Python module")
    sys.path.insert(0, str(path.parent))
    module = importlib.import_module(".".join(parts))
    value = module
    for part in symbol.split("."):
        value = getattr(value, part)
    return value


def _entry_uses_factory(root, spec, checks):
    entry = ast.parse((root / spec["entry_file"]).read_text(encoding="utf-8-sig"))
    module_name = ".".join(Path(spec["factory_file"]).with_suffix("").parts)
    names, modules = set(), set()
    if spec["entry_file"] == spec["factory_file"]:
        names.add(spec["factory_name"])
    for node in ast.walk(entry):
        imported = getattr(node, "module", None)
        if isinstance(node, ast.ImportFrom) and node.level:
            package = ".".join(Path(spec["entry_file"]).parent.parts)
            try:
                imported = importlib.util.resolve_name("." * node.level + (node.module or ""), package)
            except (ImportError, ValueError):
                imported = None
        local_module = Path(spec["factory_file"]).stem if Path(spec["factory_file"]).parent == Path(spec["entry_file"]).parent else None
        if isinstance(node, ast.ImportFrom) and imported in {module_name, local_module}:
            names.update(alias.asname or alias.name for alias in node.names if alias.name == spec["factory_name"])
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == module_name:
                    modules.add(alias.asname or alias.name)
    scope = entry
    if spec.get("entry_function"):
        scope = next((n for n in entry.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                      and n.name == spec["entry_function"]), None)
    checks.require(scope is not None, "Declared existing entry function was not found")
    found = False
    for n in ast.walk(scope):
        if isinstance(n, ast.Call):
            if isinstance(n.func, ast.Name) and n.func.id in names:
                found = True
            if isinstance(n.func, ast.Attribute) and n.func.attr == spec["factory_name"]:
                base = ast.unparse(n.func.value)
                if base in modules:
                    found = True
    checks.require(found, "The real entry point does not call the tested repository factory")
    checks.details["entry_link"] = f"{spec['entry_file']} -> {spec['factory_file']}:{spec['factory_name']} (static call evidence)"


def _base(dataset):
    seen = set()
    while id(dataset) not in seen:
        seen.add(id(dataset))
        other = getattr(dataset, "dataset", None)
        if other is None:
            break
        dataset = other
    return dataset


def _transforms(value):
    if value is None:
        return []
    return [type(value).__name__] + [name for item in getattr(value, "transforms", ()) for name in _transforms(item)]


def _build(root, spec, case, checks):
    import torch
    factory = _load(root, spec["factory_file"], spec["factory_name"])
    checks.require(callable(factory), "Components factory is not callable")
    def create():
        controlled = dict(dataset_name=case['name'], data_root=str(root / spec['data_root']),
                          download=False, device='cpu')
        if spec.get('interface_mode') == 'component_adapter':
            kwargs = {parameter: controlled[key] for key, parameter in spec['argument_map'].items()}
        else:
            kwargs = controlled
        return factory(**kwargs, **spec.get('factory_kwargs', {}))
    folder = sys.modules.get('torchvision.datasets.folder')
    if folder is not None:
        from unittest.mock import patch
        original_init = folder.ImageFolder.__init__
        candidate_root = root
        def local_images(instance, root, *args, **kwargs):
            path = Path(root).resolve()
            checks.require(any(path.is_relative_to(candidate_root / parent) for parent in ('data', 'datasets')),
                           'ImageFolder must use copied data inside the temporary repository, not an external folder')
            return original_init(instance, root, *args, **kwargs)
        with patch.object(folder.ImageFolder, '__init__', local_images):
            objects = create()
    else:
        objects = create()
    if isinstance(objects, (tuple, list)) and len(objects) == 3:
        objects = dict(zip(('train_dataset', 'test_dataset', 'model'), objects))
    checks.require(isinstance(objects, dict), "Factory must return a components dictionary or three-item tuple")
    objects['_has_bound_wrapper'] = bool(spec.get('source_bindings'))
    for key in ("train_dataset", "test_dataset", "model"):
        checks.require(key in objects, f"Factory missing {key}")
    for key in ("train_dataset", "test_dataset"):
        ds = objects[key]
        checks.require(isinstance(ds, torch.utils.data.Dataset), f"{key} is not a PyTorch Dataset")
        checks.require(len(ds) > 0, f"{key} is empty")
        known = {"svhn": "SVHN", "cifar10": "CIFAR10", "cifar100": "CIFAR100", "mnist": "MNIST",
                 "fashionmnist": "FashionMNIST", "stl10": "STL10", "celeba": "CelebA"}
        alias = case["name"].lower().replace("-", "").replace("_", "")
        if alias in known:
            import torchvision.datasets as datasets
            base = _base(ds)
            checks.require(isinstance(base, getattr(datasets, known[alias])),
                           f"{case['name']} must use its real loader, not a synthetic replacement")
            training = key == "train_dataset"
            if hasattr(base, "train"):
                checks.require(bool(base.train) == training, "Dataset uses the wrong official train/test split")
            elif hasattr(base, "split"):
                checks.require(base.split == ("train" if training else "test"), "Dataset uses the wrong official train/test split")
    checks.require(isinstance(objects["model"], torch.nn.Module), "Factory model is not a PyTorch Module")
    # Bind any existing wrapper to its actual moved/refactored implementation.
    for binding in spec.get("source_bindings", []):
        expected = _load(root, binding["candidate_file"], binding["candidate_symbol"])
        if isinstance(expected, type) and issubclass(expected, torch.utils.data.Dataset):
            ds = objects["train_dataset"]
            matched, visited = False, set()
            while id(ds) not in visited:
                visited.add(id(ds))
                matched = matched or isinstance(ds, expected)
                if not hasattr(ds, "dataset"):
                    break
                ds = ds.dataset
            checks.require(matched, "The tested training data bypasses a required original wrapper")
    return objects


def _sample(objects, case, checks):
    import torch
    sample = objects["train_dataset"][0]
    checks.require(isinstance(sample, (list, tuple)) and len(sample) >= 2, "Dataset sample must include inputs and target")
    images, label = sample[:2]
    views = list(images) if isinstance(images, (tuple, list)) else [images]
    checks.require(len(views) == case["views"], "Unexpected number of data views")
    for image in views:
        checks.require(isinstance(image, torch.Tensor), "Transformed input is not a Tensor")
        checks.require(list(image.shape) == case["input_shape"], "Input shape does not match the integration plan")
        checks.require(torch.is_floating_point(image) and bool(torch.isfinite(image).all()), "Invalid transformed values")
    checks.require(0 <= int(label) < case["num_classes"], "Label outside expected class range")
    forbidden = set(case.get("forbidden_operations", []))
    alias = case["name"].lower().replace("-", "").replace("_", "")
    if alias in {"svhn", "mnist", "fashionmnist"}:
        forbidden.update({"RandomHorizontalFlip", "mirror"})
    transforms, visited = [], set()
    ds = objects["train_dataset"]
    while id(ds) not in visited:
        visited.add(id(ds))
        transforms.extend(_transforms(getattr(ds, "transform", None)))
        if not hasattr(ds, "dataset"):
            break
        ds = ds.dataset
    checks.require(not forbidden.intersection(transforms), "Dataset transforms include a forbidden operation")
    probe = objects.get("augmentation_probe")
    if case.get("internal_augmentation") or (objects.get('_has_bound_wrapper') and len(views) > 1):
        checks.require(callable(probe), "Provide the actual augmentation_probe for internal augmentation coverage")
        checks.require(Path(inspect.getsourcefile(probe)).resolve().is_relative_to(Path.cwd()), "Augmentation probe must come from the candidate repository")
        import numpy as np
        from unittest.mock import patch
        choice = np.random.choice
        offered = []
        def checked_choice(values, *args, **kwargs):
            if isinstance(values, (list, tuple)) and values and callable(values[0]):
                names = [getattr(op, "__name__", "") for op in values]
                offered.extend(names)
                checks.require(not forbidden.intersection(names), "Internal augmentation offers a forbidden operation")
            return choice(values, *args, **kwargs)
        raw = _base(objects["train_dataset"])[0][0]
        with patch.object(np.random, "choice", checked_choice):
            probe(raw)
        checks.require(bool(offered), "Internal augmentation choices were not observed; this adapter is not yet supported")
    checks.details.update(input_shape=case["input_shape"], views=len(views))


def _batch(objects, case, checks):
    import torch
    loader = torch.utils.data.DataLoader(objects["train_dataset"], batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    checks.require(isinstance(batch, (list, tuple)) and len(batch) >= 2, "Batch must contain inputs and targets")
    images, labels = batch[:2]
    views = list(images) if isinstance(images, (list, tuple)) else [images]
    checks.require(len(views) == case["views"], "Batch view count changed")
    for view in views:
        checks.require(list(view.shape) == [2, *case["input_shape"]], "Batch shape mismatch")
        checks.require(torch.is_floating_point(view) and bool(torch.isfinite(view).all()), "Invalid batch values")
    checks.require(labels.dtype == torch.int64 and tuple(labels.shape) == (2,), "Classification targets must be int64 [batch]")
    checks.require(bool(((labels >= 0) & (labels < case["num_classes"])).all()), "Batch target outside class range")
    checks.details["batch_shape"] = list(views[0].shape)
    return views, labels


def _step(objects, case, checks):
    import torch
    import torch.nn.functional as F
    views, labels = _batch(objects, case, checks)
    net = objects["model"].cpu().train()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
    optimizer.zero_grad()
    logits = net(torch.cat(views, dim=0))
    if case.get("logits_index") is not None:
        logits = logits[case["logits_index"]]
    checks.require(isinstance(logits, torch.Tensor) and tuple(logits.shape) == (2 * len(views), case["num_classes"]), "Model output incompatible with targets")
    chunks = torch.split(logits, 2)
    loss = F.cross_entropy(chunks[0], labels)
    if case["loss"] == "jsd":
        checks.require(len(views) == 3, "JSD requires three views")
        p = [F.softmax(v, dim=1) for v in chunks]
        mixture = torch.clamp(sum(p) / 3, 1e-7, 1).log()
        loss += 12 * sum(F.kl_div(mixture, v, reduction="batchmean") for v in p) / 3
    checks.require(bool(torch.isfinite(loss)), "Non-finite classification loss")
    loss.backward()
    params = [p for p in net.parameters() if p.grad is not None]
    checks.require(bool(params) and all(bool(torch.isfinite(p.grad).all()) for p in params), "Missing or invalid model gradients")
    changed = next((p for p in params if bool((p.grad != 0).any())), None)
    checks.require(changed is not None, "All model gradients are zero")
    before = changed.detach().clone()
    optimizer.step()
    checks.require(not torch.equal(before, changed.detach()), "Optimizer did not update parameters")
    checks.details.update(loss=float(loss.detach()), cpu_step=True, model_output_shape=list(logits.shape))


def run_stage(root, stage, spec):
    root = Path(root).resolve()
    sys.path.insert(0, str(root))
    checks = Checks()
    if spec.get('interface_mode') == 'component_adapter':
        checks.details.update(validation_scope='component_data_flow_only', entry_link='not_verified',
                              full_cli_verified=False)
    else:
        _entry_uses_factory(root, spec, checks)
    import torch
    torch.set_num_threads(1)
    case = next(c for c in spec["cases"] if c["name"].casefold() == spec["target_dataset"].casefold())
    cases = [c for c in spec["cases"] if c is not case] if stage == "regression" else [case]
    checks.require(bool(cases), "Regression requires at least one original dataset")
    opened = set()
    data_root = (root / spec["data_root"]).resolve()
    def audit(event, args):
        if event == "socket.connect":
            raise Blocked("Network access is disabled during contract checks")
        if event == "open" and isinstance(args[0], (str, bytes)):
            path = Path(os.fsdecode(args[0])).resolve()
            if path.is_relative_to(data_root) and path.is_file():
                opened.add(str(path))
    sys.addaudithook(audit)
    for current in cases:
        opened.clear()
        objects = _build(root, spec, current, checks)
        if stage in {"transformation", "dataloader", "regression"}:
            _sample(objects, current, checks)
        if stage in {"dataloader", "regression"}:
            _batch(objects, current, checks)
        if stage == "pipeline":
            _sample(objects, current, checks)
            _step(objects, current, checks)
        if stage in {"transformation", "dataloader", "pipeline", "regression"}:
            test_objects = dict(objects, train_dataset=objects["test_dataset"])
            test_case = dict(current, views=1, internal_augmentation=False)
            _sample(test_objects, test_case, checks)
            test_views, test_labels = _batch(test_objects, test_case, checks)
            objects["model"].cpu().eval()
            with torch.no_grad():
                output = objects["model"](test_views[0])
                if current.get("logits_index") is not None:
                    output = output[current["logits_index"]]
                checks.require(isinstance(output, torch.Tensor) and tuple(output.shape) == (2, current["num_classes"]) and bool(torch.isfinite(output).all()), "Clean evaluation output mismatch")
        if stage == "data_loading":
            for name in ("train_dataset", "test_dataset"):
                sample = objects[name][0]
                checks.require(isinstance(sample, (tuple, list)) and len(sample) >= 2,
                               "Real loader must read a sample with inputs and label")
            checks.details.update(train_size=len(objects["train_dataset"]), test_size=len(objects["test_dataset"]))
        checks.require(bool(opened), "No real copied dataset file was read; synthetic data cannot certify integration")
        if stage == "regression":
            _step(objects, current, checks)
            checks.details.setdefault("regressions", []).append(current["name"])
        del objects
    return {"status": "passed", "checks_executed": checks.count, "details": checks.details}


def main():
    root, stage, receipt, spec_json = sys.argv[1:]
    report = {"profile": "tensor_classification", "stage": stage, "status": "failed", "checks_executed": 0}
    try:
        report.update(run_stage(root, stage, json.loads(spec_json)))
    except (Blocked, ModuleNotFoundError, FileNotFoundError) as exc:
        report.update(status="blocked", reason=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    except RuntimeError as exc:
        missing = any(text in str(exc).lower() for text in ('dataset not found', 'dataset not found or corrupted', 'data not found'))
        report.update(status='blocked' if missing else 'failed', reason=f'{type(exc).__name__}: {exc}')
        traceback.print_exc()
    except BaseException as exc:
        report.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    Path(receipt).write_text(json.dumps(report), encoding="utf-8")
    print(json.dumps(report), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
