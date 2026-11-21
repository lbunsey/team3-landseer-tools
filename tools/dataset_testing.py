"""CPU smoke checks in disposable copies. This is NOT an OS security sandbox.

Tests and imported repository code execute with the user's process permissions.
Trusted terminal consent is required; never treat model-provided consent as approval.
"""
import json
import ast
import hashlib
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from . import patch_store, patch_utils
from .dataset_preflight import validate_candidate
from pydantic import BaseModel, Field
from typing import Literal

STAGES = ("data_loading", "transformation", "dataloader", "pipeline", "regression")
MAX_COPY_BYTES = 100_000_000
MAX_FILES = 5000
MAX_DATA_BYTES = 1_000_000_000
MAX_DATA_FILES = 30000
SKIP_DIRS = {".git", "venv", ".venv", "__pycache__", "data", "datasets", "checkpoint", "checkpoints", "runs", "wandb"}
_certificates = {}


def _framework_hash():
    digest = hashlib.sha256()
    for name in ('dataset_testing.py', 'dataset_contract.py', 'dataset_preflight.py'):
        digest.update(name.encode())
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def invalidate(patch_ids):
    for pid in patch_ids:
        _certificates.pop(pid, None)


def _data_manifest(root, paths, destination=None):
    """Copy explicitly selected local data, with no links to the live repo.

    Paths must be inside a data/ or datasets/ directory in the repository.
    Hash while copying; no downloads, symlinks, hard links or external mounts.
    """
    files, total = {}, 0
    for relative in paths:
        parts = patch_utils._path_parts(relative)
        if parts[0] not in {"data", "datasets"} or any(p.startswith(".") for p in parts):
            raise ValueError("dataset_paths must be inside repository data/ or datasets/")
        target = root
        for part in parts:
            target = target / part
            patch_utils._reject_link(target)
        if not target.resolve().is_relative_to(root.resolve()):
            raise ValueError("Dataset path escapes the repository")
        if not target.exists():
            raise FileNotFoundError(f"Local dataset is missing: {relative}; prepare it before retrying")
        pending = [target]
        while pending:
            path = pending.pop()
            patch_utils._reject_link(path)
            if path.name.startswith("."):
                raise ValueError("Hidden dataset paths are excluded")
            if path.is_dir():
                pending.extend(sorted(path.iterdir(), reverse=True))
                continue
            if not path.is_file():
                raise ValueError("Dataset input must be regular files or directories")
            rel = path.relative_to(root).as_posix()
            if rel in files:
                continue
            size = path.stat().st_size
            total += size
            if total > MAX_DATA_BYTES or len(files) >= MAX_DATA_FILES:
                raise ValueError("Local dataset copy exceeds 1 GB or 30000 files; select a smaller input")
            digest = hashlib.sha256()
            output = None
            try:
                if destination is not None:
                    dest = destination / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    output = dest.open("wb")
                actual = 0
                with path.open("rb") as stream:
                    while chunk := stream.read(1024 * 1024):
                        actual += len(chunk)
                        if actual > size:
                            raise ValueError("Dataset changed during copy")
                        digest.update(chunk)
                        if output:
                            output.write(chunk)
                if actual != size:
                    raise ValueError("Dataset changed during copy")
                files[rel] = digest.hexdigest()
            finally:
                if output:
                    output.close()
    return files


def _files(root):
    files, total = {}, 0
    for folder, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d not in SKIP_DIRS)
        for d in dirs:
            path = Path(folder) / d
            if path.is_symlink() or (os.name == "nt" and path.lstat().st_file_attributes & 0x400):
                raise ValueError("Linked directories are not copied")
        for name in sorted(names):
            path = Path(folder) / name
            if name.startswith(".") or path.suffix.lower() not in patch_utils.TEXT_EXTENSIONS:
                continue
            target = patch_utils.resolve_target(root.name, path.relative_to(root).as_posix())
            data = target.read_bytes()
            total += len(data)
            if total > MAX_COPY_BYTES or len(files) >= MAX_FILES:
                raise ValueError("Scratch source copy exceeds limits")
            files[path.relative_to(root).as_posix()] = data
    return files


def _check_bases(records):
    for p in records:
        if p.status not in {patch_store.PatchStatus.PENDING_REVIEW, patch_store.PatchStatus.APPROVED, patch_store.PatchStatus.APPLIED}:
            raise ValueError("Batch contains a rejected or superseded proposal")
        path = patch_utils.resolve_target(p.repo_name, p.file_path, allow_missing=p.operation == "create")
        if p.status == patch_store.PatchStatus.APPLIED:
            if not path.exists() or path.read_bytes() != p.updated_content:
                raise ValueError("Applied batch file changed")
        elif p.operation == "create":
            if os.path.lexists(path):
                raise ValueError("Create target already exists")
        elif patch_utils.compute_file_hash(path.read_bytes()) != p.original_hash:
            raise ValueError("Batch base changed; inspect and revise")


def assert_tested(patch_id):
    certificate = _certificates.get(patch_id)
    if not certificate:
        raise ValueError("Dataset patch requires a passing automatic smoke test for the latest complete batch")
    ids, root, expected, data_paths, data_manifest, framework_hash = certificate
    if _framework_hash() != framework_hash:
        raise ValueError("Framework checks changed since testing; rerun validation")
    records = [patch_store.get_patch(pid) for pid in ids]
    _check_bases(records)
    current = _files(root)
    # Earlier files of this tested batch may already have been applied.
    for record in records:
        if record.status == patch_store.PatchStatus.APPLIED:
            current.pop(record.file_path, None)
    original = dict(expected)
    for record in records:
        if record.status == patch_store.PatchStatus.APPLIED:
            original.pop(record.file_path, None)
    if current != original:
        raise ValueError("Repository source changed since testing; rerun smoke tests")
    if _data_manifest(root, data_paths) != data_manifest:
        raise ValueError("Local test data changed since testing; rerun smoke tests")


class DatasetSettings(BaseModel):
    num_classes: int | None = Field(default=None, ge=2, le=100000)
    input_shape: list[int] | None = Field(default=None, min_length=1, max_length=4)
    views: int = Field(default=1, ge=1, le=3)
    loss: Literal["ce", "jsd"] = "ce"
    logits_index: int | None = None
    forbidden_operations: list[str] = Field(default_factory=list)
    internal_augmentation: bool = False


class DatasetCase(DatasetSettings):
    name: str


class SourceBinding(BaseModel):
    original_file: str
    original_symbol: str
    candidate_file: str
    candidate_symbol: str


class IntegrationContract(BaseModel):
    entry_file: str
    entry_function: str = ""
    factory_file: str
    factory_name: str = "build_dataset_components"
    interface_mode: Literal['shared_factory', 'component_adapter'] = 'shared_factory'
    argument_map: dict[str, str] = Field(default_factory=dict, description='component_adapter only: controlled name to existing parameter name, e.g. dataset_name -> dataset. Omit unsupported device/download; never override offline controls.')
    data_root: str = "data"
    data_files: list[str] = Field(min_length=1)
    target_case: DatasetSettings | None = Field(default=None, description="Settings for the task's target dataset. Do not repeat its name: the program supplies it from the task.")
    cases: list[DatasetCase] = Field(default_factory=list, max_length=20, description="Settings for original datasets reported by inspect_contract. Legacy target entries are also accepted; identical duplicates are merged, conflicting duplicates rejected.")
    factory_kwargs: dict = Field(default_factory=dict)
    source_bindings: list[SourceBinding] = Field(default_factory=list)


CONTROLLED_ARGUMENTS = ('dataset_name', 'data_root', 'download', 'device')


class ContractMappingError(ValueError):
    def __init__(self, issues):
        self.mapping_issues = issues
        super().__init__('; '.join(f"{item['field']}: {item['reason']}" for item in issues))


def _callable_signatures(tree):
    """Describe formal parameters without imports or evaluating defaults."""
    functions = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        positional = node.args.posonlyargs + node.args.args
        required_count = len(positional) - len(node.args.defaults)
        parameters = [dict(name=arg.arg, kind='positional_only' if i < len(node.args.posonlyargs) else 'positional_or_keyword',
                           required=i < required_count) for i, arg in enumerate(positional)]
        parameters += [dict(name=arg.arg, kind='keyword_only', required=default is None)
                       for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults)]
        returns, pending = [], list(node.body)
        while pending:
            child = pending.pop()
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                continue
            if isinstance(child, ast.Return):
                returns.append(child.value)
            pending.extend(ast.iter_child_nodes(child))
        values = [value for value in returns if value is not None
                  and not (isinstance(value, ast.Constant) and value.value is None)]
        def components_hint(value):
            if isinstance(value, (ast.Tuple, ast.List)):
                return len(value.elts) == 3
            required = {'train_dataset', 'test_dataset', 'model'}
            if isinstance(value, ast.Dict):
                return required.issubset({k.value for k in value.keys if isinstance(k, ast.Constant)})
            return (isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == 'dict'
                    and required.issubset({k.arg for k in value.keywords}))
        functions.append(dict(name=node.name, line=node.lineno, parameters=parameters[:64],
                              parameters_truncated=len(parameters) > 64,
                              accepts_extra_keywords=node.args.kwarg is not None,
                              asynchronous=isinstance(node, ast.AsyncFunctionDef), decorated=bool(node.decorator_list),
                              has_value_return=bool(values), component_return_hint=any(components_hint(value) for value in values),
                              only_scalar_returns=bool(values) and all(isinstance(value, ast.Constant) for value in values)))
    return functions


def inspect_contract_parameters(root, contract, candidate_files=None):
    """Bounded AST guidance for the declared callable; never execute repo code."""
    data = contract.model_dump() if hasattr(contract, 'model_dump') else contract
    root = Path(root).resolve()
    reports = []
    for relative in dict.fromkeys([data['factory_file'], data['entry_file']]):
        report = dict(file_path=relative)
        try:
            clean = relative.replace('\\', '/')
            while clean.startswith('./'):
                clean = clean[2:]
            clean = clean.rstrip('/')
            parts = patch_utils._path_parts(clean)
            if any(p.startswith('.') for p in parts) or Path(clean).suffix.lower() != '.py':
                raise ValueError('Use a non-hidden repository-relative Python file')
            if candidate_files and clean in candidate_files:
                raw = candidate_files[clean]
                report['source'] = 'saved_candidate'
            else:
                path = root
                for part in parts:
                    path = path / part
                    patch_utils._reject_link(path)
                if not path.resolve().is_relative_to(root) or path.stat().st_size > patch_utils.MAX_FILE_BYTES:
                    raise ValueError('Source is outside the repository or exceeds the read limit')
                with path.open('rb') as stream:
                    raw = stream.read(patch_utils.MAX_FILE_BYTES + 1)
                report['source'] = 'original'
            if len(raw) > patch_utils.MAX_FILE_BYTES:
                raise ValueError('Source exceeds the read limit')
            functions = _callable_signatures(ast.parse(raw.decode('utf-8-sig')))
            report.update(status='read', callables=functions[:20])
            if relative == data['factory_file']:
                target = next((f for f in functions if f['name'] == data['factory_name']), None)
                report['declared_callable'] = target
                if target is None:
                    report['note'] = 'Declared callable not found in this source. Do not invent its parameters; inspect existing functions or save the proposed helper before rechecking.'
                else:
                    names = {p['name'] for p in target['parameters'] if p['kind'] != 'positional_only'}
                    report['exact_name_mapping_suggestion'] = {name: name for name in CONTROLLED_ARGUMENTS if name in names}
                    report['mapped_parameters_not_in_signature'] = sorted(set(data.get('argument_map', {}).values()) - names) if not target['accepts_extra_keywords'] else []
        except (ValueError, OSError, SyntaxError, UnicodeError) as exc:
            report.update(status='unavailable', reason=f'{type(exc).__name__}: {exc}')
        reports.append(report)
    return dict(controlled_names=list(CONTROLLED_ARGUMENTS), files=reports,
                note='Source signature evidence only, not semantic mapping or runtime validation. Suggestions are exact name matches; do not map to args.dataset, literal values or data paths.')


def assess_component_interface(guidance):
    """Identify definite entry/adapter mismatches, not arbitrary model semantics."""
    reports = guidance.get('files', [])
    if not reports:
        return dict(status='unknown', issues=[])
    report = reports[0]
    target = report.get('declared_callable')
    issues = []
    if report.get('status') == 'unavailable':
        if not report.get('reason', '').startswith('FileNotFoundError:'):
            return dict(status='unknown', issues=[])
        issues.append(dict(code='component_source_missing', reason='The declared component source has not been saved.'))
    elif target is None:
        issues.append(dict(code='component_callable_missing', reason='The declared callable does not exist in the inspected source.'))
    elif target.get('decorated') or target.get('parameters_truncated'):
        return dict(status='requires_runtime_verification', issues=[],
                    note='Static signature/return behavior is uncertain; no adapter is forced from this evidence.')
    else:
        keywords = [p for p in target['parameters'] if p['kind'] != 'positional_only']
        if not target['accepts_extra_keywords'] and len(keywords) < 2:
            issues.append(dict(code='insufficient_component_parameters',
                reason='This callable cannot accept distinct dataset and data-root inputs through the supported interface.'))
        if not target['has_value_return'] or target.get('only_scalar_returns'):
            issues.append(dict(code='no_component_return',
                reason='This callable has no component-valued return; a CLI/side-effect entry is not a dataset/model builder.'))
        if target['asynchronous']:
            issues.append(dict(code='async_component_callable', reason='The current CPU worker expects a synchronous component builder.'))
    if not issues:
        return dict(status='interface_candidate', issues=[],
                    note='No definite structural mismatch found. Real data flow and output types still require CPU validation.')
    alternatives = [dict(file_path=f['file_path'], name=fn['name'], line=fn['line'])
                    for f in reports for fn in f.get('callables', [])
                    if fn.get('component_return_hint') and not fn['asynchronous']
                    and fn is not target
                    and (fn['accepts_extra_keywords'] or len([p for p in fn['parameters'] if p['kind'] != 'positional_only']) >= 2)]
    return dict(status='needs_component_selection' if alternatives else 'needs_component_adapter',
                issues=issues, existing_component_leads=alternatives,
                note='Structural source evidence only. Existing leads are not certified correct or import-safe.')


def _symbol(tree, symbol):
    current = tree
    for part in symbol.split("."):
        current = next((n for n in current.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == part), None)
        if current is None:
            raise ValueError(f"Repository symbol not found: {symbol}")
    return current


class ContractRequirementsError(ValueError):
    def __init__(self, issues, requirements):
        self.issues = issues
        self.requirements = requirements
        super().__init__("; ".join(issues))


def canonical_dataset_name(name):
    value = name.strip()
    key = value.casefold().replace('-', '').replace('_', '').replace(' ', '')
    known = {'svhn', 'cifar10', 'cifar100', 'mnist', 'fashionmnist', 'stl10', 'celeba'}
    return key if key in known else value


def contract_requirements(root, entry_file, target_dataset=None):
    """Read-only hints from the selected entry; never guesses research choices."""
    path = Path(root)
    for part in patch_utils._path_parts(entry_file):
        if part.startswith('.'):
            raise ValueError("Hidden entry paths are not permitted")
        path = path / part
        patch_utils._reject_link(path)
    original = ast.parse(path.read_text(encoding="utf-8-sig"))
    aliases = {"CIFAR10": "cifar10", "CIFAR100": "cifar100", "SVHN": "svhn",
               "MNIST": "mnist", "FashionMNIST": "fashionmnist", "STL10": "stl10"}
    imported = {a.asname or a.name: aliases[a.name]
                for n in ast.walk(original) if isinstance(n, ast.ImportFrom)
                and n.module and n.module.startswith('torchvision')
                for a in n.names if a.name in aliases}
    loaders = set()
    for n in ast.walk(original):
        if isinstance(n, ast.Call):
            if isinstance(n.func, ast.Attribute) and n.func.attr in aliases:
                loaders.add(aliases[n.func.attr])
            elif isinstance(n.func, ast.Name) and n.func.id in imported:
                loaders.add(imported[n.func.id])
    wrappers = {n.name: n.lineno for n in original.body if isinstance(n, ast.ClassDef)
                and any(ast.unparse(base).endswith('Dataset') for base in n.bases)}
    used = {n.func.id for n in ast.walk(original) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    result = dict(entry_file=entry_file, original_dataset_cases=sorted(loaders),
                callable_signatures=_callable_signatures(original)[:20],
                required_source_bindings=[dict(original_file=entry_file, original_symbol=name,
                                               line=wrappers[name], candidate_file=entry_file,
                                               candidate_symbol=name) for name in sorted(wrappers.keys() & used)],
                path_rules="Use paths relative to the repository, without ./ or ../. Data paths must start with data/ or datasets/; never use placeholders such as <path>.",
                note="Static leads from this entry only, not exhaustive discovery. Inspect referenced modules too. Bindings identify real original wrappers; choose candidate locations after analyzing the code, not test-only copies.")
    if target_dataset is not None:
        target = canonical_dataset_name(target_dataset)
        result.update(target_case_name=target, required_case_names=list(dict.fromkeys([target, *sorted(loaders)])),
                      target_case_fields=['num_classes', 'input_shape'],
                      settings_instructions='Use target_case for target settings without a name. Supply cases for required original datasets. Names come from this task and source evidence; never invent missing class counts or shapes.')
    return result


def normalize_cases(data, dataset_name, requirements):
    """Bind settings to program-owned identities without guessing any settings."""
    target = canonical_dataset_name(dataset_name)
    expected = requirements['required_case_names']
    identity = {name.casefold(): name for name in expected}
    received = [c['name'] for c in data['cases']]
    supplied = list(data['cases'])
    if data.get('target_case') is not None:
        supplied.insert(0, dict(data['target_case'], name=target))
    normalized, conflicts, issues = {}, [], []
    for item in supplied:
        name = canonical_dataset_name(item['name'])
        if not name:
            issues.append('A dataset case has an empty name. Use target_case for unnamed target settings.')
            continue
        name = identity.get(name.casefold(), name)
        case = dict(item, name=name)
        key = name.casefold()
        if key in normalized:
            fields = [field for field in case if field != 'name' and case[field] != normalized[key][field]]
            if fields:
                conflicts.append(dict(name=name, fields=fields))
                issues.append(f"Conflicting settings for {name}: {', '.join(fields)}. Supply one agreed configuration; no version was selected.")
        else:
            normalized[key] = case
    missing = []
    for name in expected:
        case = normalized.get(name.casefold())
        fields = [field for field in ('num_classes', 'input_shape') if case is None or case.get(field) is None]
        if fields:
            missing.append(dict(name=name, fields=fields))
            label = 'target dataset' if name == target else 'original dataset'
            issues.append(f"Missing settings for {label} {name}: {', '.join(fields)}.")
    for case in normalized.values():
        if case['name'] not in expected:
            fields = [f for f in ('num_classes', 'input_shape') if case.get(f) is None]
            if fields:
                issues.append(f"Missing settings for additional dataset {case['name']}: {', '.join(fields)}.")
    requirements.update(received_case_names=received, missing_case_settings=missing,
                        conflicting_case_settings=conflicts)
    data['cases'] = list(normalized.values())
    data['target_case_name'] = target
    return issues


def validate_contract(root, contract, dataset_name):
    root = Path(root).resolve()
    data = contract.model_dump()
    def canonical_path(value):
        value = value.replace('\\', '/')
        while value.startswith('./'):
            value = value[2:]
        return value.rstrip('/')
    for key in ('entry_file', 'factory_file', 'data_root'):
        data[key] = canonical_path(data[key])
    data['data_files'] = [canonical_path(value) for value in data['data_files']]
    def checked_path(relative):
        parts = patch_utils._path_parts(relative)
        if any(p.startswith(".") for p in parts):
            raise ValueError("Hidden contract paths are not permitted")
        path = root
        for part in parts:
            path = path / part
            if path.exists() or path.is_symlink():
                patch_utils._reject_link(path)
        if not path.resolve().is_relative_to(root):
            raise ValueError("Contract path escapes repository")
        return path
    for relative in [data["entry_file"], data["factory_file"], data["data_root"], *data["data_files"]]:
        try:
            checked_path(relative)
        except (ValueError, OSError) as exc:
            raise ValueError(f"Invalid contract path {relative!r}: {exc}. Use repository-relative paths without ./, ../ or placeholders.") from exc
    if not (root / data["entry_file"]).is_file():
        raise ValueError("Contract must reference an existing real entry point")
    if Path(data["factory_file"]).suffix != ".py" or not data["factory_name"].isidentifier():
        raise ValueError("Factory must reference an import-safe Python file and function")
    mapping = data['argument_map']
    mapping_issues = []
    for key, value in mapping.items():
        if key not in CONTROLLED_ARGUMENTS:
            mapping_issues.append(dict(field=f'argument_map.{key}', received=value,
                reason='Unknown controlled name. Mapping direction is controlled name -> actual formal parameter.',
                allowed_names=list(CONTROLLED_ARGUMENTS)))
        if not value.isidentifier():
            mapping_issues.append(dict(field=f'argument_map.{key}', received=value,
                reason='Target must be a formal parameter identifier, not an expression such as args.dataset, a literal, or a filesystem path.'))
    if len(set(mapping.values())) != len(mapping):
        mapping_issues.append(dict(field='argument_map', reason='Target parameter names must be distinct.'))
    if data['interface_mode'] == 'shared_factory' and mapping:
        mapping_issues.append(dict(field='interface_mode', received=data['interface_mode'],
            reason='argument_map is only supported for component_adapter; shared_factory uses standard controlled parameter names.'))
    if data['interface_mode'] == 'component_adapter' and not {'dataset_name', 'data_root'}.issubset(mapping):
        for key in sorted({'dataset_name', 'data_root'} - set(mapping)):
            mapping_issues.append(dict(field=f'argument_map.{key}', received=None,
                reason='Required controlled argument is missing; bind it to an actual formal parameter.'))
    if set(mapping.values()) & set(data['factory_kwargs']):
        mapping_issues.append(dict(field='factory_kwargs', reason='factory_kwargs cannot override argument_map parameters'))
    if mapping_issues:
        raise ContractMappingError(mapping_issues)
    if Path(data["data_root"]).parts[0] not in {"data", "datasets"}:
        raise ValueError("Data root must be inside data/ or datasets/")
    requirements = contract_requirements(root, data['entry_file'], dataset_name)
    issues = normalize_cases(data, dataset_name, requirements)
    for case in data["cases"]:
        case["name"] = case["name"].strip()
        if not case["name"]:
            raise ValueError("Case names must be non-empty")
        if (case["loss"], case["views"]) not in {("ce", 1), ("jsd", 3)}:
            raise ValueError("Supported contracts are single-view CE or three-view JSD")
        if case['input_shape'] is not None and any(type(n) is not int or n <= 0 or n > 100000 for n in case["input_shape"]):
            raise ValueError("Input shapes must contain positive dimensions")
    if set(data["factory_kwargs"]) & {"dataset_name", "data_root", "download", "device"}:
        raise ValueError("Factory kwargs cannot override controlled dataset/device/download arguments")
    if len(json.dumps(data)) > 64000:
        raise ValueError("Contract exceeds metadata size limit")
    # Do not let the adapter quietly omit original loader branches in the entry.
    original = ast.parse((root / data["entry_file"]).read_text(encoding="utf-8-sig"))
    aliases = {"CIFAR10": "cifar10", "CIFAR100": "cifar100", "SVHN": "svhn", "MNIST": "mnist", "FashionMNIST": "fashionmnist", "STL10": "stl10"}
    original_names = {aliases[n.func.attr] for n in ast.walk(original) if isinstance(n, ast.Call)
                      and isinstance(n.func, ast.Attribute) and n.func.attr in aliases}
    normalized = {canonical_dataset_name(c['name']).casefold() for c in data['cases']}
    original_names.update(requirements['original_dataset_cases'])
    if not original_names.issubset(normalized):
        absent = sorted(original_names - normalized)
        issues = [issue for issue in issues if not any(issue.startswith('Missing settings for original dataset ' + name + ':') for name in absent)]
        issues.append("Regression cases omit existing loaders: " + ", ".join(absent) + ". Supply num_classes and input_shape for each missing original dataset.")
    if len(normalized) < 2:
        issues.append('Provide settings for the target and at least one distinct original dataset; duplicates do not count as regression coverage.')
    # Original wrappers used by the entry must remain real, not test-only copies.
    wrappers = {n.name for n in original.body if isinstance(n, ast.ClassDef)
                and any(ast.unparse(base).endswith("Dataset") for base in n.bases)}
    used = {n.func.id for n in ast.walk(original) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    bound = {b["original_symbol"] for b in data["source_bindings"] if b["original_file"] == data["entry_file"]}
    if not (wrappers & used).issubset(bound):
        issues.append("Bind the original training wrapper into the shared factory: " + ", ".join(sorted((wrappers & used) - bound)))
    if issues:
        raise ContractRequirementsError(issues, requirements)
    for binding in data["source_bindings"]:
        for key in ("original_file", "candidate_file"):
            checked_path(binding[key])
        _symbol(ast.parse((root / binding["original_file"]).read_text(encoding="utf-8-sig")), binding["original_symbol"])
    return data


def resolve_profile(root, dataset_name, contract=None):
    return "tensor_classification" if contract else None


def profile_description(profile):
    return {"profile": profile, "checks": list(STAGES),
        "contract": "Provide an import-safe factory(dataset_name, data_root, download=False, device='cpu', **kwargs), used by the real existing entry point. Return a dictionary with train_dataset, test_dataset, model or a three-item tuple in that order; optional augmentation_probe is the actual repo function. The framework owns all assertions. No test scripts.",
        "scope": "Tensor multiclass classification with CE or three-view JSD. Other tasks must remain blocked until a suitable contract exists; scientific metrics/full training are not certified.",
        "schema": _inline_contract_schema()}


def _inline_contract_schema():
    """Return self-contained tool-result metadata, without provider-sensitive refs.

    Gemini can interpret $ref values in a function response as references to
    response parts. Pydantic's local JSON Schema definitions are not such parts.
    Keep the tool's input validation unchanged; expand only this result metadata.
    """
    schema = IntegrationContract.model_json_schema()
    definitions = schema.get("$defs", {})

    def expand(value, ancestors=()):
        if isinstance(value, list):
            return [expand(item, ancestors) for item in value]
        if not isinstance(value, dict):
            return value
        reference = value.get("$ref")
        if reference is not None:
            prefix = "#/$defs/"
            if not reference.startswith(prefix) or reference in ancestors:
                raise ValueError("Contract metadata requires non-recursive local schema references")
            name = reference[len(prefix):]
            resolved = expand(definitions[name], ancestors + (reference,))
            resolved.update(expand({k: v for k, v in value.items() if k != "$ref"}, ancestors))
            return resolved
        return {key: expand(item, ancestors) for key, item in value.items() if key != "$defs"}

    return expand(schema)


def inspect_local_data(root, contract=None):
    files = contract.get("data_files", []) if contract else []
    missing = []
    for relative in files:
        parts = patch_utils._path_parts(relative)
        if parts[0] not in {"data", "datasets"} or any(p.startswith(".") for p in parts):
            raise ValueError("Local inputs must be inside repository data/ or datasets/")
        path = root
        for part in parts:
            path = path / part
            if path.exists() or path.is_symlink():
                patch_utils._reject_link(path)
        if not path.exists():
            missing.append(relative)
    return {"ready": bool(files) and not missing, "missing": missing,
            "contract_required": not bool(files), "recommended_dataset_paths": files,
            "note": "Presence only; real loaders validate contents. No downloads performed."}


def validate_source_bindings(source, scratch, contract):
    warnings = []
    for binding in contract.get("source_bindings", []):
        before = _symbol(ast.parse(source[binding["original_file"]].decode("utf-8-sig")), binding["original_symbol"])
        candidate = (scratch[binding['candidate_file']].decode('utf-8-sig') if isinstance(scratch, dict)
                     else (scratch / binding["candidate_file"]).read_text(encoding="utf-8-sig"))
        after = _symbol(ast.parse(candidate), binding["candidate_symbol"])
        if ast.dump(before, include_attributes=False) != ast.dump(after, include_attributes=False):
            original_methods = {n.name for n in getattr(before, 'body', []) if isinstance(n, ast.FunctionDef)}
            candidate_methods = {n.name: n for n in getattr(after, 'body', []) if isinstance(n, ast.FunctionDef)}
            if not original_methods.issubset(candidate_methods) or any(
                all(isinstance(s, ast.Pass) or (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))
                    for s in candidate_methods[name].body) for name in original_methods):
                raise ValueError("Original wrapper/algorithm changed instead of being connected: " + binding["original_symbol"])
            warnings.append(dict(symbol=binding['original_symbol'], file_path=binding['candidate_file'],
                                 note='Wrapper changed; candidate runtime coverage and original dataset regression are required. Scientific equivalence is not certified.'))
    return warnings


def _profile_command(profile, scratch, stage, receipt, spec):
    if profile != "tensor_classification":
        raise ValueError("No executable framework contract")
    worker = Path(__file__).with_name("dataset_contract.py").resolve()
    return [sys.executable, "-B", str(worker), str(scratch), stage, str(receipt), json.dumps(spec)]


def run_checks(root, patch_ids, timeout_seconds=60, dataset_paths=(), dataset_name="", contract=None):
    """Run framework-owned checks. The model supplies paths, never test code.

    The temporary copy is not an OS security sandbox. Source and local-data
    hashes bind results to exact candidates. No receipt means no pass, even
    with exit code zero. Missing files/dependencies are blocked, not skipped.
    """
    if not 1 <= timeout_seconds <= 120:
        raise ValueError("timeout_seconds must be between 1 and 120 per stage")
    records = [patch_store.get_patch(pid) for pid in patch_ids]
    if not records or len({(p.session_id, p.repo_name) for p in records}) != 1:
        raise ValueError("Tests require a non-empty same-session repository batch")
    _check_bases(records)
    source = _files(root)
    invalidate(patch_ids)
    result = {"status": "blocked", "patch_ids": list(patch_ids), "checks": [],
              "tested_at": datetime.now(timezone.utc).isoformat(), "dataset_paths": list(dataset_paths),
              "note": "Framework checks execute shared repository components and a bounded CPU classification update. Entry linkage is checked statically; the full CLI training/evaluation loop and scientific validity are not verified."}
    profile = resolve_profile(root, dataset_name, contract)
    result["profile"] = profile
    if profile is None:
        result.update(reason="No fixed validation profile for this repository/dataset yet. No agent-authored script can certify it.", blocked_kind="unsupported_profile")
        return result
    framework_hash = _framework_hash()
    spec = dict(contract or {})
    result['validation_scope'] = ('component_data_flow_only' if spec.get('interface_mode') == 'component_adapter'
                                  else 'shared_components_with_static_entry_link')
    result['full_cli_verified'] = False
    if spec.get('interface_mode') == 'component_adapter':
        result['note'] = 'Component adapter checks real data flow only. Entry linkage, full CLI training/evaluation and scientific validity are not verified.'
    spec["target_dataset"] = spec.get('target_case_name', dataset_name)
    result["contract_spec"] = spec
    if profile == "tensor_classification":
        candidate_files = dict(source)
        candidate_files.update({p.file_path: p.updated_content for p in records})
        try:
            result['candidate_preflight'] = validate_candidate(candidate_files, spec)
            validate_source_bindings(source, candidate_files, spec)
        except (ValueError, SyntaxError, KeyError, OSError) as exc:
            result.update(status='failed', reason=str(exc), failed_kind='candidate_preflight',
                          preflight_issues=getattr(exc, 'preflight_issues', []))
            return result
        local = inspect_local_data(root, contract)
        result["local_data"] = local
        if not local["ready"]:
            result.update(reason="Required real local data is missing; prepare the listed paths before testing.", blocked_kind="local_data")
            return result
        dataset_paths = local["recommended_dataset_paths"]
    print("\nFramework dataset checks:", profile, flush=True)
    print("Entry:", spec.get("entry_file"), "Factory:", spec.get("factory_file"), spec.get("factory_name"), flush=True)
    print("Cases:", spec.get("cases", []), flush=True)
    print("The runner executes repository code with your account permissions, not in an OS sandbox. "
          "It performs no downloads, installs or full training. Python-level offline controls do not "
          "provide OS-level memory/network isolation.", flush=True)
    print(f"Local data to copy (max 1 GB): {list(dataset_paths)}", flush=True)
    try:
        answer = input("Execute framework CPU checks in a temporary copy? [Y/N]: ").strip().upper()
    except (EOFError, KeyboardInterrupt):
        answer = "N"
    if answer != "Y":
        result["reason"] = "Execution not approved"
        return result
    with tempfile.TemporaryDirectory(prefix="dataset-smoke-") as temp:
        scratch = Path(temp) / "repo"
        scratch.mkdir()
        try:
            data_manifest = _data_manifest(root, dataset_paths, scratch)
        except (ValueError, OSError) as exc:
            result.update(reason=str(exc), blocked_kind="local_data")
            return result
        for rel, data in source.items():
            target = scratch / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        for record in records:
            target = scratch / record.file_path
            if not target.parent.is_dir():
                result.update(reason="Proposal parent excluded from scratch copy", blocked_kind="source_layout")
                return result
            target.write_bytes(record.updated_content)
        if profile == "tensor_classification":
            try:
                validate_source_bindings(source, scratch, spec)
            except (ValueError, SyntaxError, OSError) as exc:
                result.update(status="failed", reason=str(exc))
                return result
        candidate = _files_from_scratch(scratch)
        environment = {k: v for k, v in os.environ.items()
                       if k.upper() in {"PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "LANG", "LC_ALL"}}
        environment.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                           PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1", WANDB_MODE="disabled")
        environment.update(DATASET_SMOKE_ROOT=str(scratch), DATASET_SMOKE_DATA=str(scratch / spec.get("data_root", "data")))
        for key in ("HOME", "USERPROFILE", "TMPDIR", "TEMP", "TMP"):
            environment[key] = temp
        for stage in STAGES:
            receipt = Path(temp) / f"{stage}.json"
            command = _profile_command(profile, scratch, stage, receipt, spec)
            print(f"[DEBUG][dataset_testing] Framework stage: {stage}", flush=True)
            with tempfile.TemporaryFile() as output:
                process = subprocess.Popen(command, cwd=scratch, env=environment,
                                           stdout=output, stderr=subprocess.STDOUT,
                                           start_new_session=os.name != "nt")
                try:
                    code = process.wait(timeout=timeout_seconds)
                    state = "failed"
                except subprocess.TimeoutExpired:
                    if os.name == "nt":
                        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        process.kill()
                    else:
                        import signal
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    code, state = process.returncode, "timeout"
                output.seek(0)
                log = output.read(24000).decode("utf-8", errors="replace")
            report = {}
            try:
                if receipt.exists() and receipt.stat().st_size <= 64000:
                    report = json.loads(receipt.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                pass
            if state != "timeout":
                valid = isinstance(report, dict) and report.get("profile") == profile and report.get("stage") == stage
                if valid and report.get("status") == "blocked":
                    state = "blocked"
                elif valid and code == 0 and report.get("status") == "passed" and type(report.get("checks_executed")) is int and report["checks_executed"] > 0:
                    state = "passed"
                else:
                    state = "failed"
            result["checks"].append(dict(stage=stage, status=state, returncode=code, log=log, report=report))
            print(f"[DEBUG][dataset_testing] {stage}: {state}", flush=True)
            if state != "passed":
                result["status"] = "blocked" if state == "blocked" else "failed"
                result["reason"] = report.get("reason", "Framework did not produce a valid successful stage receipt") if isinstance(report, dict) else "Invalid stage receipt"
                return result
            if _files_from_scratch(scratch) != candidate:
                result.update(status="failed", reason="Test modified candidate source files")
                return result
            try:
                if _data_manifest(scratch, dataset_paths) != data_manifest:
                    result.update(status="failed", reason="Test modified copied dataset inputs")
                    return result
            except (ValueError, OSError) as exc:
                result.update(status="failed", reason=f"Test modified copied dataset inputs: {exc}")
                return result
        if _files(root) != source or _data_manifest(root, dataset_paths) != data_manifest:
            result.update(status="failed", reason="Repository source or input data changed during testing")
            return result
        _check_bases([patch_store.get_patch(pid) for pid in patch_ids])
        if _framework_hash() != framework_hash:
            result.update(status="failed", reason="Framework changed while tests were running")
            return result
        result["framework_hash"] = framework_hash
        certificate = (tuple(patch_ids), root, source, tuple(dataset_paths), data_manifest, framework_hash)
        for pid in patch_ids:
            _certificates[pid] = certificate
        result["status"] = "passed"
        return result


def _files_from_scratch(root):
    return {p.relative_to(root).as_posix(): p.read_bytes()
            for p in root.rglob("*") if p.is_file() and p.suffix.lower() in patch_utils.TEXT_EXTENSIONS}
