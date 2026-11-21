"""Bounded AST checks before any dataset copy, imports or execution.

These checks reject concrete interface mistakes; passing is not proof of
semantic correctness. Dynamic/indirect expressions still need runtime checks.
"""
import ast
from pathlib import PurePosixPath


class CandidateError(ValueError):
    def __init__(self, issues):
        self.preflight_issues = issues
        super().__init__('Candidate preflight failed: ' + '; '.join(i['message'] for i in issues))


def inspect_candidate(files, spec):
    issues = []
    def report(code, filename, node, message, **details):
        item = dict(code=code, file_path=filename, line=getattr(node, 'lineno', 1), message=message)
        item.update(details)
        if item not in issues:
            issues.append(item)
    def tree(filename):
        if filename not in files:
            report('missing_module', filename, None, 'Candidate module is missing: ' + filename)
            return ast.Module(body=[], type_ignores=[])
        try:
            return ast.parse(files[filename].decode('utf-8-sig'), filename=filename)
        except (UnicodeError, SyntaxError) as exc:
            report('invalid_module', filename, None, str(exc))
            return ast.Module(body=[], type_ignores=[])

    filename = spec['factory_file']
    module = tree(filename)
    entry = module if spec['entry_file'] == filename else tree(spec['entry_file'])
    factory_module = str(PurePosixPath(filename).with_suffix('')).replace('/', '.')
    names = {spec['factory_name']} if spec['entry_file'] == filename else set()
    for n in ast.walk(entry):
        if isinstance(n, ast.ImportFrom) and n.module in {factory_module, PurePosixPath(filename).stem}:
            names.update(a.asname or a.name for a in n.names if a.name == spec['factory_name'])
    scope = entry if not spec.get('entry_function') else next((n for n in entry.body if isinstance(n, ast.FunctionDef) and n.name == spec['entry_function']), None)
    def factory_call(n):
        return isinstance(n, ast.Call) and ((isinstance(n.func, ast.Name) and n.func.id in names)
            or (isinstance(n.func, ast.Attribute) and n.func.attr == spec['factory_name'] and ast.unparse(n.func.value) == factory_module))
    adapter = spec.get('interface_mode') == 'component_adapter'
    if not adapter and (scope is None or not any(factory_call(n) for n in ast.walk(scope))):
        report('entry_link', spec['entry_file'], scope, 'Existing entry point must call the same components factory used by framework checks')
    if scope:
        for n in ast.walk(scope):
            if isinstance(n, ast.Assign) and factory_call(n.value) and any(isinstance(t, (ast.Tuple, ast.List)) for t in n.targets):
                report('entry_unpack', spec['entry_file'], n, 'Entry point must read components dictionary values by key; tuple-unpacking the dictionary yields key strings')
    factory = next((n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == spec['factory_name']), None)
    if factory is None:
        report('missing_factory', filename, None, 'Define the declared import-safe components factory')
        return issues
    parameters = {a.arg for a in factory.args.args + factory.args.kwonlyargs}
    required = set(spec.get('argument_map', {}).values()) if adapter else {'dataset_name', 'data_root', 'download', 'device'}
    if factory.args.kwarg is None and not required.issubset(parameters):
        report('factory_signature', filename, factory, 'Factory must accept dataset_name, data_root, download and device as keyword arguments')
    if factory.args.posonlyargs:
        report('factory_signature', filename, factory, 'Controlled factory arguments cannot be positional-only')
    functions = {n.name: n for n in module.body if isinstance(n, ast.FunctionDef)}
    reachable, pending = {}, [factory]
    while pending:
        fn = pending.pop()
        if fn.name in reachable:
            continue
        reachable[fn.name] = fn
        for n in ast.walk(fn):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in functions:
                pending.append(functions[n.func.id])
    assignments = {n.targets[0].id: n.value for n in ast.walk(factory)
                   if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)}
    def deref(node, seen=()):
        if isinstance(node, ast.Name) and node.id in assignments and node.id not in seen:
            return deref(assignments[node.id], seen + (node.id,))
        return node
    returns = [n for n in ast.walk(factory) if isinstance(n, ast.Return)]
    dictionary_return = any(isinstance(deref(n.value), ast.Dict) or
        (isinstance(deref(n.value), ast.Call) and isinstance(deref(n.value).func, ast.Name)
         and deref(n.value).func.id == 'dict') for n in returns)
    if not dictionary_return:
        issues[:] = [i for i in issues if i['code'] != 'entry_unpack']
    for n in returns:
        value = deref(n.value)
        if isinstance(value, ast.Constant) or (isinstance(value, (ast.Tuple, ast.List)) and len(value.elts) != 3):
            report('factory_return', filename, n, 'Return a components dictionary or a three-item (train_dataset, test_dataset, model) tuple/list')
        keys = None
        if isinstance(value, ast.Dict):
            keys = {k.value for k in value.keys if isinstance(k, ast.Constant)}
        elif isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == 'dict':
            keys = {k.arg for k in value.keywords}
        if keys is not None and not {'train_dataset', 'test_dataset', 'model'}.issubset(keys):
            report('factory_return', filename, n, 'Factory dictionary is missing train_dataset, test_dataset or model')
    if not returns:
        report('factory_return', filename, factory, 'Factory has no return statement')

    # Importing the factory must not consume the agent runner's command line.
    for top in module.body:
        if isinstance(top, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(top, ast.If) and '__name__' in ast.unparse(top.test):
            continue
        for n in ast.walk(top):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in {'parse_args', 'parse_known_args'}:
                report('import_cli', filename, n, 'Move command-line parsing behind the __main__ guard; the factory must be import-safe')

    def path_value(node, seen=(), environment=None, local_assignments=None):
        environment = {'data_root': spec['data_root']} if environment is None else environment
        local_assignments = assignments if local_assignments is None else local_assignments
        if isinstance(node, ast.Name):
            if node.id in local_assignments and node.id not in seen:
                return path_value(local_assignments[node.id], seen + (node.id,), environment, local_assignments)
            return environment.get(node.id)
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value.replace('\\', '/').removeprefix('./')
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func)
            if name in {'os.path.join', 'posixpath.join'}:
                parts = [path_value(a, seen, environment, local_assignments) for a in node.args]
                if parts and all(p is not None for p in parts):
                    return '/'.join(p.strip('/') for p in parts)
            if isinstance(node.func, ast.Attribute) and node.func.attr == 'get' and node.args and isinstance(node.args[0], ast.Constant):
                key = node.args[0].value
                if key in spec.get('factory_kwargs', {}):
                    v = spec['factory_kwargs'][key]
                    return v if isinstance(v, str) else None
                if len(node.args) > 1:
                    return path_value(node.args[1], seen, environment, local_assignments)
        return None
    copied = [str(PurePosixPath(p)) for p in spec['data_files']]
    numeric = any(c['name'].lower().replace('-', '').replace('_', '') in {'svhn', 'mnist'} for c in spec['cases'])
    # Propagate each helper call's actual path arguments. A helper parameter
    # named data_root is not necessarily the factory's dataset root.
    contexts, seen_contexts = [(factory, {'data_root': spec['data_root']})], set()
    for fn, environment in contexts:
        signature = (fn.name, tuple(sorted(environment.items())))
        if signature in seen_contexts:
            continue
        seen_contexts.add(signature)
        if len(seen_contexts) > 64:
            report('unresolved_input', filename, fn, 'Too many helper path contexts for bounded static analysis')
            break
        local_assignments = {n.targets[0].id: n.value for n in ast.walk(fn)
                             if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)}
        parents = {child: parent for parent in ast.walk(fn) for child in ast.iter_child_nodes(parent)}
        locals_ = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
        for n in ast.walk(fn):
            if not adapter and isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id == 'args' and 'args' not in locals_:
                report('global_cli', filename, n, 'Factory/setup depends on global CLI args; pass explicit configuration with defaults instead')
            if not isinstance(n, ast.Call):
                continue
            name = ast.unparse(n.func)
            if isinstance(n.func, ast.Name) and n.func.id in reachable:
                child = reachable[n.func.id]
                bound = {}
                params = child.args.args
                for parameter, arg in zip(params, n.args):
                    bound[parameter.arg] = path_value(arg, environment=environment, local_assignments=local_assignments)
                for keyword in n.keywords:
                    if keyword.arg:
                        bound[keyword.arg] = path_value(keyword.value, environment=environment, local_assignments=local_assignments)
                for parameter, default in zip(params[len(params) - len(child.args.defaults):], child.args.defaults):
                    if parameter.arg not in bound:
                        bound[parameter.arg] = path_value(default, environment={}, local_assignments={})
                contexts.append((child, bound))
            if numeric and isinstance(n.func, ast.Attribute) and n.func.attr == 'RandomHorizontalFlip':
                parent, conditional = parents.get(n), False
                while parent is not None:
                    if isinstance(parent, (ast.If, ast.IfExp)) and 'dataset_name' in ast.unparse(parent.test):
                        conditional = True
                    parent = parents.get(parent)
                if not conditional:
                    report('digit_transform', filename, n, 'Unconditional horizontal flip is applied in setup for digit datasets; use a dataset-specific transform branch')
            if name in {'Image.new', 'PIL.Image.new', 'torch.randn', 'torch.rand', 'torch.zeros', 'torch.ones', 'np.zeros', 'np.ones', 'numpy.zeros', 'numpy.ones'}:
                report('synthetic_input', filename, n, 'Synthetic input construction in dataset setup is not permitted; missing real data must stop testing: ' + name)
            if isinstance(n.func, ast.Attribute) and n.func.attr == 'ImageFolder':
                arg = n.args[0] if n.args else next((k.value for k in n.keywords if k.arg == 'root'), None)
                path = path_value(arg, environment=environment, local_assignments=local_assignments)
                details = dict(expression=ast.unparse(n), resolved_path=path, copied_paths=copied,
                               helper=fn.name, bound_path_arguments=environment)
                if path is None:
                    report('unresolved_input', filename, n, 'ImageFolder input path cannot be verified statically; use an explicit data_root-relative path or factory_kwargs value', **details)
                elif not any(path == p or path.startswith(p.rstrip('/') + '/') for p in copied):
                    report('uncopied_input', filename, n,
                           'ImageFolder reads ' + path + ', which is not covered by contract data_files. '
                           'Check the actual image-directory expression and factory_kwargs; point the loader at the real auxiliary image folder and include that same path. Do not switch loss/views or copy unrelated parent data just to hide this error.', **details)

    # Wrappers may return multiple augmented views. Detect the actual return
    # structure rather than relying on repo-specific variable or class names.
    for binding in spec.get('source_bindings', []):
        source = tree(binding['candidate_file'])
        cls = next((n for n in source.body if isinstance(n, ast.ClassDef) and n.name == binding['candidate_symbol']), None)
        if cls is None:
            continue  # Exact source binding validation also runs before data IO.
        getter = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__getitem__'), None)
        if getter is None:
            continue
        values = {n.targets[0].id: n.value for n in ast.walk(getter) if isinstance(n, ast.Assign)
                  and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)}
        multi = False
        for n in ast.walk(getter):
            if isinstance(n, ast.Return) and isinstance(n.value, (ast.Tuple, ast.List)) and n.value.elts:
                x = n.value.elts[0]
                if isinstance(x, ast.Name):
                    x = values.get(x.id, x)
                multi |= isinstance(x, (ast.Tuple, ast.List)) and len(x.elts) > 1
        if multi:
            if not all(c['internal_augmentation'] for c in spec['cases']):
                report('augmentation_coverage', binding['candidate_file'], getter, 'Bound wrapper returns augmented views; enable internal_augmentation coverage for all tested cases')
            if any(c['views'] == 1 or c['loss'] != 'jsd' for c in spec['cases']):
                report('view_loss_mismatch', binding['candidate_file'], getter, 'Bound wrapper has multi-view returns but cases declare single-view/CE. Supply a matching multi-view contract; conditional single-view wrappers need explicit interface adaptation before this static check can certify compatibility')
    return issues


def validate_candidate(files, spec):
    issues = inspect_candidate(files, spec)
    # Constructing a transform does not prove that a target branch uses it.
    # The actual dataset transform and augmentation pool are checked at runtime.
    advisory = {'unresolved_input', 'augmentation_coverage', 'view_loss_mismatch', 'digit_transform'}
    errors = [i for i in issues if i['code'] not in advisory]
    warnings = [dict(i, severity='requires_runtime_verification') for i in issues if i['code'] in advisory]
    if errors:
        raise CandidateError(errors)
    return {'status': 'passed', 'warnings': warnings,
            'note': 'Static checks found no definite interface error. Warnings require runtime verification; no runtime success is implied.'}
