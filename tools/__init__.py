from .clone_repo import clone_repo
from .list_datasets import list_datasets
from .inspect_repo import inspect_repo

TOOLS = [
    clone_repo,
    list_datasets,
    inspect_repo,
]