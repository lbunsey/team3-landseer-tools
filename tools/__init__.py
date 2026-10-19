from .clone_repo import clone_repo
from .list_datasets import list_datasets
from .inspect_repo import inspect_repo
from .add_model import add_model

TOOLS = [
    clone_repo,
    list_datasets,
    inspect_repo,
    add_model,
]