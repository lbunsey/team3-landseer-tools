from .clone_repo import clone_repo
from .list_datasets import list_datasets
from .read_repo import READ_TOOLS
from .list_models import list_models
from .inspect_repo import inspect_repo
from .add_model import add_model

TOOLS = [
    clone_repo,
    list_datasets,
    list_models,
    *READ_TOOLS,
    inspect_repo,
    add_model,
]
