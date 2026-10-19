from .clone_repo import clone_repo
from .list_datasets import list_datasets
from .read_repo import READ_TOOLS
from .list_models import list_models

TOOLS = [
    clone_repo,
    list_datasets,
    list_models,
    *READ_TOOLS,
]