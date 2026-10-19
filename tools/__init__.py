from .clone_repo import clone_repo
from .list_datasets import list_datasets
from .inspect_repo import inspect_repo
from .patch_preview import preview_patch, restore_patch_proposal
from .patch_apply import apply_patch

TOOLS = [
    clone_repo,
    list_datasets,
    inspect_repo,
    preview_patch,
    restore_patch_proposal,
    apply_patch,
]
