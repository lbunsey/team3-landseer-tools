"""Shared, read-only path validation and hashing for patch workflows.

Validation is a point-in-time check, not a lock against concurrent path changes.
Preview must validate UTF-8 content; apply must revalidate paths and file hashes.
"""

import hashlib
from pathlib import Path, PureWindowsPath
import stat


REPOS_DIR = Path(__file__).resolve().parent.parent / "repos"
TEXT_EXTENSIONS = {
    ".py", ".yaml", ".yml", ".json", ".toml", ".ini", ".cfg", ".md", ".txt",
}
MAX_FILE_BYTES = 500_000


def _path_parts(value: str) -> tuple[str, ...]:
    """Accept portable relative paths, rejecting Windows aliases and streams."""
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("Path must be a non-empty string without NUL characters")
    windows_path = PureWindowsPath(value)
    if windows_path.drive or windows_path.root or Path(value).is_absolute():
        raise ValueError("Absolute, rooted and drive-relative paths are not allowed")
    parts = tuple(value.replace("\\", "/").split("/"))
    for part in parts:
        if (
            not part or part in {".", ".."} or part.endswith((".", " "))
            or any(c in part for c in '<>:"|?*')
            or any(ord(c) < 32 for c in part)
            or PureWindowsPath(part).is_reserved()
        ):
            raise ValueError("Invalid path component or path traversal")
    return parts


def _reject_link(path: Path) -> None:
    info = path.lstat()
    # Windows junctions/reparse points are not necessarily reported as symlinks.
    if stat.S_ISLNK(info.st_mode) or (
        getattr(info, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    ):
        raise ValueError("Symlinks, junctions and reparse points are not permitted")


def resolve_target(repo_name: str, file_path: str) -> Path:
    """Return an existing permitted file inside a direct child of repos/.

Raises ValueError for unsafe/unsupported targets and FileNotFoundError for
missing paths. This does not read file contents or guarantee UTF-8 encoding.
Hidden paths, private-key names, links and files over 500 KB are excluded.
"""
    repo_parts = _path_parts(repo_name)
    if len(repo_parts) != 1 or repo_parts[0].startswith("."):
        raise ValueError("repo_name must be a single non-hidden folder name")
    parts = _path_parts(file_path)
    if any(part.startswith(".") for part in parts):
        raise ValueError("Hidden paths and repository metadata are not permitted")
    if parts[-1].lower().split(".")[0] in {"id_rsa", "id_ed25519", "id_dsa", "id_ecdsa"}:
        raise ValueError("Private-key files are not permitted")

    repos = REPOS_DIR.resolve(strict=True)
    root = repos / repo_parts[0]
    _reject_link(root)
    root = root.resolve(strict=True)
    if root.parent != repos or not root.is_dir():
        raise ValueError("Repository must be a directory directly inside repos/")

    target = root
    for part in parts:
        target = target / part
        _reject_link(target)
    target = target.resolve(strict=True)
    if not target.is_relative_to(root):
        raise ValueError("Target resolves outside the repository")
    if not target.is_file():
        raise ValueError("Target must be an existing regular file")
    if target.suffix.lower() not in TEXT_EXTENSIONS:
        raise ValueError("Unsupported text-file extension")
    if target.stat().st_size > MAX_FILE_BYTES:
        raise ValueError(f"Target exceeds {MAX_FILE_BYTES} bytes")
    return target


def compute_file_hash(content: bytes) -> str:
    """SHA-256 of exact bytes, including BOM and line endings; no normalization."""
    if not isinstance(content, bytes):
        raise TypeError("content must be bytes")
    return hashlib.sha256(content).hexdigest()
