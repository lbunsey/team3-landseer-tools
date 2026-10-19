
"""inspect_repo: inspect a cloned research repository without modifying it."""

import json
import os
from pathlib import Path

from langchain_core.tools import tool
from . import patch_utils


REPOS_DIR = Path(__file__).resolve().parent.parent / "repos"

SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    "node_modules",
    "site-packages",
    "build",
    "dist",
}

SKIP_FILES = {
    ".env",
    ".env.local",
    ".gitignore",
    ".env.example",
    "id_rsa",
    "id_ed25519",
}

TEXT_EXTENSIONS = {
    ".py",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".ini",
    ".cfg",
    ".md",
    ".txt",
}

MAX_SCANNED_FILES = 2000
MAX_CANDIDATES = 12
MAX_FILE_CHARS = 6000
MAX_TOTAL_CHARS = 20000


def get_repo_root(repo_name: str) -> Path:
    """Resolve a repository name to a direct child of repos/."""

    name = repo_name.strip()

    if (
        not name
        or Path(name).name != name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
    ):
        raise ValueError(
            "repo_name must be a folder name inside repos/, not a path."
        )

    root = REPOS_DIR / name

    if root.is_symlink() or not root.is_dir():
        raise FileNotFoundError(
            f"Repository '{name}' was not found in repos/."
        )

    # Ensure the resolved directory is still a direct child of repos/.
    resolved_root = root.resolve()
    resolved_repos = REPOS_DIR.resolve()

    if resolved_root.parent != resolved_repos:
        raise ValueError("Repository path is outside repos/.")

    return resolved_root


def is_safe_text_file(path: Path) -> bool:
    """Allow relevant text files while excluding likely secrets."""

    if path.is_symlink() or not path.is_file():
        return False

    if path.name.lower() in SKIP_FILES:
        return False

    if path.suffix.lower() not in TEXT_EXTENSIONS:
        return False

    try:
        return path.stat().st_size <= 500_000
    except OSError:
        return False


def candidate_score(rel_path: str) -> int:
    """Prioritize files likely to explain dataset loading and execution."""

    path = Path(rel_path)
    name = path.name.lower()
    score = 0

    if name.startswith("readme"):
        score += 100

    if name in {
        "requirements.txt",
        "pyproject.toml",
        "setup.py",
        "environment.yml",
    }:
        score += 40

    keywords = {
        "dataset": 50,
        "dataloader": 50,
        "data_loader": 50,
        "data": 35,
        "train": 30,
        "test": 30,
        "eval": 25,
        "config": 20,
        "main": 15,
    }

    for keyword, points in keywords.items():
        if keyword in name:
            score += points

    if path.suffix.lower() == ".py":
        score += 10

    return score


def read_text_file(path: Path) -> str:
    """Read a text file and attach line numbers for easier analysis."""

    content = path.read_text(encoding="utf-8", errors="replace")
    lines = content.splitlines()

    numbered = "\n".join(
        f"{i}: {line}" for i, line in enumerate(lines, start=1)
    )

    if len(numbered) > MAX_FILE_CHARS:
        numbered = numbered[:MAX_FILE_CHARS]
        numbered += "\n... [TRUNCATED: inspect this file separately]"

    return numbered


@tool
def inspect_repo(repo_name: str, file_path: str = "") -> str:
    """Inspect a cloned research repository without modifying it.

    Args:
        repo_name: Folder name inside the local repos/ directory.
        file_path: Optional path relative to the repository root.
            If omitted, return the file tree and relevant file excerpts.
            If provided, return JSON with complete UTF-8 content, exact-byte
            SHA-256 original_hash and complete=true (up to 500,000 bytes).
            Use this mode before patching; overview excerpts are insufficient.

    Returns:
        A repository overview or the requested file contents.
        This tool is read-only and does not execute repository code.
    """
    
    print(
        f"[DEBUG][inspect_repo] inspect_repo started: {repo_name}",
        flush=True,
    )

    try:
        print(
            f"[DEBUG][inspect_repo] Resolving repository: {repo_name}",
            flush=True,
        )

        # Single-file reads share patch validation and preserve exact bytes.
        if file_path.strip():
            target = patch_utils.resolve_target(repo_name, file_path)
            with target.open("rb") as stream:
                raw = stream.read(patch_utils.MAX_FILE_BYTES + 1)
            if len(raw) > patch_utils.MAX_FILE_BYTES:
                raise ValueError("File exceeds the patch size limit")
            content = raw.decode("utf-8")
            if "\x00" in content:
                raise ValueError("NUL-containing files are not supported text files")
            print("[DEBUG][inspect_repo] Complete file read finished.", flush=True)
            return json.dumps({
                "ok": True, "repo_name": repo_name,
                "file_path": file_path.replace("\\", "/"),
                "encoding": "utf-8", "size_bytes": len(raw), "complete": True,
                "original_hash": patch_utils.compute_file_hash(raw), "content": content,
            }, ensure_ascii=False)

        root = get_repo_root(repo_name)

        print(
            f"[DEBUG][inspect_repo] Repository resolved: {root}",
            flush=True,
        )

        # Mode 2: scan the repository and prioritize relevant files.
        print(
            f"[DEBUG][inspect_repo] Scanning repository: {root}",
            flush=True,
        )

        all_files = []
        candidates = []
        scanned = 0

        for folder, subfolders, filenames in os.walk(root):
            subfolders[:] = sorted(
                d for d in subfolders
                if d not in SKIP_DIRS and not d.startswith(".")
            )

            for filename in sorted(filenames):
                path = Path(folder) / filename

                if path.is_symlink():
                    continue

                scanned += 1

                if scanned > MAX_SCANNED_FILES:
                    break

                rel = path.relative_to(root).as_posix()
                all_files.append(rel)

                if (
                    is_safe_text_file(path)
                    and candidate_score(rel) > 0
                ):
                    candidates.append(
                        (candidate_score(rel), rel, path)
                    )

            if scanned > MAX_SCANNED_FILES:
                break

        candidates.sort(key=lambda item: (-item[0], item[1]))

        print(
            f"[DEBUG][inspect_repo] Scan completed: {len(all_files)} files,",
            flush=True,
        )

        lines = [
            f"Repository: {repo_name}",
            f"Files encountered: {len(all_files)}",
            "",
            "FILE TREE (first 150 files):",
        ]

        lines.extend(f"  {name}" for name in all_files[:150])

        if scanned > MAX_SCANNED_FILES:
            lines.append(
                f"... scan stopped at {MAX_SCANNED_FILES} files."
            )

        lines.extend([
            "",
            "RELEVANT FILE EXCERPTS:",
        ])

        remaining = MAX_TOTAL_CHARS

        for _, rel, path in candidates[:MAX_CANDIDATES]:
            if remaining <= 0:
                break

            try:
                content = read_text_file(path)
            except (OSError, UnicodeError) as exc:
                print(
                    f"[ERROR][inspect_repo] Failed to read {rel}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                continue

            section = f"\n--- {rel} ---\n{content}\n"
            section = section[:remaining]

            lines.append(section)
            remaining -= len(section)

        if not candidates:
            lines.append(
                "No relevant text files were recognized. "
                "Inspect the repository structure manually."
            )

        lines.extend([
            "",
            "IMPORTANT:",
            "- This tool only reads files; it does not execute code.",
            "- Excerpts may be truncated.",
            "- Use file_path to inspect a specific file in more detail.",
        ])

        result = "\n".join(lines)

        print(
            f"[DEBUG][inspect_repo] inspect_repo finished: "
            f"{len(all_files)} files scanned",
            flush=True,
        )

        return result

    except (ValueError, FileNotFoundError) as exc:
        print(
            f"[ERROR][inspect_repo] inspect_repo failed: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        return f"Error: {exc}"

    except OSError as exc:
        print(
            f"[ERROR][inspect_repo] inspect_repo failed: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        return f"Error inspecting repository: {exc}"



if __name__ == "__main__":
    import sys

    if len(sys.argv) not in (2, 3):
        sys.exit(
            "Usage: python -m tools.inspect_repo <repo_name> [file_path]"
        )

    repo = sys.argv[1]
    requested_file = sys.argv[2] if len(sys.argv) == 3 else ""

    print(
        inspect_repo.invoke({
            "repo_name": repo,
            "file_path": requested_file,
        })
    )
