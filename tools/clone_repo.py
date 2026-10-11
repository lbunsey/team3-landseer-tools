import os
import subprocess
import sys
from pathlib import Path

from langchain_core.tools import tool

REPOS_DIR = Path(__file__).resolve().parent.parent / "repos"


@tool
def clone_repo(url: str) -> str:
    """Clones a public gitHub repository into a local 'repos' folder.

    Input: the repository URL ( https://github.com/owner/repo )
    Returns: the folder name the repo was cloned into or error message.
    only https://github.com URLs are accepted
    """
    url = url.strip()
    if not url.startswith("https://github.com/"):
        return "Error: URL must start with https://github.com/"

    name = url.rstrip("/").split("/")[-1].removesuffix(".git")
    dest = REPOS_DIR / name
    if dest.exists():
        return f"Repo already cloned. Folder name: {name}"

    REPOS_DIR.mkdir(exist_ok=True)
    try:
        result = subprocess.run(
            ["git", "clone", "--depth", "1", "--", url, str(dest)],
            capture_output=True, text=True, timeout=120,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
    except FileNotFoundError:
        return "Error: git is not installed."
    except subprocess.TimeoutExpired:
        return "Error: clone timed out after 120 seconds."

    if result.returncode != 0:
        return f"Error: git clone failed: {result.stderr.strip()}"
    return f"Cloned successfully. Folder name: {name}"


if __name__ == "__main__":
    print(clone_repo.invoke({"url": sys.argv[1]}))