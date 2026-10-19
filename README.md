# Landseer Agent Tools

A provider-configurable LangChain / Deep Agents CLI for inspecting research repositories and proposing single-file edits with explicit human approval.

## Setup

Developed with Python 3.11.

Run from the project root:

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

**Linux/macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Keep your configuration if `.env` already exists.

Select `LLM_PROVIDER` and the corresponding variables documented in `.env.example`:

| Value | Provider |
| --- | --- |
| `rcac` | OpenAI-compatible endpoint |
| `openai` | OpenAI |
| `claude` | Anthropic |
| `google` | Google |

Do not commit `.env` or API keys. Running the agent uses your configured model service and may incur API charges.

## Run the agent

```bash
python agent.py
```

Use `exit` or `quit` to leave. Keep a modification task within the same running CLI conversation: patch records, approvals and checkpoints are in memory and are lost on process exit. Saved file edits remain on disk.

Repositories are direct children of the project's `repos/` directory. Ask the agent to clone or inspect a repository, identify datasets, or propose a code change.

For code changes, the agent saves a proposal with a unique Patch ID. You can request revisions or restore an earlier proposal. Before applying a patch, the CLI displays the complete saved diff and asks for Y/N confirmation.

| Confirmation input | Result |
| --- | --- |
| `Y` / `y` | Approve the displayed ID, resume, recheck the hash and attempt the write |
| `N` / `n` | Skip this tool call and retain the proposal |
| Blank or other text, including `yes` | Ask again; no new approval |
| EOF / Ctrl+C at confirmation | Cancel this confirmation; no new approval |

Whitespace around Y/N is accepted. Chat text saying "yes" is not approval. Every apply is interrupted, including retries after failed writes. Cancelling does not revoke old approval, but skips this call. Multiple requested applies are confirmed individually, not as an all-or-nothing multi-file transaction.

## Tools and architecture

```text
langchain_agent/
├── agent.py                   # CLI, sessions and patch confirmation
├── model_factory.py           # Provider selection
├── tools/
│   ├── __init__.py            # Agent tool registry
│   ├── clone_repo.py
│   ├── list_datasets.py
│   ├── inspect_repo.py
│   ├── patch_store.py
│   ├── patch_utils.py
│   ├── patch_preview.py
│   └── patch_apply.py
├── repos/                     # Local repositories, ignored by Git
├── .env.example
├── requirements.txt
└── README.md
```

| Registered tool | Responsibility |
| --- | --- |
| `clone_repo(url)` | Shallow-clone a GitHub repository into `repos/` |
| `list_datasets(repo_name)` | Static dataset discovery with evidence |
| `inspect_repo(repo_name, file_path="")` | Overview or full single-file read |
| `preview_patch(...)` | Validate a full proposed file, create diff and save version |
| `restore_patch_proposal(source_patch_id)` | Restore old contents under a new pending-review ID |
| `apply_patch(patch_id)` | Apply saved bytes after approval and validation |

| Patch module | Main functions |
| --- | --- |
| `patch_store.py` | `create_patch_record`, `get_patch`, `list_session_patches`, lifecycle operations |
| `patch_utils.py` | `resolve_target`, `compute_file_hash` |
| `patch_preview.py` | `prepare_patch`, `preview_patch`, internal `restore_patch` and tool wrapper |
| `patch_apply.py` | `atomic_write`, `apply_saved_patch`, `apply_patch` |

Helpers and `approve_patch` are not agent tools. The CLI records direct human approval. The synchronous agent binds patches to its thread rather than trusting the model's session argument. Built-in virtual-file tools, shell execution and subagent dispatch are filtered from model requests and rejected at the tool boundary. The backend stays `StateBackend`, not a local disk backend. New agent tools must be deliberately added to `tools/__init__.py`.

### Complete reads and previews

Overview excerpts are capped at 6,000 characters per file; they are not complete edit bases. With `file_path`, inspection returns a JSON string containing `content`, `original_hash`, `size_bytes`, `encoding` and `complete=true`. Single-file content has no added line numbers and is not truncated. Invalid reads return an error string.

Reads use strict UTF-8. Exact bytes, including BOM and CRLF/LF, determine SHA-256. Preview accepts the FULL proposed file and expected base hash without newline normalization. No-op, stale, unsafe or oversized proposals are rejected. Previews never write repository files.

Large tool-output offloading is disabled so full reads are not replaced with head/tail summaries. Model context/output limits still apply. Full content does not prove the model understood it or the modification is correct; review the complete diff before approving.

### Patch revisions

Each successful preview creates an ID and immutable content snapshot. A revision supersedes its active parent without overwriting contents. Restoring a version creates a new child of that source with `pending_review` status; approval is not inherited and other branches are not automatically superseded. Parents describe ancestry, not required application order.

```text
pending_review -> approved -> applied
pending_review / approved -> superseded (when a child is created)
pending_review / approved -> rejected
```

Superseded, rejected and applied records cannot be reapproved or applied. Old contents can be restored as a new proposal if their original base is current. A changed base requires inspection and a new proposal.

### Applying and errors

Application checks approval and hash, stages saved bytes in the same directory, flushes them, rechecks the path/hash and replaces the target. Only then does it mark the record applied. Pre-replacement failures do not write the target or mark success; external edits are retained. Normal exception handling cleans temporary staging files.

If writing succeeds but status update fails, `ok=false, file_written=true` means the file changed but bookkeeping did not finish. Inspect the file/record before retrying. Normal success is `ok=true, status=applied`.

## Current scope and limits

- One existing UTF-8 file per patch; original/proposed content each limited to 500,000 bytes. No creation/deletion or multi-file transaction.
- Extensions: `.py`, `.yaml`, `.yml`, `.json`, `.toml`, `.ini`, `.cfg`, `.md`, `.txt`. Hidden paths, private-key names, links/junctions, unsafe Windows paths and targets outside the repository are rejected.
- Records/checkpoints are volatile. No persistence, crash recovery or cross-process transaction is implemented or required for this prototype.
- Hash rechecks detect ordinary edits but cannot eliminate the small race before replacement. Basic mode bits are preserved; arbitrary ACLs/metadata may not be.
- Full replacement can include unintended deletions. Hash/UTF-8 validation does not establish semantic correctness or valid Python syntax.
- Invocation is synchronous CLI. Programmatic callers must implement trusted confirmation/interrupt handling; helpers are not an authentication boundary.
- Automatic dataset/model integration (`add_dataset`, etc.) is not implemented.
