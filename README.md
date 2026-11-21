# Landseer Agent Tools

A provider-configurable LangChain / Deep Agents CLI for inspecting research repositories and preparing dataset and model integrations through file modifications and additions with explicit human approval.

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

Repositories are direct children of the project's `repos/` directory. Ask the agent to clone or inspect a repository, identify supported datasets and models, or propose a code change.

For code changes, the agent saves a proposal with a unique Patch ID. You can request revisions or restore an earlier proposal. Before applying a patch, the CLI displays the complete saved diff and asks for Y/N confirmation.

To add a dataset, describe the repository, dataset and intended experiment in plain language. The agent first examines repository execution evidence and traces the relevant data and training flow. Dataset tools manage the analysis, task, CPU validation and application status. All code proposals use the shared `preview_patch`; the CLI associates them with the matching dataset task automatically. CPU checks run in a temporary copy after execution consent, using real local data without downloads or installations. Missing inputs are reported as blocked. These bounded checks do not establish full training or experiment correctness. You review and approve each file separately. Task histories are lost when the process exits.

To add a model, describe the repository and model name. `add_model` inspects model-selection code, reuses an existing definition or a supported template, or accepts a full model module supplied by the agent. It prepares model and registration changes and runs a scratch-copy smoke test before saving proposals. Review the returned results and approve each file through the same CLI confirmation flow. A smoke test does not establish successful training or experiment correctness.

New files use the same patch workflow. `preview_patch` accepts `operation="create"` with `expected_hash=""`; the target must not exist and its parent directory must already exist. The file is only created after approval, and a file created in the meantime is never overwritten.

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
│   ├── add_dataset.py
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
| `list_models(repo_name)` | Static model discovery with evidence |
| `list_repo_files(repo, max_depth=3)` | List repository files |
| `list_attacks(repo_name)` | Static attack discovery (backdoor or adversarial) with evidence |
| `list_arguments(repo_name, script="")` | List a repository's argparse arguments with defaults and choices |
| `list_patches(session_id, repo_name="", status="", limit=30)` | List this session's patch proposals and their status |
| `revert_patch(patch_id)` | Propose undoing an applied patch as a new pending-review Patch ID |
| `read_repo_file(repo, path, start_line=1, end_line=300)` | Read a numbered range of repository text |
| `search_repo(repo, pattern)` | Search repository text |
| `inspect_repo(repo_name, file_path="")` | Overview or full single-file read |
| `preview_patch(...)` | Shared dataset/model/general proposal entry: accept full contents or original line-range edits, check paths/hash/Python syntax, and save a version with its complete diff |
| `restore_patch_proposal(source_patch_id)` | Restore old contents under a new pending-review ID |
| `apply_patch(patch_id)` | Apply saved bytes after approval and validation |
| `add_dataset(...)` | Start/query dataset tasks, discover workflow evidence and read numbered original lines with pagination |
| `record_dataset_analysis(...)` | Save the integration analysis and unresolved research questions |
| `configure_dataset_validation(...)` | Configure existing component interfaces and real local inputs |
| `validate_dataset(...)` | Run bounded CPU data-flow and regression checks on the latest proposed batch |
| `add_model(...)` | Inspect model-selection code, prepare model and registration proposals, and run a scratch-copy smoke test |

| Patch module | Main functions |
| --- | --- |
| `list_patches.py` | `list_patches` |
| `revert_patch.py` | `revert_patch_proposal`, `revert_patch` |
| `patch_store.py` | `create_patch_record`, `get_patch`, `list_session_patches`, lifecycle operations |
| `patch_utils.py` | `resolve_target`, `compute_file_hash` |
| `patch_preview.py` | `prepare_patch`, `preview_patch`, internal `restore_patch` and tool wrapper |
| `patch_apply.py` | `atomic_write`, `apply_saved_patch`, `apply_patch` |

Helpers and `approve_patch` are not agent tools. The CLI records direct human approval. The synchronous agent binds patches to its thread rather than trusting the model's session argument. Built-in virtual-file tools, direct shell tools and subagent dispatch are filtered from model requests and rejected at the tool boundary. `add_model` can execute its internal scratch-copy smoke test; this restriction does not disable that check. The backend stays `StateBackend`, not a local disk backend. New agent tools must be deliberately added to `tools/__init__.py`.

### Model integration workflow

`add_model` discovers model-selection code, prepares model definitions and registration changes, and returns saved Patch IDs. The agent runtime supplies the current conversation session. In the main agent CLI, request application of the proposals and confirm each saved patch with Y/N. No separate post-response patch review runs.

For standalone model integration, run from the project root:

```bash
python -m tools.add_model <repo_name> <model_name> [model_code.py]
```

This entry point creates its own session and uses `patch_review.py` to display all pending changes in that session and ask once before applying them sequentially. It shares the same patch storage and file-writing core as the main agent. Rejection leaves repository files unchanged; application stops on failure and reports partial results. Multi-file application is not atomic.

Model smoke tests run in a scratch copy and depend on the repository's dependencies. Read the reported smoke-test result separately from patch application status. Dataset downloads, training and scientific validation remain user-run.

### Dataset integration workflow

Dataset analysis, proposal storage and CPU validation are separate steps. The agent uses one shared proposal tool; there is no separate dataset preview tool in the registry.

The minimum deliverable across repositories is a reviewable, versioned dataset integration proposal checked for Python syntax and file integrity. The agent traces the actual repository workflow and saves minimal changes first. CPU validation is optional and is attempted only when requested and supported by existing interfaces. An unsupported runner interface is reported as execution not verified; it does not require refactoring into a factory or block proposal storage. Syntax checks do not establish that training or evaluation works. User-requested application shows the saved diff and CPU verification status, then requires direct CLI Y/N confirmation. A passing CPU test is not an application prerequisite; approval, version/hash, path and no-overwrite checks remain enforced.

A complete file read covers line-range proposals for the same file/hash; a second numbered read is not required. Partial reads may cover a larger interval than the edit, or contiguous pages, provided every modified line is covered at the same hash. Unseen lines, stale contents, invalid ranges and Python syntax errors remain rejected.

Range-edit `new_lines` accepts individual source lines or multiline source blocks. Real LF/CRLF separators are accepted and the original file's line-ending style is retained; literal backslash escapes inside code are never decoded. Syntax errors remain errors, rather than being repaired by string unescaping.

The dataset service also stores an advisory static review for each Python proposal: removed torchvision loader calls, conflicting constant class counts within one block, unsupported constructor keywords where installed torchvision source provides evidence, and existing string newline/tab/carriage-return values changed to literal backslash text in the same scope/call context. Escape findings include original/candidate values and source locations; the tool never globally unescapes or rewrites source. Findings are returned with the saved proposal and status and feed bounded agent repair; they neither rewrite code nor block user-approved application. This is separate from shared patch syntax/storage checks and does not change `add_model`. No findings does not certify augmentation semantics, evaluation compatibility or runtime correctness.

The CLI prints each new current dataset proposal's exact stored diff separately from model prose and reports outstanding static findings. Review this stored diff, rather than treating a model-written excerpt as the applicable contents.

Static-review repair feedback includes bounded numbered excerpts of the actual original branch and candidate context, plus related local evaluation calls. Original and candidate coordinates are labeled separately; truncated excerpts are explicitly incomplete. The agent must revise against the original file and inspect evaluation guards/logging rather than assuming empty paths disable evaluation. This supplies source evidence, not an automatic code repair or execution certificate.

For active dataset tasks, complete Python-file reads also provide original loader blocks and available installed target-constructor keywords before editing. The source content and its byte hash are unchanged. This is early evidence for the model, not a new public tool or automatic code generation.

If a dataset proposal has invalid tool arguments or Python syntax and the model stops with prose, the runtime gives it at most two additional repair attempts in that turn. Syntax failures include bounded, numbered candidate/original excerpts; candidate line numbers may shift, while edit ranges always refer to the inspected original. No code or line numbers are guessed by the tool. Ordinary planning replies are not automatically continued. Saving a proposal still does not imply CPU validation passed.

Malformed argument errors are bounded by both item count and field-name/message length, so code accidentally emitted as JSON keys is not echoed back as an enormous repair prompt. Full received arguments and errors remain in local diagnostics. The CLI prints a concise error at each rejection instead of only the final retry count.

After two dataset range-proposal failures for the same file in a turn (argument-schema errors or syntax errors after assembly), repair guidance switches to the existing complete-file `updated_content` mode and asks the model to omit `replacements`. It must retain prior intended edits and unrelated source. This adds no public tool or extra retry budget; complete-file proposals still require source/hash, syntax and static review checks. The fallback changes instructions, not source automatically, and cannot guarantee the model supplies a correct complete file.

For an explicit request to save a proposal, a single active task with zero patches, no recorded blocker and no unresolved research questions gets one bounded continuation if the model stops at planning. Analysis-only and negated save requests are excluded. If no proposal is saved after that reminder, the CLI response marks the save incomplete instead of claiming completion or asking for the same permission again. This reminder does not authorize application or execution.

| Action | Behavior |
| --- | --- |
| `add_dataset(plan/inspect_workflow)` | Return repository execution evidence and candidate entry points; the agent follows the relevant source calls before proposing changes. |
| `record_dataset_analysis` | Save analysis incrementally. Omitted areas/fields are retained; missing evidence remains pending with correction guidance. Incomplete analysis does not block proposal storage. |
| `preview_patch` | Save full contents or edits identified by original inclusive line ranges. Redundant legacy `text` metadata is ignored with a warning when `new_lines` is supplied; unknown fields remain rejected. Runtime retains other file proposals and records revisions. Pending analysis is reported, not treated as validation. No target files are written and no code logic is certified. |
| `configure_dataset_validation` | Optional: select existing supported interfaces, original dataset regressions and real local inputs after saving proposals. Unsupported interfaces return explicit limitations; do not restructure a repository merely for the runner. |
| `validate_dataset` | Check real data loading, transformations/wrappers, batches, forward/loss/backward/update and regressions. Component-adapter results do not verify CLI linkage. |
| `add_dataset(status)` | Report proposals, application state and the latest validation result. |

Unresolved questions are retained in the task and block preview creation. After discussing them with the user, the agent submits `open_questions=[]` and records the agreed choices in the integration summary.

| Identifier or grouping | Meaning |
| --- | --- |
| Task ID | One dataset integration task, retained throughout discussion and revisions. |
| Patch batch | A complete proposal version containing all required file patches. `batch_revision` is its version number; there is no separate Batch ID. |
| Patch ID | One exact proposed modification or addition to one file. |

When revising a proposal, submit only files being changed or added. The next batch retains all other Patch IDs and their approval/application states. For example, revising A in `[A:P1, B:P2]` produces `[A:P3, B:P2]`, not just `[A:P3]`. Identical active proposals reuse their IDs without adding a batch; discussion alone does not create a version. Earlier batches remain in history. Removing files from a proposal is not supported.

Each new Patch ID requires CLI approval. Retained applied patches must not be applied again. Applying only some files reports `partially_applied_unverified`; applying all files in the latest complete batch reports `code_applied_unverified`. Neither status certifies a working experiment, and multi-file application is not atomic.

This first version manages the proposal workflow rather than automatically producing a verified dataset integration. It does not enforce complete-read evidence, structured requirement compliance or cross-file interface correctness. Dataset-related edits can still use `preview_patch` directly, so the dataset workflow is not mandatory. Review the proposed code and manually validate it before treating the integration as complete.

### Complete reads and previews

Overview excerpts are capped at 6,000 characters per file; they are not complete edit bases. With `file_path`, inspection returns a JSON string containing `content`, `original_hash`, `size_bytes`, `encoding` and `complete=true`. Single-file content has no added line numbers and is not truncated. Invalid reads return an error string.

Reads use strict UTF-8. Exact original bytes, including BOM and CRLF/LF, determine SHA-256. Preview accepts the FULL proposed file and expected base hash, then preserves the original file's LF or CRLF style before generating the diff and saving the proposal. Originals with mixed LF/CRLF or bare CR endings are rejected; files without line endings use LF for new lines. Apply writes the exact saved proposal without further conversion. No-op, stale, unsafe or oversized proposals are rejected. Previews never write repository files.

Large tool-output offloading is disabled so full reads are not replaced with head/tail summaries. Model context/output limits still apply. Full content does not prove the model understood it or the modification is correct; review the complete diff before approving.

### Python syntax checks

Before saving a `.py` proposal, preview compiles its contents with the agent's running Python interpreter without executing or importing the code, downloading data or writing bytecode caches. This applies to modifications, new files, restored proposals and changes submitted through `add_dataset`; other file types do not undergo Python syntax checks.

Compilation errors return the filename, line, column and reason. For an `add_dataset` submission, if any proposed Python file fails, no new Patch ID or batch is saved, previous proposals and approvals remain unchanged, and no target files are written. Standalone previews also reject invalid Python before saving. `add_model` prepares and saves files sequentially; if patch preparation fails after earlier proposals were saved, those earlier proposals are marked rejected and repository files remain unchanged. The agent can correct the error and submit a new preview.

Passing this check does not verify installed dependencies, function arguments, model shapes or experiment correctness, and does not guarantee compatibility with a different Python version. Dataset execution and scientific validation remain manual.

### Patch revisions

Each successful standalone `preview_patch` call creates an ID and immutable content snapshot. In dataset batches, unchanged active proposals reuse their existing IDs. A revised file supersedes its active parent without overwriting contents. Restoring a version creates a new child of that source with `pending_review` status; approval is not inherited and other branches are not automatically superseded. Parents describe ancestry, not required application order.

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

- One existing or new UTF-8 file per patch; original/proposed content each limited to 500,000 bytes. New files require an existing parent directory. Deletion, rename and multi-file application transactions are not supported.
- Extensions: `.py`, `.yaml`, `.yml`, `.json`, `.toml`, `.ini`, `.cfg`, `.md`, `.txt`. Hidden paths, private-key names, links/junctions, unsafe Windows paths and targets outside the repository are rejected.
- Records/checkpoints are volatile. No persistence, crash recovery or cross-process transaction is implemented or required for this prototype.
- Hash rechecks detect ordinary edits but cannot eliminate the small race before replacement. Basic mode bits are preserved; arbitrary ACLs/metadata may not be.
- Full replacement can include unintended deletions. Python proposals undergo compile-only checks, but passing syntax checks does not establish semantic correctness or runtime compatibility.
- Invocation is synchronous CLI. Programmatic callers must implement trusted confirmation/interrupt handling; helpers are not an authentication boundary.
- Dataset analysis and code proposals are model-assisted; dataset downloads, training and validation remain manual. Model integration uses `add_model`, including its internal scratch-copy smoke test; successful proposals or application do not certify experiment correctness.
- revert_patch cannot undo a patch that created a file (deletion is not supported) and refuses when the file changed after the patch; revert later patches first, newest first.
