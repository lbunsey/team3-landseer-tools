from pyexpat.errors import messages
from pathlib import Path

from dotenv import load_dotenv
from deepagents import create_deep_agent
from langchain.agents.middleware import ModelRetryMiddleware
from model_factory import get_LLMagent_model
from langchain_core.messages import HumanMessage
from tools import *
from tools.patch_review import review_pending_patches

load_dotenv()

model = get_LLMagent_model()

# repos cloned in earlier sessions, so the agent knows they exist without looking
REPOS_DIR = Path(__file__).resolve().parent / "repos"
CLONED = sorted(p.name for p in REPOS_DIR.iterdir() if p.is_dir()) if REPOS_DIR.is_dir() else []

SYSTEM_PROMPT = """You are a research assistant that analyzes machine learning GitHub repos.
Workflow: clone the repo with clone_repo, run list_models for leads, then open the cited
files with read_repo_file to confirm them. Use search_repo for anything the scanner missed.
"Supported" means a model that can be selected through the repo's own model-selection code
(a get_model-style function, an argparse --model option, or a registry), not every class or
function defined in the repo. Give file:line for each supported model, and list models that
are defined but not selectable separately.

Cloned repos live on disk in the local repos/ folder, and only the repository tools
(clone_repo, list_models, list_datasets, list_repo_files, read_repo_file, search_repo,
inspect_repo, add_model) can see them. The built-in ls, read_file, write_file and edit_file
tools use a separate, empty scratch space: never use them for repository code, and never
conclude a repo is missing because they cannot find it. When the user names a repo, call a
repository tool with that name first; only ask for a GitHub URL if the tool says it is not
in repos/.

To add a model to a repo, call add_model(repo_name, model_name) directly; it inspects the
repo itself. Tools that change a repo only propose patches: the user approves or rejects
them at a prompt after your reply, so say what was proposed, never that it was applied,
unless a later "Patch review" message reports that it was."""
SYSTEM_PROMPT += f"\nRepositories already cloned in repos/: {', '.join(CLONED) or 'none yet'}"

retry = ModelRetryMiddleware(
    max_retries=5,
    retry_on=lambda e: "rate limit" in str(e).lower(),
    on_failure="error",
    initial_delay=10.0,
    backoff_factor=2.0,
    max_delay=60.0,
)

agent = create_deep_agent(
    model=model,
    tools=TOOLS,
    system_prompt=SYSTEM_PROMPT,
    middleware=[retry],
)

def chat():
    messages = []

    print("Research Assistant: Hello! How can I help you today?")
    print("Type 'exit' or 'quit' to end the conversation.\n")

    while True:
        user_input = input("You: ").strip()

        # end conversation
        if user_input.lower() in ["exit", "quit"]:
            print("Research Assistant: Goodbye!")
            break

        # ignore empty input
        if not user_input:
            continue

        # add user message to conversation history
        messages.append(HumanMessage(content=user_input))

        # remember how many messages exist before the agent runs
        prev = len(messages)

        # Call agent and show progress
        print("[DEBUG][agent] Sending request to agent...", flush=True)
        
        try:
            result = agent.invoke(
                {"messages": messages},
                config={"recursion_limit": 50},
            )
        except Exception as e:
            messages.pop()  # drop the unanswered message so the history stays consistent
            print(f"[ERROR][agent] {type(e).__name__}: {e}", flush=True)
            continue
        
        print("[DEBUG][agent] Agent invocation completed.", flush=True)
        
        # update conversation history, keeping Agent and tool messages
        messages = result["messages"]

        # show which tools the agent called this turn
        for m in messages[prev:]:
            for call in getattr(m, "tool_calls", None) or []:
                print(f"  [tool] {call['name']}({call['args']})")

        # get and display the latest response
        response = messages[-1].content

        if isinstance(response, str):
            print(f"Research Assistant: {response}\n")
        else:
            text = "\n".join(
                item["text"]
                for item in response
                if isinstance(item, dict) and item.get("type") == "text"
            )
            print(f"Research Assistant: {text}\n")

        # proposed repo changes need a human yes/no before anything is written
        review = review_pending_patches()
        if review:
            messages.append(HumanMessage(content=f"[CLI] {review}"))


if __name__ == "__main__":
    chat()
