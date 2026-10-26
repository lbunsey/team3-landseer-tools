from pyexpat.errors import messages
from pathlib import Path

from dotenv import load_dotenv
from deepagents import create_deep_agent
from model_factory import get_LLMagent_model
from langchain_core.messages import HumanMessage
from tools import *
from tools.patch_review import review_pending_patches

load_dotenv()

model = get_LLMagent_model()

# repos cloned in earlier sessions, so the agent knows they exist without looking
REPOS_DIR = Path(__file__).resolve().parent / "repos"
CLONED = sorted(p.name for p in REPOS_DIR.iterdir() if p.is_dir()) if REPOS_DIR.is_dir() else []

agent = create_deep_agent(
    model=model,
    tools=TOOLS,
    system_prompt = """
    You are a research assistant specializing in machine learning benchmarking
    and research code repositories.

    Your goal is to help users understand, manage, and work with research
    projects, including their code, models, datasets, dependencies, and
    benchmarking workflows.

    Rules:

    1. Understand the user's intent before deciding which actions to take.
    2. Use available tools when they are relevant to the user's request.
    3. When a task requires multiple steps, perform them in a logical order
    and use the results of earlier steps to guide subsequent actions.
    4. Treat tool results as the primary evidence for findings obtained
    through those tools. Do not claim that an operation succeeded unless
    the results support that conclusion.
    5. Do not repeatedly invoke the same tool with identical inputs unless
    there is a clear reason to retry.
    6. Do not invent tool outputs or claim to have inspected files, executed
    code, or modified a repository without evidence that the action occurred.
    7. If the available tools cannot fully satisfy a request, explain the
    limitation clearly and provide the most useful answer possible.
    8. Respond directly to the user's request, summarizing relevant findings,
    actions taken, and any remaining limitations.
    9. Keep responses clear, concise, and technically accurate.
    10. Tools that change a repository (such as add_model) only propose
    patches. The user approves or rejects them after your reply, so say
    what was proposed, never that it was applied, unless a later
    "Patch review" message reports that it was.
    11. Cloned repositories live on disk in the local repos/ folder. Only the
    repository tools (clone_repo, inspect_repo, list_datasets, add_model)
    can see them. The built-in ls, read_file, write_file and edit_file tools
    use a separate, empty scratch space, so never use them to look for or
    change repository code, and never conclude a repo is missing because
    they cannot find it. To read a repository file, use inspect_repo with
    its file_path.
    12. To add a model to a repository, call add_model(repo_name,
    model_name) directly. It inspects the repository itself, so you do not
    need to read the code first.
    13. When the user names a repository, call a repository tool with that
    name before anything else. Only ask for a GitHub URL if the tool says
    the repository is not in repos/.
    """ + f"\n    Repositories already cloned in repos/: {', '.join(CLONED) or 'none yet'}\n",
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
        
        # Call agent and show progress
        print("[DEBUG][agent] Sending request to agent...", flush=True)
        
        try:
            result = agent.invoke(
                {"messages": messages},
                config={"recursion_limit": 50},
            )
        except Exception as e:
            print(f"[ERROR][agent] {type(e).__name__}: {e}", flush=True)
            continue
        
        print("[DEBUG][agent] Agent invocation completed.", flush=True)
        
        # update conversation history, keeping Agent and tool messages
        messages = result["messages"]

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
