from dotenv import load_dotenv
from deepagents import create_deep_agent
from langchain.agents.middleware import ModelRetryMiddleware
from model_factory import get_LLMagent_model
from langchain_core.messages import HumanMessage
from tools import *

load_dotenv()

model = get_LLMagent_model()

SYSTEM_PROMPT = """You are a research assistant that analyzes machine learning GitHub repos.
Workflow: clone the repo with clone_repo, run list_models for leads, then open the cited
files with read_repo_file to confirm them. Use search_repo for anything the scanner missed.
"Supported" means a model that can be selected through the repo's own model-selection code
(a get_model-style function, an argparse --model option, or a registry), not every class or
function defined in the repo. Give file:line for each supported model, and list models that
are defined but not selectable separately."""

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

        # call Agent with the full conversation history
        result = agent.invoke({"messages": messages})

        # update conversation history, keeping Agent and tool messages
        messages = result["messages"]

        # show which tools the agent called this turn
        for m in messages[prev:]:
            for call in getattr(m, "tool_calls", None) or []:
                print(f"  [tool] {call['name']}({call['args']})")

        if isinstance(response, str):
            print(f"Research Assistant: {response}\n")
        else:
            text = "\n".join(
                item["text"]
                for item in response
                if isinstance(item, dict) and item.get("type") == "text"
            )
            print(f"Research Assistant: {text}\n")


if __name__ == "__main__":
    chat()
