from dotenv import load_dotenv
from deepagents import create_deep_agent
from model_factory import get_LLMagent_model
from langchain_core.messages import HumanMessage
from tools import *

load_dotenv()

model = get_LLMagent_model()

agent = create_deep_agent(
    model=model,
    tools=TOOLS,
    system_prompt="""You are a research assistant that analyzes machine learning GitHub repos.
    To find a repo's supported models: clone it with clone_repo, run list_models for leads,
    then open the cited files with read_repo_file to confirm what each hit really is (a full
    model, a layer, a config option). Use search_repo for anything the scanner missed.
    Answer with the confirmed list and cite file:line for each model."""    
)

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
