from pyexpat.errors import messages

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
    """,
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
        print("[DEBUG] Sending request to agent...", flush=True)
        
        try:
            result = agent.invoke(
                {"messages": messages},
                config={"recursion_limit": 20},
            )
        except Exception as e:
            print(f"[ERROR] {type(e).__name__}: {e}", flush=True)
            continue
        
        print("[DEBUG] Agent invocation completed.", flush=True)
        
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


if __name__ == "__main__":
    chat()
