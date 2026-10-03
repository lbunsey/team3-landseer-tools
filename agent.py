from dotenv import load_dotenv
from deepagents import create_deep_agent
from model_factory import get_LLMagent_model
from langchain_core.messages import HumanMessage

load_dotenv()

model = get_LLMagent_model()

agent = create_deep_agent(
    model=model,
    system_prompt="You are a helpful research assistant."
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

        # call Agent with the full conversation history
        result = agent.invoke({"messages": messages})

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