import os

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI


def get_LLMagent_model():
    provider = os.getenv("LLM_PROVIDER", "rcac").lower()

    if provider == "rcac":
        return ChatOpenAI(
            model=os.getenv("RCAC_MODEL"),
            base_url=os.getenv("RCAC_BASE_URL"),
            api_key=os.getenv("RCAC_API_KEY") or "not-needed",
        )

    elif provider == "claude":
        return ChatAnthropic(
            model=os.getenv("CLAUDE_MODEL"),
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )

    elif provider == "openai":
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL"),
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    elif provider == "google":
        return ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_MODEL", "gemini-3.8-flash"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )

    else:
        raise ValueError(f"Unsupported provider: {provider}")