#research_agent.py
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType


load_dotenv()


def build_research_agent():
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
    search_tool = DuckDuckGoSearchRun()

    tools = [
        {
            "name": "Web Search",
            "func": search_tool.run,
            "description": "Search the web for current public information.",
        }
    ]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )
    return agent


def main():
    print("=== AI Research Agent ===")
    query = input("Enter research topic: ").strip()

    agent = build_research_agent()
    result = agent.invoke(
        {
            "input": (
                f"Research this topic: {query}. "
                "Summarize key findings, important facts, and practical insights."
            )
        }
    )

    print("\n=== Research Output ===")
    print(result["output"])


if __name__ == "__main__":
    main()