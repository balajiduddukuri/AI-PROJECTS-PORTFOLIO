'''
CrewAI code from your orchestrator setup, using OpenAI via ChatOpenAI 
'''
import os
import logging
from typing import Optional
from dotenv import load_dotenv

from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from langchain_openai import ChatOpenAI


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("AgenticOrchestrator")


class ResearchTool(BaseTool):
    name: str = "Research Tool"
    description: str = (
        "Use this tool to research a topic and gather factual information. "
        "Input: a research query string. "
        "Output: a structured summary of key facts, trends, and insights."
    )

    def _run(self, query: str) -> str:
        logger.info(f"ResearchTool invoked with query: '{query}'")
        return f"""
RESEARCH FINDINGS for: "{query}"

1. OVERVIEW:
   - This topic is actively evolving with significant developments in 2024-2025.

2. KEY TRENDS:
   - Increased adoption of agentic AI workflows in enterprise environments.
   - Multi-agent systems showing strong productivity gains.

3. CHALLENGES:
   - Hallucination risk.
   - Cost management.
   - Governance and explainability.
"""


class AnalysisTool(BaseTool):
    name: str = "Analysis Tool"
    description: str = (
        "Use this tool to analyze research data and extract strategic insights. "
        "Input: raw research text or data. "
        "Output: structured analysis with recommendations and action items."
    )

    def _run(self, data: str) -> str:
        logger.info("AnalysisTool invoked for data analysis.")
        return f"""
STRATEGIC ANALYSIS REPORT

INPUT SUMMARY:
{data[:300]}...

INSIGHTS:
- Early mover advantage exists.
- Observability and governance are essential.
- Framework-agnostic design reduces lock-in.

RECOMMENDATIONS:
- Start with low-risk workflows.
- Add monitoring and guardrails.
- Keep the architecture modular.
"""


class AgenticOrchestrator:
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.7,
        verbose: bool = True,
    ):
        self.verbose = verbose
        self.model_name = model_name
        self.temperature = temperature

        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError(
                "OpenAI API key is required. Pass it as 'openai_api_key' or set OPENAI_API_KEY."
            )

        os.environ["OPENAI_API_KEY"] = self.openai_api_key
        self._llm = self._initialize_llm()
        self._tools = self._initialize_tools()

    def _initialize_llm(self) -> ChatOpenAI:
        return ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            api_key=self.openai_api_key,
        )

    def _initialize_tools(self) -> dict:
        return {
            "research": ResearchTool(),
            "analysis": AnalysisTool(),
        }

    def _create_researcher_agent(self) -> Agent:
        return Agent(
            role="Senior Research Specialist",
            goal="Conduct comprehensive research and deliver a factual research brief.",
            backstory=(
                "You are an experienced research specialist focused on technology intelligence "
                "and market analysis."
            ),
            tools=[self._tools["research"]],
            llm=self._llm,
            verbose=self.verbose,
            allow_delegation=False,
            max_iter=5,
        )

    def _create_analyst_agent(self) -> Agent:
        return Agent(
            role="Strategic AI Analyst",
            goal="Analyze research findings and produce strategic recommendations.",
            backstory=(
                "You are a strategic analyst specializing in AI transformation and business outcomes."
            ),
            tools=[self._tools["analysis"]],
            llm=self._llm,
            verbose=self.verbose,
            allow_delegation=False,
            max_iter=5,
        )

    def _create_writer_agent(self) -> Agent:
        return Agent(
            role="Executive Content Strategist",
            goal="Turn research and analysis into a polished executive briefing.",
            backstory=(
                "You are an executive communications expert who writes concise, persuasive reports."
            ),
            tools=[],
            llm=self._llm,
            verbose=self.verbose,
            allow_delegation=False,
            max_iter=3,
        )

    def _create_research_task(self, topic: str, agent: Agent) -> Task:
        return Task(
            description=(
                f"Research the topic thoroughly: {topic}\n"
                "Include overview, key facts, trends, challenges, and stakeholders."
            ),
            expected_output="A structured research brief.",
            agent=agent,
        )

    def _create_analysis_task(self, topic: str, agent: Agent) -> Task:
        return Task(
            description=(
                f"Analyze the research findings for: {topic}\n"
                "Identify patterns, risks, opportunities, and recommendations."
            ),
            expected_output="A strategic analysis report.",
            agent=agent,
        )

    def _create_writing_task(self, topic: str, agent: Agent) -> Task:
        return Task(
            description=(
                f"Write an executive briefing on: {topic}\n"
                "Use the research brief and analysis to create a polished final report."
            ),
            expected_output="A complete executive briefing document.",
            agent=agent,
        )

    def run(self, topic: str) -> str:
        researcher = self._create_researcher_agent()
        analyst = self._create_analyst_agent()
        writer = self._create_writer_agent()

        research_task = self._create_research_task(topic, researcher)
        analysis_task = self._create_analysis_task(topic, analyst)
        writing_task = self._create_writing_task(topic, writer)

        analysis_task.context = [research_task]
        writing_task.context = [research_task, analysis_task]

        crew = Crew(
            agents=[researcher, analyst, writer],
            tasks=[research_task, analysis_task, writing_task],
            process=Process.sequential,
            verbose=self.verbose,
        )

        result = crew.kickoff()
        return str(result)


def main():
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = input("Enter your OpenAI API key: ").strip()

    topic = input("Enter a topic: ").strip() or "Impact of agentic AI on enterprise SDLC"

    orchestrator = AgenticOrchestrator(
        openai_api_key=api_key,
        model_name="gpt-4o-mini",
        temperature=0.7,
        verbose=True,
    )

    result = orchestrator.run(topic)
    print(result)


if __name__ == "__main__":
    main()
