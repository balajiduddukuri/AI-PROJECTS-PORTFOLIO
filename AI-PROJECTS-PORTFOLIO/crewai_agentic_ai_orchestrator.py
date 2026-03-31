"""
=============================================================================
Agentic AI Orchestrator using CrewAI + OpenAI
=============================================================================

Description:
    A fully documented, production-ready Agentic AI Orchestrator built with
    CrewAI and OpenAI GPT models. Demonstrates multi-agent collaboration
    with specialized roles: Researcher, Analyst, and Writer agents working
    together to complete complex tasks autonomously.

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                   Agentic AI Orchestrator                       │
    │                                                                 │
    │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
    │  │  Researcher  │→ │   Analyst    │→ │       Writer         │  │
    │  │   Agent      │  │   Agent      │  │       Agent          │  │
    │  └──────────────┘  └──────────────┘  └──────────────────────┘  │
    │         │                 │                     │               │
    │    [Research Task]  [Analysis Task]     [Content Task]          │
    │                                                                 │
    │                    CrewAI Crew (Sequential)                     │
    └─────────────────────────────────────────────────────────────────┘

Dependencies:
    pip install crewai crewai-tools openai python-dotenv

Usage:
    python agentic_ai_orchestrator.py

    Or import and call programmatically:
        from agentic_ai_orchestrator import AgenticOrchestrator
        orchestrator = AgenticOrchestrator(openai_api_key="sk-...")
        result = orchestrator.run(topic="AI trends in 2025")

Author: Balu
Version: 1.0.0
"""

import os
import logging
from typing import Optional
from dotenv import load_dotenv

# CrewAI imports
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool

# OpenAI LLM configuration
from langchain_openai import ChatOpenAI

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("AgenticOrchestrator")


# ---------------------------------------------------------------------------
# Custom Tool: Web Research Simulator
# ---------------------------------------------------------------------------

class ResearchTool(BaseTool):
    """
    A custom CrewAI tool that simulates web research by generating
    structured research data based on a given query.

    In production, replace the `_run` method body with actual web scraping,
    API calls (e.g., Serper, Tavily, or BraveSearch), or RAG pipeline lookups.

    Attributes:
        name (str): Tool identifier used by the agent's reasoning loop.
        description (str): Natural language description so the LLM knows
                           when and how to invoke this tool.
    """

    name: str = "Research Tool"
    description: str = (
        "Use this tool to research a topic and gather factual information. "
        "Input: a research query string. "
        "Output: a structured summary of key facts, trends, and insights."
    )

    def _run(self, query: str) -> str:
        """
        Execute a research query and return synthesized findings.

        Args:
            query (str): The research question or topic to investigate.

        Returns:
            str: A structured research summary with key points and data.
        """
        logger.info(f"ResearchTool invoked with query: '{query}'")

        # Simulate research output (replace with real API in production)
        research_output = f"""
RESEARCH FINDINGS for: "{query}"
{'=' * 50}

1. OVERVIEW:
   - This topic is actively evolving with significant developments in 2024-2025.
   - Key stakeholders include enterprises, researchers, and technology providers.

2. KEY TRENDS:
   - Increased adoption of agentic AI workflows in enterprise environments.
   - Multi-agent systems showing 3x productivity gains vs single-agent approaches.
   - Open-source frameworks (CrewAI, AutoGen, LangGraph) gaining market traction.

3. STATISTICS:
   - 68% of Fortune 500 companies piloting agentic AI in 2025.
   - Average ROI from AI orchestration: 240% over 18 months.
   - Token efficiency improved by 45% with structured agent delegation.

4. CHALLENGES:
   - Hallucination rates remain a concern for autonomous decision-making.
   - Cost management in multi-agent token chains is critical.
   - Governance and explainability frameworks are still maturing.

5. SOURCES (simulated):
   - MIT Technology Review 2025 AI Report
   - Gartner Hype Cycle for AI 2025
   - Stanford AI Index Annual Report
"""
        return research_output


# ---------------------------------------------------------------------------
# Custom Tool: Data Analysis Simulator
# ---------------------------------------------------------------------------

class AnalysisTool(BaseTool):
    """
    A custom CrewAI tool that performs structured analysis on research data.

    In production, replace with actual analytics pipelines, statistical
    analysis (pandas/numpy), or LLM-powered chain-of-thought reasoning.

    Attributes:
        name (str): Tool identifier used by the agent's reasoning loop.
        description (str): Natural language description for LLM tool selection.
    """

    name: str = "Analysis Tool"
    description: str = (
        "Use this tool to analyze research data and extract strategic insights. "
        "Input: raw research text or data. "
        "Output: structured analysis with recommendations and action items."
    )

    def _run(self, data: str) -> str:
        """
        Analyze provided data and produce strategic insights.

        Args:
            data (str): Raw research data or text to analyze.

        Returns:
            str: Structured analysis with insights, patterns, and recommendations.
        """
        logger.info("AnalysisTool invoked for data analysis.")

        analysis_output = f"""
STRATEGIC ANALYSIS REPORT
{'=' * 50}

INPUT DATA SUMMARY:
{data[:300]}...

ANALYSIS RESULTS:

1. SENTIMENT ANALYSIS:
   - Overall tone: Optimistic with cautionary notes
   - Confidence score: 78% (High)
   - Risk factors: Moderate

2. KEY PATTERNS IDENTIFIED:
   - Pattern A: Rapid iteration cycles (2-4 week sprints in AI agent projects)
   - Pattern B: Cost-performance optimization is top priority for CIOs
   - Pattern C: Hybrid human-AI collaboration models outperform fully autonomous

3. STRATEGIC INSIGHTS:
   ✅ OPPORTUNITY: Early-mover advantage in enterprise agentic AI deployment
   ✅ OPPORTUNITY: Vertical-specific agent specialization (legal, finance, HR)
   ⚠️  RISK: Vendor lock-in with proprietary LLM ecosystems
   ⚠️  RISK: Regulatory uncertainty in EU AI Act compliance

4. RECOMMENDATIONS:
   → Prioritize framework-agnostic architecture (CrewAI + LangGraph compatible)
   → Invest in observability and monitoring for multi-agent pipelines
   → Build internal AI governance committee before scaling deployment
   → Pilot with low-stakes workflows before critical business processes

5. CONFIDENCE LEVELS:
   - Short-term outlook (6 months): HIGH confidence
   - Long-term outlook (2+ years): MODERATE confidence
"""
        return analysis_output


# ---------------------------------------------------------------------------
# Agentic Orchestrator Class
# ---------------------------------------------------------------------------

class AgenticOrchestrator:
    """
    Main orchestrator class that configures and runs a multi-agent CrewAI crew.

    This class manages the lifecycle of:
    - LLM initialization (OpenAI GPT)
    - Agent creation with roles, goals, and backstories
    - Task definition and assignment
    - Crew assembly and execution

    The default crew uses a Sequential process where:
        Researcher → Analyst → Writer

    Each agent passes its output as context to the next agent in the chain.

    Args:
        openai_api_key (str, optional): OpenAI API key. Falls back to the
            OPENAI_API_KEY environment variable if not provided.
        model_name (str): OpenAI model to use. Defaults to "gpt-4o-mini"
            for cost efficiency. Use "gpt-4o" for higher quality outputs.
        temperature (float): LLM sampling temperature (0.0 = deterministic,
            1.0 = highly creative). Defaults to 0.7 for balanced creativity.
        verbose (bool): Enable detailed agent action logging. Defaults to True.

    Raises:
        ValueError: If no OpenAI API key is found in args or environment.

    Example:
        >>> orchestrator = AgenticOrchestrator(
        ...     openai_api_key="sk-...",
        ...     model_name="gpt-4o",
        ...     temperature=0.5
        ... )
        >>> result = orchestrator.run(topic="Future of autonomous AI agents")
        >>> print(result)
    """

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

        # Resolve API key: argument > environment variable
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError(
                "OpenAI API key is required. Pass it as 'openai_api_key' parameter "
                "or set the OPENAI_API_KEY environment variable."
            )

        # Set globally for underlying OpenAI clients
        os.environ["OPENAI_API_KEY"] = self.openai_api_key

        logger.info(f"Initializing AgenticOrchestrator with model: {model_name}")
        self._llm = self._initialize_llm()
        self._tools = self._initialize_tools()

    # -----------------------------------------------------------------------
    # Private Initialization Methods
    # -----------------------------------------------------------------------

    def _initialize_llm(self) -> ChatOpenAI:
        """
        Initialize the LangChain ChatOpenAI LLM instance.

        Returns:
            ChatOpenAI: Configured LLM instance for all agents.
        """
        logger.info(f"Loading LLM: {self.model_name} (temperature={self.temperature})")
        return ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            api_key=self.openai_api_key,
        )

    def _initialize_tools(self) -> dict:
        """
        Initialize and register all custom tools available to agents.

        Returns:
            dict: Mapping of tool names to tool instances.
        """
        logger.info("Initializing agent tools...")
        return {
            "research": ResearchTool(),
            "analysis": AnalysisTool(),
        }

    # -----------------------------------------------------------------------
    # Agent Factory Methods
    # -----------------------------------------------------------------------

    def _create_researcher_agent(self) -> Agent:
        """
        Create the Researcher Agent responsible for gathering information.

        The Researcher agent uses the ResearchTool to collect relevant
        data and synthesize it into a structured research brief.

        Returns:
            Agent: Fully configured CrewAI Researcher agent.
        """
        return Agent(
            role="Senior Research Specialist",
            goal=(
                "Conduct comprehensive research on the given topic and deliver "
                "a detailed, factual research brief with data-backed insights."
            ),
            backstory=(
                "You are a veteran research specialist with 15 years of experience "
                "in technology intelligence and market analysis. You have an "
                "exceptional ability to synthesize complex information from diverse "
                "sources into clear, actionable research briefs. You are meticulous, "
                "citation-focused, and always verify information before reporting."
            ),
            tools=[self._tools["research"]],
            llm=self._llm,
            verbose=self.verbose,
            allow_delegation=False,  # Researcher works independently
            max_iter=5,              # Limit reasoning loops to prevent runaway
        )

    def _create_analyst_agent(self) -> Agent:
        """
        Create the Analyst Agent responsible for interpreting research data.

        The Analyst agent receives the Researcher's output and applies
        structured analytical frameworks to extract strategic insights.

        Returns:
            Agent: Fully configured CrewAI Analyst agent.
        """
        return Agent(
            role="Strategic AI Analyst",
            goal=(
                "Analyze the research findings, identify patterns and risks, "
                "and produce a strategic insights report with clear recommendations."
            ),
            backstory=(
                "You are a strategic analyst specializing in AI and technology "
                "transformation. You have advised Fortune 500 companies on "
                "AI adoption strategies. Your analyses are always structured, "
                "evidence-based, and focused on actionable business outcomes. "
                "You excel at separating signal from noise in complex data sets."
            ),
            tools=[self._tools["analysis"]],
            llm=self._llm,
            verbose=self.verbose,
            allow_delegation=False,
            max_iter=5,
        )

    def _create_writer_agent(self) -> Agent:
        """
        Create the Writer Agent responsible for producing the final deliverable.

        The Writer agent synthesizes research and analysis into a polished,
        audience-appropriate final report or document.

        Returns:
            Agent: Fully configured CrewAI Writer agent.
        """
        return Agent(
            role="Executive Content Strategist",
            goal=(
                "Transform research and analysis into a compelling, well-structured "
                "executive report that clearly communicates key findings and "
                "strategic recommendations to a C-suite audience."
            ),
            backstory=(
                "You are an executive communications expert who has written "
                "hundreds of board-level strategy documents, whitepapers, and "
                "technology briefings. You have a gift for distilling complex "
                "technical topics into clear, persuasive narratives. You always "
                "write with purpose, precision, and the audience firmly in mind."
            ),
            tools=[],               # Writer synthesizes from context, no external tools
            llm=self._llm,
            verbose=self.verbose,
            allow_delegation=False,
            max_iter=3,
        )

    # -----------------------------------------------------------------------
    # Task Factory Methods
    # -----------------------------------------------------------------------

    def _create_research_task(self, topic: str, agent: Agent) -> Task:
        """
        Define the research task for the Researcher agent.

        Args:
            topic (str): The subject or question to research.
            agent (Agent): The Researcher agent assigned to this task.

        Returns:
            Task: Configured CrewAI Task for research execution.
        """
        return Task(
            description=(
                f"Research the following topic thoroughly:\n\n"
                f"TOPIC: {topic}\n\n"
                f"Your deliverable must include:\n"
                f"1. A concise topic overview (3-4 sentences)\n"
                f"2. At least 5 key facts or data points\n"
                f"3. Current trends and developments\n"
                f"4. Key challenges or open questions\n"
                f"5. Notable players or stakeholders\n\n"
                f"Use the Research Tool to gather data. Be specific and cite sources."
            ),
            expected_output=(
                "A structured research brief (400-600 words) covering overview, "
                "key facts, trends, challenges, and stakeholders. Formatted with "
                "clear sections and bullet points."
            ),
            agent=agent,
        )

    def _create_analysis_task(self, topic: str, agent: Agent) -> Task:
        """
        Define the analysis task for the Analyst agent.

        Args:
            topic (str): The original research topic for context.
            agent (Agent): The Analyst agent assigned to this task.

        Returns:
            Task: Configured CrewAI Task for strategic analysis.
        """
        return Task(
            description=(
                f"Analyze the research findings provided about: {topic}\n\n"
                f"Your analysis must include:\n"
                f"1. Pattern recognition across the research data\n"
                f"2. SWOT-style opportunities and risks assessment\n"
                f"3. Strategic recommendations (minimum 3, maximum 6)\n"
                f"4. Confidence levels for each recommendation\n"
                f"5. A prioritized action roadmap\n\n"
                f"Use the Analysis Tool on the research output. "
                f"Focus on actionable, business-relevant insights."
            ),
            expected_output=(
                "A strategic analysis report (500-700 words) with pattern insights, "
                "opportunities, risks, ranked recommendations, and a 90-day "
                "action roadmap. Include confidence percentages."
            ),
            agent=agent,
        )

    def _create_writing_task(self, topic: str, agent: Agent) -> Task:
        """
        Define the writing task for the Writer agent.

        Args:
            topic (str): The original research topic for context.
            agent (Agent): The Writer agent assigned to this task.

        Returns:
            Task: Configured CrewAI Task for final report generation.
        """
        return Task(
            description=(
                f"Write a polished executive briefing on: {topic}\n\n"
                f"Using the research brief and strategic analysis provided, "
                f"create a professional document with:\n\n"
                f"1. EXECUTIVE SUMMARY (100 words max)\n"
                f"2. KEY FINDINGS (5-7 bullet points)\n"
                f"3. STRATEGIC IMPLICATIONS (2-3 paragraphs)\n"
                f"4. RECOMMENDED ACTIONS (numbered, priority-ordered)\n"
                f"5. CONCLUSION (50 words max)\n\n"
                f"Tone: Professional, authoritative, and forward-looking. "
                f"Audience: C-suite executives with limited time. "
                f"Language: Clear, jargon-free, decisive."
            ),
            expected_output=(
                "A complete executive briefing document (600-800 words) with all "
                "five required sections. Properly formatted with headers, bullets, "
                "and numbered lists. Ready for immediate distribution."
            ),
            agent=agent,
        )

    # -----------------------------------------------------------------------
    # Public Execution Method
    # -----------------------------------------------------------------------

    def run(self, topic: str) -> str:
        """
        Execute the full multi-agent orchestration pipeline.

        Assembles the crew with all agents and tasks, then kicks off the
        sequential process: Research → Analysis → Writing.

        Args:
            topic (str): The subject or question the crew should work on.
                         Can be a specific question, a broad topic, or a
                         business challenge that requires analysis.

        Returns:
            str: The final output produced by the Writer agent — a complete
                 executive briefing on the given topic.

        Raises:
            Exception: Propagates any CrewAI or OpenAI API exceptions with
                       contextual logging for easier debugging.

        Example:
            >>> result = orchestrator.run("Impact of agentic AI on software development")
            >>> print(result)
        """
        logger.info(f"Starting orchestration for topic: '{topic}'")
        logger.info("=" * 60)

        # Step 1: Instantiate agents
        researcher = self._create_researcher_agent()
        analyst = self._create_analyst_agent()
        writer = self._create_writer_agent()

        # Step 2: Define tasks with agent assignments
        research_task = self._create_research_task(topic, researcher)
        analysis_task = self._create_analysis_task(topic, analyst)
        writing_task = self._create_writing_task(topic, writer)

        # Step 3: Link tasks so analyst and writer receive prior context
        # CrewAI passes previous task outputs as context automatically
        # in sequential mode when tasks are ordered correctly.
        analysis_task.context = [research_task]
        writing_task.context = [research_task, analysis_task]

        # Step 4: Assemble the Crew
        crew = Crew(
            agents=[researcher, analyst, writer],
            tasks=[research_task, analysis_task, writing_task],
            process=Process.sequential,  # Each agent waits for previous output
            verbose=self.verbose,
        )

        # Step 5: Execute and capture the final result
        try:
            logger.info("Crew kickoff initiated...")
            result = crew.kickoff()
            logger.info("Orchestration completed successfully.")
            return str(result)

        except Exception as e:
            logger.error(f"Orchestration failed: {e}", exc_info=True)
            raise


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    """
    Command-line entry point for the Agentic AI Orchestrator.

    Loads the OpenAI API key from the .env file or environment, prompts
    for a research topic (or uses a default), and runs the crew.

    Environment Variables:
        OPENAI_API_KEY: Your OpenAI API key (required).
                        Can also be set in a .env file in the project root.

    Usage:
        python agentic_ai_orchestrator.py
    """
    # Load .env file if present (supports local development)
    load_dotenv()

    print("\n" + "=" * 60)
    print("   Agentic AI Orchestrator | CrewAI + OpenAI")
    print("=" * 60)

    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = input("\nEnter your OpenAI API key: ").strip()
        if not api_key:
            print("ERROR: OpenAI API key is required.")
            return

    # Get research topic
    default_topic = "The impact of multi-agent AI systems on enterprise software development"
    topic = input(f"\nEnter a research topic\n[Default: {default_topic}]\n> ").strip()
    if not topic:
        topic = default_topic

    print(f"\n📋 Topic: {topic}")
    print("🤖 Agents: Researcher → Analyst → Writer")
    print("🔁 Process: Sequential")
    print("=" * 60)
    print("Starting crew... (this may take 30-90 seconds)\n")

    try:
        # Initialize and run the orchestrator
        orchestrator = AgenticOrchestrator(
            openai_api_key=api_key,
            model_name="gpt-4o-mini",   # Cost-efficient; swap to gpt-4o for quality
            temperature=0.7,
            verbose=True,
        )

        result = orchestrator.run(topic=topic)

        print("\n" + "=" * 60)
        print("   FINAL REPORT")
        print("=" * 60)
        print(result)
        print("=" * 60)

    except ValueError as e:
        print(f"\n❌ Configuration Error: {e}")
    except Exception as e:
        print(f"\n❌ Execution Error: {e}")
        logger.error("Fatal error during orchestration", exc_info=True)


# ---------------------------------------------------------------------------
# Module Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
