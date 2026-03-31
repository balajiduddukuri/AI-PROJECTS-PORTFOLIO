"""
=============================================================================
6-Agent SDLC Agentic AI Orchestrator using CrewAI + OpenAI
=============================================================================

Description:
    A production-style multi-agent SDLC orchestrator built with CrewAI and
    OpenAI. This version models an AI-led software delivery lifecycle using
    six specialized agents:

    1. Business Analyst
    2. Solution Architect
    3. Developer & ML Engineer
    4. QA & Security Engineer
    5. DevOps Engineer
    6. Monitoring & Feedback Analyst

    The crew runs in a sequential process, where each agent receives context
    from the previous SDLC phase and produces traceable artifacts that can be
    reviewed, extended, and operationalized.

Use Cases:
    - AI-led SDLC demonstrations
    - Multi-agent enterprise delivery workflows
    - Requirements-to-monitoring orchestration patterns
    - CrewAI starter template for software lifecycle automation

Dependencies:
    pip install crewai crewai-tools langchain-openai python-dotenv

Usage:
    python agentic_ai_sdlc_orchestrator_6_agents.py

Author: Balu
Version: 1.0.0
"""

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
logger = logging.getLogger("SixAgentSDLCOrchestrator")


class RequirementsTool(BaseTool):
    name: str = "Requirements Tool"
    description: str = (
        "Use this tool to convert a product vision into structured business analysis artifacts, "
        "including scope, user stories, acceptance criteria, assumptions, dependencies, and risks."
    )

    def _run(self, product_brief: str) -> str:
        logger.info("RequirementsTool invoked.")
        return f"""
BUSINESS ANALYSIS ARTIFACT
{'=' * 60}
INPUT PRODUCT BRIEF:
{product_brief}

OUTPUTS:
1. PROBLEM STATEMENT:
   - Clarified business objective and target users.
2. SCOPE:
   - Defined in-scope and out-of-scope capabilities.
3. USER STORIES:
   - As a user, I want to complete core workflows efficiently.
   - As an admin, I want visibility into operational status.
4. ACCEPTANCE CRITERIA:
   - Functional flows are validated.
   - Auditability and non-functional constraints are captured.
5. RISKS & DEPENDENCIES:
   - Dependency on API reliability, data quality, and compliance rules.
"""


class ArchitectureTool(BaseTool):
    name: str = "Architecture Tool"
    description: str = (
        "Use this tool to design a modular solution architecture from business requirements, "
        "including components, data flow, integration points, and non-functional considerations."
    )

    def _run(self, requirements: str) -> str:
        logger.info("ArchitectureTool invoked.")
        return f"""
SOLUTION ARCHITECTURE ARTIFACT
{'=' * 60}
INPUT REQUIREMENTS SUMMARY:
{requirements[:500]}...

OUTPUTS:
1. ARCHITECTURE STYLE:
   - Modular multi-agent orchestration with clear role boundaries.
2. CORE COMPONENTS:
   - Orchestrator layer
   - Agent layer
   - Tooling/integration layer
   - Observability and logging layer
3. DATA FLOW:
   - Product brief -> requirements -> architecture -> implementation -> validation -> deployment -> monitoring.
4. NON-FUNCTIONALS:
   - Scalability, traceability, fault isolation, governance, and security.
5. TECHNICAL DECISIONS:
   - CrewAI for orchestration, OpenAI for reasoning, environment-based secret handling.
"""


class DevelopmentTool(BaseTool):
    name: str = "Development Tool"
    description: str = (
        "Use this tool to convert architecture into an implementation plan, code modules, interfaces, "
        "engineering tasks, and ML/AI integration steps."
    )

    def _run(self, architecture: str) -> str:
        logger.info("DevelopmentTool invoked.")
        return f"""
DEVELOPMENT ARTIFACT
{'=' * 60}
INPUT ARCHITECTURE SUMMARY:
{architecture[:500]}...

OUTPUTS:
1. IMPLEMENTATION PLAN:
   - Build orchestrator class
   - Implement agent factories
   - Define task pipeline
   - Add tools and environment configuration
2. CODE MODULES:
   - config.py / main.py / orchestrator.py / tools.py / prompts.py / tests/
3. ML / AI PLAN:
   - LLM configuration, prompt structure, guardrails, and token usage strategy.
4. ENGINEERING TASKS:
   - Code skeleton, integrations, tests, logging, exception handling.
5. DELIVERY NOTES:
   - Ensure maintainable abstractions and framework-agnostic design.
"""


class QualitySecurityTool(BaseTool):
    name: str = "Quality and Security Tool"
    description: str = (
        "Use this tool to review implementation artifacts for quality assurance, security, compliance, "
        "test coverage, and prompt-safety guardrails."
    )

    def _run(self, implementation_artifact: str) -> str:
        logger.info("QualitySecurityTool invoked.")
        return f"""
QA & SECURITY ARTIFACT
{'=' * 60}
INPUT IMPLEMENTATION SUMMARY:
{implementation_artifact[:500]}...

OUTPUTS:
1. QA CHECKS:
   - Unit test coverage required for orchestrator and tools.
   - Integration testing required for multi-agent task handoff.
2. SECURITY CHECKS:
   - API key handling via environment variables only.
   - Prompt-injection and unsafe tool usage checks required.
3. COMPLIANCE CHECKS:
   - Logging, audit trail, and traceability controls defined.
4. DEFECTS / GAPS:
   - Add validation for malformed inputs and tool failures.
5. RELEASE GATE:
   - Pass only after critical findings are closed or accepted.
"""


class DevOpsTool(BaseTool):
    name: str = "DevOps Tool"
    description: str = (
        "Use this tool to define CI/CD, environment promotion, deployment approvals, rollback plan, "
        "and operational readiness for the solution."
    )

    def _run(self, qa_security_artifact: str) -> str:
        logger.info("DevOpsTool invoked.")
        return f"""
DEVOPS ARTIFACT
{'=' * 60}
INPUT QA / SECURITY SUMMARY:
{qa_security_artifact[:500]}...

OUTPUTS:
1. CI/CD PLAN:
   - Lint, test, build, package, and deploy pipeline.
2. ENVIRONMENTS:
   - Dev -> QA -> UAT -> Production promotion workflow.
3. APPROVALS:
   - Security sign-off and release approval required before production.
4. ROLLBACK:
   - Versioned release with rollback and incident response checklist.
5. READINESS:
   - Logging, monitoring hooks, secrets handling, and deployment validation.
"""


class MonitoringTool(BaseTool):
    name: str = "Monitoring Tool"
    description: str = (
        "Use this tool to define KPIs, monitoring, drift detection, incident signals, and feedback loops "
        "back into the product backlog."
    )

    def _run(self, devops_artifact: str) -> str:
        logger.info("MonitoringTool invoked.")
        return f"""
MONITORING & FEEDBACK ARTIFACT
{'=' * 60}
INPUT DEVOPS SUMMARY:
{devops_artifact[:500]}...

OUTPUTS:
1. KPIs:
   - Task completion time
   - Failure rate
   - Escalation frequency
   - Deployment success rate
2. AI / AGENT METRICS:
   - Token usage, latency, hallucination incidents, tool success rate.
3. OPERATIONS:
   - Alerting thresholds and incident triage flow.
4. DRIFT / FEEDBACK:
   - Capture quality drift, prompt drift, and user feedback.
5. BACKLOG LOOP:
   - Feed defects, enhancement requests, and insights into next sprint planning.
"""


class SixAgentSDLCOrchestrator:
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.4,
        verbose: bool = True,
    ):
        self.verbose = verbose
        self.model_name = model_name
        self.temperature = temperature
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

        if not self.openai_api_key:
            raise ValueError(
                "OpenAI API key is required. Pass it as 'openai_api_key' parameter or set OPENAI_API_KEY."
            )

        os.environ["OPENAI_API_KEY"] = self.openai_api_key
        self._llm = self._initialize_llm()
        self._tools = self._initialize_tools()

    def _initialize_llm(self) -> ChatOpenAI:
        logger.info(f"Initializing ChatOpenAI model={self.model_name}, temperature={self.temperature}")
        return ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            api_key=self.openai_api_key,
        )

    def _initialize_tools(self) -> dict:
        return {
            "requirements": RequirementsTool(),
            "architecture": ArchitectureTool(),
            "development": DevelopmentTool(),
            "quality_security": QualitySecurityTool(),
            "devops": DevOpsTool(),
            "monitoring": MonitoringTool(),
        }

    def _create_business_analyst_agent(self) -> Agent:
        return Agent(
            role="Business Analyst",
            goal="Turn product vision into structured business requirements, user stories, and acceptance criteria.",
            backstory=(
                "You are a senior business analyst who specializes in translating ambiguous business intent "
                "into clear, testable delivery artifacts for engineering teams."
            ),
            tools=[self._tools["requirements"]],
            llm=self._llm,
            verbose=self.verbose,
            allow_delegation=False,
            max_iter=5,
        )

    def _create_architect_agent(self) -> Agent:
        return Agent(
            role="Solution Architect",
            goal="Design a modular, scalable, secure architecture from the approved requirements.",
            backstory=(
                "You are an enterprise solution architect experienced in modular systems, integrations, "
                "security boundaries, and multi-agent application design."
            ),
            tools=[self._tools["architecture"]],
            llm=self._llm,
            verbose=self.verbose,
            allow_delegation=False,
            max_iter=5,
        )

    def _create_developer_agent(self) -> Agent:
        return Agent(
            role="Developer and ML Engineer",
            goal="Convert architecture into implementation-ready engineering artifacts and delivery tasks.",
            backstory=(
                "You are a hands-on software developer and ML engineer who can structure codebases, "
                "define interfaces, and prepare AI-enabled systems for implementation."
            ),
            tools=[self._tools["development"]],
            llm=self._llm,
            verbose=self.verbose,
            allow_delegation=False,
            max_iter=5,
        )

    def _create_qa_security_agent(self) -> Agent:
        return Agent(
            role="QA and Security Engineer",
            goal="Validate implementation quality, risk posture, security controls, and release readiness.",
            backstory=(
                "You are a senior QA and security specialist who designs quality gates, threat checks, "
                "compliance controls, and prompt-safety reviews for AI-enabled systems."
            ),
            tools=[self._tools["quality_security"]],
            llm=self._llm,
            verbose=self.verbose,
            allow_delegation=False,
            max_iter=5,
        )

    def _create_devops_agent(self) -> Agent:
        return Agent(
            role="DevOps Engineer",
            goal="Create a release pipeline, environment promotion workflow, and operational deployment plan.",
            backstory=(
                "You are a DevOps engineer with experience in CI/CD, release automation, rollback strategy, "
                "environment hardening, and production deployment processes."
            ),
            tools=[self._tools["devops"]],
            llm=self._llm,
            verbose=self.verbose,
            allow_delegation=False,
            max_iter=5,
        )

    def _create_monitoring_agent(self) -> Agent:
        return Agent(
            role="Monitoring and Feedback Analyst",
            goal="Define KPIs, production monitoring, drift detection, and backlog feedback mechanisms.",
            backstory=(
                "You are an operations intelligence specialist focused on observability, KPIs, user feedback, "
                "quality trends, and continuous improvement loops."
            ),
            tools=[self._tools["monitoring"]],
            llm=self._llm,
            verbose=self.verbose,
            allow_delegation=False,
            max_iter=5,
        )

    def _create_requirements_task(self, product_brief: str, agent: Agent) -> Task:
        return Task(
            description=(
                f"Analyze the following product brief and produce business analysis artifacts.\n\n"
                f"PRODUCT BRIEF:\n{product_brief}\n\n"
                f"Deliverables:\n"
                f"1. Problem statement\n"
                f"2. Scope\n"
                f"3. User stories\n"
                f"4. Acceptance criteria\n"
                f"5. Risks, assumptions, and dependencies"
            ),
            expected_output="A structured business analysis document with scope, stories, and acceptance criteria.",
            agent=agent,
        )

    def _create_architecture_task(self, product_brief: str, agent: Agent) -> Task:
        return Task(
            description=(
                f"Design the solution architecture for the product brief below using the business analysis output as context.\n\n"
                f"PRODUCT BRIEF:\n{product_brief}\n\n"
                f"Deliverables:\n"
                f"1. Architecture style\n"
                f"2. Components and integrations\n"
                f"3. Data and control flow\n"
                f"4. Non-functional requirements\n"
                f"5. Technical decisions"
            ),
            expected_output="A modular solution architecture document with components, integration flow, and NFRs.",
            agent=agent,
        )

    def _create_development_task(self, product_brief: str, agent: Agent) -> Task:
        return Task(
            description=(
                f"Create an implementation plan for the following product brief using architecture context.\n\n"
                f"PRODUCT BRIEF:\n{product_brief}\n\n"
                f"Deliverables:\n"
                f"1. Engineering work breakdown\n"
                f"2. Code module plan\n"
                f"3. Interfaces and contracts\n"
                f"4. ML/AI integration notes\n"
                f"5. Delivery sequencing"
            ),
            expected_output="An implementation artifact with code structure, engineering tasks, and ML integration notes.",
            agent=agent,
        )

    def _create_quality_security_task(self, product_brief: str, agent: Agent) -> Task:
        return Task(
            description=(
                f"Review the implementation context for the following product brief and produce QA and security controls.\n\n"
                f"PRODUCT BRIEF:\n{product_brief}\n\n"
                f"Deliverables:\n"
                f"1. Test strategy\n"
                f"2. Security controls\n"
                f"3. Compliance checkpoints\n"
                f"4. Defect and risk log\n"
                f"5. Release gate decision"
            ),
            expected_output="A QA and security artifact covering test coverage, controls, defects, and release readiness.",
            agent=agent,
        )

    def _create_devops_task(self, product_brief: str, agent: Agent) -> Task:
        return Task(
            description=(
                f"Create the DevOps and release plan for the following product brief using QA and security context.\n\n"
                f"PRODUCT BRIEF:\n{product_brief}\n\n"
                f"Deliverables:\n"
                f"1. CI/CD pipeline\n"
                f"2. Environment promotion flow\n"
                f"3. Deployment approvals\n"
                f"4. Rollback strategy\n"
                f"5. Operational readiness checklist"
            ),
            expected_output="A DevOps release plan with CI/CD, environments, approvals, rollback, and readiness checklist.",
            agent=agent,
        )

    def _create_monitoring_task(self, product_brief: str, agent: Agent) -> Task:
        return Task(
            description=(
                f"Create a monitoring and feedback strategy for the following product brief using all prior SDLC outputs.\n\n"
                f"PRODUCT BRIEF:\n{product_brief}\n\n"
                f"Deliverables:\n"
                f"1. KPIs and SLIs\n"
                f"2. AI-specific monitoring metrics\n"
                f"3. Drift and anomaly signals\n"
                f"4. Feedback loop to backlog\n"
                f"5. Continuous improvement recommendations"
            ),
            expected_output="A monitoring strategy with KPIs, AI metrics, alerting, drift checks, and backlog feedback loop.",
            agent=agent,
        )

    def run(self, product_brief: str) -> str:
        logger.info("Starting 6-agent SDLC orchestration...")

        business_analyst = self._create_business_analyst_agent()
        architect = self._create_architect_agent()
        developer = self._create_developer_agent()
        qa_security = self._create_qa_security_agent()
        devops = self._create_devops_agent()
        monitoring = self._create_monitoring_agent()

        requirements_task = self._create_requirements_task(product_brief, business_analyst)
        architecture_task = self._create_architecture_task(product_brief, architect)
        development_task = self._create_development_task(product_brief, developer)
        quality_security_task = self._create_quality_security_task(product_brief, qa_security)
        devops_task = self._create_devops_task(product_brief, devops)
        monitoring_task = self._create_monitoring_task(product_brief, monitoring)

        architecture_task.context = [requirements_task]
        development_task.context = [requirements_task, architecture_task]
        quality_security_task.context = [requirements_task, architecture_task, development_task]
        devops_task.context = [requirements_task, architecture_task, development_task, quality_security_task]
        monitoring_task.context = [
            requirements_task,
            architecture_task,
            development_task,
            quality_security_task,
            devops_task,
        ]

        crew = Crew(
            agents=[business_analyst, architect, developer, qa_security, devops, monitoring],
            tasks=[
                requirements_task,
                architecture_task,
                development_task,
                quality_security_task,
                devops_task,
                monitoring_task,
            ],
            process=Process.sequential,
            verbose=self.verbose,
        )

        try:
            result = crew.kickoff()
            logger.info("6-agent SDLC orchestration completed successfully.")
            return str(result)
        except Exception as exc:
            logger.error(f"6-agent SDLC orchestration failed: {exc}", exc_info=True)
            raise


def main():
    load_dotenv()

    print("\n" + "=" * 78)
    print("   6-Agent SDLC Agentic AI Orchestrator | CrewAI + OpenAI")
    print("=" * 78)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = input("\nEnter your OpenAI API key: ").strip()
        if not api_key:
            print("ERROR: OpenAI API key is required.")
            return

    default_brief = (
        "Build an enterprise AI support platform that captures business requirements, "
        "designs a multi-agent solution, implements orchestrated workflows, validates "
        "quality and security, deploys through CI/CD, and monitors production KPIs."
    )

    product_brief = input(
        f"\nEnter product brief\n[Default: {default_brief}]\n> "
    ).strip()
    if not product_brief:
        product_brief = default_brief

    print(f"\n📋 Product Brief: {product_brief}")
    print("🤖 Agents: BA → Architect → Developer → QA/Security → DevOps → Monitoring")
    print("🔁 Process: Sequential")
    print("=" * 78)
    print("Starting crew...\n")

    orchestrator = SixAgentSDLCOrchestrator(
        openai_api_key=api_key,
        model_name="gpt-4o-mini",
        temperature=0.4,
        verbose=True,
    )

    result = orchestrator.run(product_brief=product_brief)

    print("\n" + "=" * 78)
    print("   FINAL SDLC OUTPUT")
    print("=" * 78)
    print(result)
    print("=" * 78)


if __name__ == "__main__":
    main()
