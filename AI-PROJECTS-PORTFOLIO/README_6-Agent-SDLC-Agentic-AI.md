6-Agent SDLC Agentic AI Orchestrator with CrewAI
A production-style multi-agent SDLC orchestrator built with CrewAI and OpenAI. This project models an AI-led software delivery lifecycle using six specialized agents that work sequentially from requirements to monitoring.
Quick Notes on CrewAI
CrewAI is a multi-agent orchestration framework for defining specialized AI agents, assigning them roles, goals, tools, and tasks, and coordinating them in structured workflows. In this project, CrewAI is used to model an AI-led SDLC where each software delivery phase is handled by a dedicated agent with clear responsibility boundaries. The crew runs with a sequential process, so each downstream agent receives upstream outputs as context.
Agents
This orchestrator includes the following six agents:
Business Analyst — turns product vision into scope, user stories, and acceptance criteria.
Solution Architect — designs the modular architecture, components, and integration flow.
Developer & ML Engineer — creates the implementation plan, code structure, and AI integration notes.
QA & Security Engineer — defines test strategy, release gates, security checks, and compliance controls.
DevOps Engineer — creates CI/CD flow, deployment plan, rollback strategy, and environment promotion path.
Monitoring & Feedback Analyst — defines KPIs, drift detection, observability, and feedback loops back into the backlog.
Workflow
The orchestrator follows this SDLC sequence:
```text
Business Analyst
   ↓
Solution Architect
   ↓
Developer & ML Engineer
   ↓
QA & Security Engineer
   ↓
DevOps Engineer
   ↓
Monitoring & Feedback Analyst
```
Each task output becomes context for the next task, creating a traceable end-to-end delivery flow.
Features
6-agent SDLC workflow using CrewAI
OpenAI model integration using `langchain-openai`
Custom tools for each SDLC phase
Sequential task orchestration with context chaining
CLI entry point for quick execution
Environment variable support using `.env`
Modular structure that can be adapted for real enterprise delivery pipelines
Project File
Main Python file:
```bash
agentic_ai_sdlc_orchestrator_6_agents.py
```
Installation
Install the required dependencies:
```bash
pip install crewai crewai-tools langchain-openai python-dotenv
```
Configuration
Create a `.env` file in the project directory:
```env
OPENAI_API_KEY=sk-your-openai-api-key
```
Or export the key directly:
```bash
export OPENAI_API_KEY="sk-your-openai-api-key"
```
Usage
Run the orchestrator:
```bash
python agentic_ai_sdlc_orchestrator_6_agents.py
```
You will be prompted to enter:
OpenAI API key (if not already set)
Product brief
Example product brief:
```text
Build an enterprise AI support platform that captures business requirements,
designs a multi-agent solution, implements orchestrated workflows, validates
quality and security, deploys through CI/CD, and monitors production KPIs.
```
Example Output Stages
The orchestrator generates structured outputs for each SDLC phase:
Business analysis artifact
Solution architecture artifact
Development artifact
QA & security artifact
DevOps artifact
Monitoring & feedback artifact
Architecture Highlights
Traceable — every phase produces an artifact.
Composable — agents are swappable and independently extensible.
Production-oriented — can be adapted to CrewAI, LangGraph, Bedrock, or enterprise agent stacks.
Framework-friendly — easy to integrate with observability, CI/CD, testing, and governance workflows.
Customization Ideas
You can extend this project by:
Replacing mock tools with real APIs or enterprise systems
Adding persistence for artifacts and audit logs
Integrating Jira, GitHub, Confluence, or CI/CD tools
Adding prompt-safety checks and policy enforcement
Converting the sequential flow into conditional or feedback-driven routing
Adding human-in-the-loop approvals between phases
Example Python Usage
```python
from agentic_ai_sdlc_orchestrator_6_agents import SixAgentSDLCOrchestrator

orchestrator = SixAgentSDLCOrchestrator(
    openai_api_key="sk-your-openai-api-key",
    model_name="gpt-4o-mini",
    temperature=0.4,
    verbose=True,
)

result = orchestrator.run(
    product_brief="Build a regulated enterprise onboarding platform with AI-led SDLC automation."
)

print(result)
```
Why This Project Matters
Most teams still use AI as isolated assistance inside the SDLC. This project demonstrates a more structured pattern: a crew of role-based AI agents collaborating across the full software lifecycle. It is designed to show how Agentic AI can evolve from experimentation into a practical operating model for software delivery.
License
Add your preferred license here, such as MIT or Apache-2.0.
