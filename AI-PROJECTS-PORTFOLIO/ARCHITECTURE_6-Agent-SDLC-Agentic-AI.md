# Architecture: 6-Agent SDLC Agentic AI Orchestrator

## Overview

The **6-Agent SDLC Agentic AI Orchestrator** is a CrewAI-based multi-agent system that models an AI-led software delivery lifecycle from business requirements to production monitoring. The architecture is designed to be **sequential, traceable, modular, and production-oriented**, allowing each SDLC phase to be handled by a dedicated AI agent with clear responsibility boundaries.

This project uses:
- **CrewAI** for agent orchestration
- **OpenAI models** via `langchain-openai`
- **Custom tools** for each SDLC phase
- **Sequential task chaining** for upstream-to-downstream context propagation

## Design Goals

The architecture is built around the following goals:

- **Role specialization** — each SDLC phase is mapped to a focused agent.
- **Traceability** — every phase produces an artifact that can be reviewed and passed downstream.
- **Modularity** — tools, prompts, tasks, and agents can be swapped independently.
- **Production readiness** — the pattern can be extended to enterprise systems, CI/CD, governance, and observability.
- **Framework flexibility** — the orchestration pattern is adaptable to CrewAI, LangGraph, Bedrock, and similar agent ecosystems.

## Agent Topology

The architecture contains six agents arranged in a strict sequential flow:

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

Each agent is backed by:
- A **role**
- A **goal**
- A **backstory**
- A **tool**
- A **task**
- CrewAI **LLM configuration**

## Agent Responsibilities

### 1. Business Analyst
**Purpose:** Convert product vision into structured business requirements.

**Outputs:**
- Problem statement
- Scope definition
- User stories
- Acceptance criteria
- Risks, assumptions, and dependencies

### 2. Solution Architect
**Purpose:** Design the modular system architecture using business analysis outputs.

**Outputs:**
- Architecture style
- Components and integrations
- Data flow and control flow
- Non-functional requirements
- Technical decisions

### 3. Developer & ML Engineer
**Purpose:** Translate architecture into implementation-ready engineering artifacts.

**Outputs:**
- Code module plan
- Engineering work breakdown
- Interface definitions
- AI/ML integration notes
- Delivery sequencing

### 4. QA & Security Engineer
**Purpose:** Validate quality, security, and compliance readiness.

**Outputs:**
- Test strategy
- Security controls
- Compliance checkpoints
- Defect and risk log
- Release gate status

### 5. DevOps Engineer
**Purpose:** Define CI/CD, release promotion, and deployment readiness.

**Outputs:**
- CI/CD pipeline plan
- Environment promotion flow
- Deployment approval steps
- Rollback strategy
- Operational readiness checklist

### 6. Monitoring & Feedback Analyst
**Purpose:** Close the loop with observability, KPIs, drift detection, and backlog feedback.

**Outputs:**
- KPIs and SLIs
- AI-specific monitoring metrics
- Alerting and incident signals
- Drift detection approach
- Backlog feedback recommendations

## Core Components

The architecture is composed of the following logical layers:

### 1. Orchestration Layer
This layer is implemented by the `SixAgentSDLCOrchestrator` class. It is responsible for:
- Initializing the OpenAI LLM
- Registering tools
- Creating agents
- Creating tasks
- Linking task context
- Assembling the CrewAI crew
- Running the workflow

### 2. Agent Layer
This layer contains the six specialized CrewAI agents. Each agent owns one SDLC stage and performs role-specific reasoning.

### 3. Tooling Layer
Each agent is paired with a dedicated custom tool:
- `RequirementsTool`
- `ArchitectureTool`
- `DevelopmentTool`
- `QualitySecurityTool`
- `DevOpsTool`
- `MonitoringTool`

These tools currently generate structured mock artifacts and can be replaced with:
- Enterprise APIs
- Search systems
- RAG pipelines
- Testing frameworks
- CI/CD platforms
- Monitoring systems

### 4. LLM Layer
All agents use a shared `ChatOpenAI` instance configured through `langchain-openai`. The model is initialized once and reused across the crew.

### 5. Execution Layer
Execution is handled by CrewAI using `Process.sequential`, ensuring ordered task execution and deterministic context flow.

## Context Flow

The system uses **task context chaining** to move outputs from one SDLC phase to the next.

### Context Mapping
- Architect receives output from Business Analyst
- Developer receives output from Business Analyst + Architect
- QA & Security receives output from Business Analyst + Architect + Developer
- DevOps receives output from all upstream build and validation stages
- Monitoring receives output from the full delivery chain

This design ensures each downstream phase has full awareness of upstream decisions.

## Execution Sequence

The workflow executes in the following order:

1. Product brief is provided as input.
2. Business Analyst produces requirements artifacts.
3. Solution Architect creates architecture artifacts.
4. Developer & ML Engineer creates implementation artifacts.
5. QA & Security validates release readiness.
6. DevOps defines deployment and release flow.
7. Monitoring defines KPIs, drift checks, and backlog feedback loops.
8. Final output is returned by the crew.

## Data and Artifact Model

The architecture is **artifact-driven**. Each stage produces structured textual outputs that act as SDLC deliverables. These artifacts can later be extended into:
- Markdown documents
- JSON outputs
- Jira tickets
- GitHub issues
- Confluence pages
- Audit records

This makes the architecture suitable for both demo and enterprise workflow adaptation.

## Configuration Model

The architecture uses environment-based configuration for secrets and runtime behavior.

### Required Configuration
- `OPENAI_API_KEY`

### Runtime Parameters
- `model_name`
- `temperature`
- `verbose`

This allows the same architecture to be reused across environments with minimal changes.

## Strengths of the Architecture

- **Clear ownership model** across SDLC phases
- **Structured agent collaboration** instead of a single overloaded LLM call
- **Simple extensibility** for tools and enterprise integrations
- **High interpretability** because every phase has explicit outputs
- **Good fit for AI governance** because decisions are broken into auditable stages

## Limitations

Current implementation limitations include:
- Tools return mock structured outputs rather than live system results
- Workflow is strictly sequential with no conditional branching
- No persistent artifact store or database
- No human-in-the-loop approval checkpoints yet
- No built-in memory across runs
- No live integration with GitHub, Jira, CI/CD, or monitoring platforms

## Extension Opportunities

The architecture can be extended in several ways:

### Conditional Routing
Introduce dynamic task branching based on risk, defects, or failed checks.

### Human Approval Gates
Add approval steps between Business Analyst, QA, Security, and Production release.

### Artifact Persistence
Store outputs in files, databases, or document systems for traceability.

### Enterprise Integrations
Connect with:
- GitHub
- Jira
- Confluence
- Jenkins / GitHub Actions
- Datadog / Grafana / CloudWatch
- Security scanning tools

### Feedback Loops
Turn Monitoring outputs into automatic triggers for backlog generation and iterative refinement.

## Reference Execution View

```text
Input Product Brief
        │
        ▼
Business Analyst Agent
        │
        ▼
Solution Architect Agent
        │
        ▼
Developer & ML Engineer Agent
        │
        ▼
QA & Security Agent
        │
        ▼
DevOps Agent
        │
        ▼
Monitoring & Feedback Agent
        │
        ▼
Final SDLC Output
```

## Summary

This architecture demonstrates how a **role-based AI crew** can model a realistic SDLC operating pattern. Rather than treating AI as isolated assistance, it structures AI as a coordinated delivery system with specialized ownership, context-aware handoffs, and phase-level artifacts. The result is a strong starting point for building **traceable, composable, and production-oriented agentic SDLC workflows**.
