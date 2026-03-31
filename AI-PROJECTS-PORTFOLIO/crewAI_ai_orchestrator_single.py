#!/usr/bin/env python3
"""
Single-file AI Orchestrator for a full AI SDLC project.

What this script does:
- models the AI SDLC lifecycle end to end
- coordinates specialized agents for each phase
- tracks artifacts, risks, decisions, and execution logs
- can run standalone without external frameworks
- can later be adapted to CrewAI, LangGraph, or other agent stacks

Run:
    python ai_orchestrator_single.py
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import json


class Phase(str, Enum):
    INITIATION = "initiation"
    REQUIREMENTS = "requirements"
    DESIGN = "design"
    DEVELOPMENT = "development"
    TESTING = "testing"
    SECURITY = "security"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"
    IMPROVEMENT = "improvement"
    CLOSED = "closed"


class Status(str, Enum):
    DRAFT = "draft"
    REVIEWED = "reviewed"
    APPROVED = "approved"
    BLOCKED = "blocked"


@dataclass
class ProjectCharter:
    project_name: str
    business_goal: str
    scope: str
    stakeholders: List[str]
    constraints: List[str]
    success_metrics: List[str]
    timeline: str
    budget: Optional[str] = None


@dataclass
class Artifact:
    name: str
    phase: Phase
    owner: str
    content: Dict[str, Any]
    status: Status = Status.DRAFT
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class Risk:
    title: str
    severity: str
    owner: str
    mitigation: str
    status: str = "open"


@dataclass
class State:
    charter: ProjectCharter
    current_phase: Phase = Phase.INITIATION
    artifacts: List[Artifact] = field(default_factory=list)
    risks: List[Risk] = field(default_factory=list)
    backlog: List[Dict[str, Any]] = field(default_factory=list)
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    deployment_events: List[Dict[str, Any]] = field(default_factory=list)
    monitoring_events: List[Dict[str, Any]] = field(default_factory=list)

    def log(self, actor: str, action: str, details: Dict[str, Any]) -> None:
        self.logs.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "actor": actor,
                "action": action,
                "details": details,
            }
        )

    def add_artifact(self, artifact: Artifact) -> None:
        self.artifacts.append(artifact)

    def add_risk(self, risk: Risk) -> None:
        self.risks.append(risk)


class Agent:
    name = "Agent"
    role = "Generic"

    def run(self, state: State) -> Dict[str, Any]:
        raise NotImplementedError


class BusinessAnalystAgent(Agent):
    name = "BA-Agent"
    role = "Business Analyst"

    def run(self, state: State) -> Dict[str, Any]:
        stories = [
            {
                "id": "US-001",
                "title": "As a sponsor, I need traceable AI SDLC requirements so governance and delivery remain aligned.",
                "priority": "high",
                "acceptance_criteria": [
                    "Requirements are documented",
                    "Each requirement maps to design and build work",
                    "Approval status is visible",
                ],
            },
            {
                "id": "US-002",
                "title": "As an operations lead, I need monitoring and feedback loops so the AI system improves after deployment.",
                "priority": "high",
                "acceptance_criteria": [
                    "KPIs are defined",
                    "Failure events are logged",
                    "Improvement triggers are documented",
                ],
            },
        ]
        result = {
            "problem_statement": state.charter.business_goal,
            "scope": state.charter.scope,
            "stakeholders": state.charter.stakeholders,
            "constraints": state.charter.constraints,
            "success_metrics": state.charter.success_metrics,
            "user_stories": stories,
        }
        state.backlog.extend(stories)
        state.add_artifact(
            Artifact("requirements_spec", Phase.REQUIREMENTS, self.name, result, Status.APPROVED)
        )
        state.log(self.name, "requirements_generated", {"stories": len(stories)})
        return result


class SolutionArchitectAgent(Agent):
    name = "SA-Agent"
    role = "Solution Architect"

    def run(self, state: State) -> Dict[str, Any]:
        result = {
            "architecture_style": "modular multi-agent orchestrator",
            "components": [
                "orchestrator engine",
                "agent registry",
                "artifact store",
                "risk register",
                "deployment controller",
                "monitoring pipeline",
            ],
            "non_functional_requirements": [
                "traceability",
                "auditability",
                "security by design",
                "human approval gates",
            ],
            "handoffs": [
                "requirements -> design",
                "design -> development",
                "development -> testing",
                "testing/security -> deployment",
                "monitoring -> improvement backlog",
            ],
        }
        state.add_artifact(
            Artifact("solution_design", Phase.DESIGN, self.name, result, Status.APPROVED)
        )
        state.decisions.append(
            {
                "title": "Use modular orchestration with phase gates",
                "owner": self.name,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
        state.log(self.name, "design_completed", {"components": len(result["components"])})
        return result


class DeveloperAgent(Agent):
    name = "DEV-Agent"
    role = "Developer"

    def run(self, state: State) -> Dict[str, Any]:
        result = {
            "modules": [
                "orchestrator core",
                "phase manager",
                "artifact manager",
                "status reporter",
            ],
            "implementation_status": "prototype_ready",
            "code_quality_practices": [
                "modular classes",
                "typed data models",
                "structured logs",
            ],
        }
        state.add_artifact(
            Artifact("development_plan", Phase.DEVELOPMENT, self.name, result, Status.REVIEWED)
        )
        state.log(self.name, "development_completed", {"modules": len(result["modules"])})
        return result


class MLEngineerAgent(Agent):
    name = "MLE-Agent"
    role = "ML Engineer"

    def run(self, state: State) -> Dict[str, Any]:
        result = {
            "data_strategy": [
                "dataset validation",
                "prompt/eval dataset curation",
                "feedback capture",
            ],
            "ml_patterns": [
                "RAG",
                "tool-using agents",
                "evaluation harness",
            ],
            "operational_metrics": [
                "accuracy",
                "groundedness",
                "latency",
                "cost per run",
            ],
        }
        state.add_artifact(
            Artifact("ml_plan", Phase.DEVELOPMENT, self.name, result, Status.REVIEWED)
        )
        state.log(self.name, "ml_plan_completed", {"patterns": len(result["ml_patterns"])})
        return result


class QAAgent(Agent):
    name = "QA-Agent"
    role = "QA Engineer"

    def run(self, state: State) -> Dict[str, Any]:
        result = {
            "test_types": ["unit", "integration", "workflow", "regression"],
            "passed": 28,
            "failed": 0,
            "coverage": "88%",
            "quality_gate": "pass",
        }
        state.add_artifact(
            Artifact("test_report", Phase.TESTING, self.name, result, Status.APPROVED)
        )
        state.log(self.name, "testing_completed", result)
        return result


class SecurityAgent(Agent):
    name = "SEC-Agent"
    role = "Security and Compliance"

    def run(self, state: State) -> Dict[str, Any]:
        result = {
            "security_checks": [
                "input validation reviewed",
                "tool access boundaries checked",
                "audit logging enabled",
                "secrets handling reviewed",
            ],
            "compliance_checks": [
                "approval gate defined",
                "risk review recorded",
                "human oversight checkpoint present",
            ],
            "status": "approved_with_actions",
        }
        state.add_risk(
            Risk(
                title="Prompt injection against tool-enabled agents",
                severity="medium",
                owner=self.name,
                mitigation="Constrain tools, sanitize inputs, add policy checks",
            )
        )
        state.add_artifact(
            Artifact("security_review", Phase.SECURITY, self.name, result, Status.REVIEWED)
        )
        state.log(self.name, "security_completed", {"status": result["status"]})
        return result


class DevOpsAgent(Agent):
    name = "CICD-Agent"
    role = "DevOps"

    def run(self, state: State) -> Dict[str, Any]:
        result = {
            "pipeline_stages": ["lint", "test", "package", "deploy"],
            "environments": ["dev", "qa", "staging", "prod"],
            "deployment_strategy": "staged rollout with manual approval",
            "status": "staging_deployed",
        }
        state.deployment_events.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "environment": "staging",
                "status": "success",
            }
        )
        state.add_artifact(
            Artifact("deployment_report", Phase.DEPLOYMENT, self.name, result, Status.APPROVED)
        )
        state.log(self.name, "deployment_completed", {"environment": "staging"})
        return result


class MonitoringAgent(Agent):
    name = "OPS-Agent"
    role = "Monitoring"

    def run(self, state: State) -> Dict[str, Any]:
        result = {
            "availability": "99.8%",
            "avg_latency_ms": 740,
            "workflow_success_rate": "96.4%",
            "drift_status": "low",
            "recommended_action": "continue monitoring and collect user feedback",
        }
        state.monitoring_events.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "metrics": result,
            }
        )
        state.add_artifact(
            Artifact("monitoring_report", Phase.MONITORING, self.name, result, Status.REVIEWED)
        )
        state.log(self.name, "monitoring_completed", result)
        return result


class PMAgent(Agent):
    name = "PM-Agent"
    role = "Project Manager"

    def run(self, state: State) -> Dict[str, Any]:
        result = {
            "phase": state.current_phase.value,
            "artifact_count": len(state.artifacts),
            "open_risks": len([r for r in state.risks if r.status == "open"]),
            "backlog_count": len(state.backlog),
        }
        state.log(self.name, "status_report", result)
        return result


class AIOrchestrator:
    def __init__(self, charter: ProjectCharter):
        self.state = State(charter=charter)
        self.pm = PMAgent()
        self.phase_agents: Dict[Phase, List[Agent]] = {
            Phase.REQUIREMENTS: [BusinessAnalystAgent()],
            Phase.DESIGN: [SolutionArchitectAgent()],
            Phase.DEVELOPMENT: [DeveloperAgent(), MLEngineerAgent()],
            Phase.TESTING: [QAAgent()],
            Phase.SECURITY: [SecurityAgent()],
            Phase.DEPLOYMENT: [DevOpsAgent()],
            Phase.MONITORING: [MonitoringAgent()],
        }

    def run_phase(self, phase: Phase) -> None:
        self.state.current_phase = phase
        for agent in self.phase_agents.get(phase, []):
            agent.run(self.state)
        self.pm.run(self.state)

    def run(self) -> State:
        ordered = [
            Phase.REQUIREMENTS,
            Phase.DESIGN,
            Phase.DEVELOPMENT,
            Phase.TESTING,
            Phase.SECURITY,
            Phase.DEPLOYMENT,
            Phase.MONITORING,
            Phase.IMPROVEMENT,
        ]
        for phase in ordered:
            if phase == Phase.IMPROVEMENT:
                self.state.current_phase = phase
                improvement_item = {
                    "title": "Post-production optimization cycle",
                    "inputs": ["monitoring metrics", "user feedback", "risk actions"],
                    "outputs": ["updated backlog", "retraining candidates", "workflow refinements"],
                }
                self.state.decisions.append(
                    {
                        "title": "Start continuous improvement loop",
                        "owner": "ORCHESTRATOR",
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )
                self.state.backlog.append(improvement_item)
                self.state.log("ORCHESTRATOR", "improvement_cycle_started", improvement_item)
                self.pm.run(self.state)
            else:
                self.run_phase(phase)
        self.state.current_phase = Phase.CLOSED
        self.state.log("ORCHESTRATOR", "project_closed", {"status": "successful"})
        return self.state

    def summary(self) -> Dict[str, Any]:
        return {
            "project": asdict(self.state.charter),
            "current_phase": self.state.current_phase.value,
            "artifacts": [
                {
                    "name": a.name,
                    "phase": a.phase.value,
                    "owner": a.owner,
                    "status": a.status.value,
                }
                for a in self.state.artifacts
            ],
            "risks": [asdict(r) for r in self.state.risks],
            "decisions": self.state.decisions,
            "deployment_events": self.state.deployment_events,
            "monitoring_events": self.state.monitoring_events,
            "logs": self.state.logs,
        }


def build_sample_charter() -> ProjectCharter:
    return ProjectCharter(
        project_name="Agentic AI SDLC Orchestrator",
        business_goal="Manage a full AI SDLC project from requirements through monitoring with clear governance and delivery traceability.",
        scope="AI-enabled software delivery lifecycle orchestration for enterprise projects.",
        stakeholders=["Business Sponsor", "PMO", "Architecture", "Engineering", "Security", "Operations"],
        constraints=["Human approval before production", "Security review mandatory", "Traceability across phases"],
        success_metrics=[
            "95% requirement traceability",
            "85%+ test coverage",
            "Deployment approval enforced",
            "Continuous monitoring active",
        ],
        timeline="12-16 weeks",
        budget="TBD",
    )


def main() -> None:
    charter = build_sample_charter()
    orchestrator = AIOrchestrator(charter)
    final_state = orchestrator.run()
    output = {
        "project_name": final_state.charter.project_name,
        "final_phase": final_state.current_phase.value,
        "artifact_count": len(final_state.artifacts),
        "open_risks": len([r for r in final_state.risks if r.status == "open"]),
        "summary": orchestrator.summary(),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
