#!/usr/bin/env python3
"""
Production-oriented AI SDLC orchestrator skeleton.

This module keeps the single-file footprint of the original demo while making
the runtime safer and more explicit for real delivery workflows:
- validates project input before execution
- models audit events, decisions, artifacts, and risks with typed records
- enforces phase-entry gates and deployment risk checks
- emits structured logs and optional JSON snapshots per phase
- keeps built-in agents clearly marked as simulated so demo output is not
  mistaken for live delivery telemetry

Replace the sample agents with real integrations as you wire this into CrewAI,
LangGraph, internal services, or your deployment platform.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence
import argparse
import json
import logging
import sys
import uuid


class OrchestratorError(Exception):
    """Base error for orchestrator failures."""


class ValidationError(OrchestratorError):
    """Raised when required input is missing or malformed."""


class PhaseGateError(OrchestratorError):
    """Raised when a phase starts without satisfying prerequisites."""


class AgentExecutionError(OrchestratorError):
    """Raised when an agent fails to complete safely."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def to_serializable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {key: to_serializable(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_serializable(item) for item in value]
    return value


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


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskStatus(str, Enum):
    OPEN = "open"
    MITIGATED = "mitigated"
    ACCEPTED = "accepted"
    CLOSED = "closed"


class EventLevel(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


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

    def validate(self) -> None:
        required_text = {
            "project_name": self.project_name,
            "business_goal": self.business_goal,
            "scope": self.scope,
            "timeline": self.timeline,
        }
        missing = [name for name, value in required_text.items() if not value.strip()]
        if missing:
            raise ValidationError(f"Missing required charter fields: {', '.join(missing)}")
        if not self.stakeholders:
            raise ValidationError("Charter must include at least one stakeholder.")
        if not self.success_metrics:
            raise ValidationError("Charter must include at least one success metric.")


@dataclass
class Artifact:
    name: str
    phase: Phase
    owner: str
    content: Dict[str, Any]
    status: Status = Status.DRAFT
    artifact_id: str = field(default_factory=lambda: new_id("artifact"))
    created_at: str = field(default_factory=utc_now)

    def validate(self) -> None:
        if not self.name.strip():
            raise ValidationError("Artifact name cannot be empty.")
        if not self.owner.strip():
            raise ValidationError(f"Artifact {self.name!r} must have an owner.")
        json.dumps(to_serializable(self.content))


@dataclass
class Risk:
    title: str
    severity: Severity
    owner: str
    mitigation: str
    status: RiskStatus = RiskStatus.OPEN
    risk_id: str = field(default_factory=lambda: new_id("risk"))
    created_at: str = field(default_factory=utc_now)

    def is_blocking(self) -> bool:
        return self.status == RiskStatus.OPEN and self.severity in {Severity.HIGH, Severity.CRITICAL}


@dataclass
class Decision:
    title: str
    owner: str
    rationale: str
    decision_id: str = field(default_factory=lambda: new_id("decision"))
    timestamp: str = field(default_factory=utc_now)


@dataclass
class AuditEvent:
    actor: str
    action: str
    details: Dict[str, Any]
    level: EventLevel = EventLevel.INFO
    event_id: str = field(default_factory=lambda: new_id("event"))
    timestamp: str = field(default_factory=utc_now)


@dataclass
class AgentResult:
    artifacts: List[Artifact] = field(default_factory=list)
    risks: List[Risk] = field(default_factory=list)
    decisions: List[Decision] = field(default_factory=list)
    backlog_items: List[Dict[str, Any]] = field(default_factory=list)
    deployment_events: List[Dict[str, Any]] = field(default_factory=list)
    monitoring_events: List[Dict[str, Any]] = field(default_factory=list)
    output: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionConfig:
    output_dir: Optional[Path] = None
    persist_phase_snapshots: bool = True
    include_logs_in_summary: bool = True
    strict_review_gates: bool = True
    fail_on_blocking_risks: bool = True
    simulated_agents: bool = True


@dataclass
class State:
    charter: ProjectCharter
    run_id: str = field(default_factory=lambda: new_id("run"))
    current_phase: Phase = Phase.INITIATION
    artifacts: List[Artifact] = field(default_factory=list)
    risks: List[Risk] = field(default_factory=list)
    backlog: List[Dict[str, Any]] = field(default_factory=list)
    decisions: List[Decision] = field(default_factory=list)
    logs: List[AuditEvent] = field(default_factory=list)
    deployment_events: List[Dict[str, Any]] = field(default_factory=list)
    monitoring_events: List[Dict[str, Any]] = field(default_factory=list)
    phase_history: List[Dict[str, Any]] = field(default_factory=list)

    def record_event(
        self,
        actor: str,
        action: str,
        details: Dict[str, Any],
        level: EventLevel = EventLevel.INFO,
    ) -> None:
        self.logs.append(AuditEvent(actor=actor, action=action, details=details, level=level))

    def add_artifact(self, artifact: Artifact) -> None:
        artifact.validate()
        self.artifacts.append(artifact)

    def add_risk(self, risk: Risk) -> None:
        self.risks.append(risk)

    def add_decision(self, decision: Decision) -> None:
        self.decisions.append(decision)

    def artifact_names(self) -> set[str]:
        return {artifact.name for artifact in self.artifacts}

    def find_artifact(self, name: str) -> Optional[Artifact]:
        for artifact in reversed(self.artifacts):
            if artifact.name == name:
                return artifact
        return None

    def blocking_risks(self) -> List[Risk]:
        return [risk for risk in self.risks if risk.is_blocking()]

    def to_dict(self, include_logs: bool = True) -> Dict[str, Any]:
        payload = {
            "run_id": self.run_id,
            "charter": to_serializable(self.charter),
            "current_phase": self.current_phase,
            "artifacts": self.artifacts,
            "risks": self.risks,
            "backlog": self.backlog,
            "decisions": self.decisions,
            "deployment_events": self.deployment_events,
            "monitoring_events": self.monitoring_events,
            "phase_history": self.phase_history,
        }
        if include_logs:
            payload["logs"] = self.logs
        return to_serializable(payload)


class Agent:
    name = "agent"
    role = "generic"

    def run(self, state: State) -> AgentResult:
        raise NotImplementedError


class BusinessAnalystAgent(Agent):
    name = "ba-agent"
    role = "Business Analyst"

    def run(self, state: State) -> AgentResult:
        stories = [
            {
                "id": "US-001",
                "title": "Provide traceable AI SDLC requirements with approval status and downstream mapping.",
                "priority": "high",
                "acceptance_criteria": [
                    "Requirements are versioned",
                    "Each requirement maps to design and delivery work",
                    "Approval status is externally visible",
                ],
            },
            {
                "id": "US-002",
                "title": "Provide operational monitoring and feedback loops after deployment.",
                "priority": "high",
                "acceptance_criteria": [
                    "Service KPIs are defined",
                    "Failure events are persisted",
                    "Improvement triggers are documented",
                ],
            },
        ]
        artifact = Artifact(
            name="requirements_spec",
            phase=Phase.REQUIREMENTS,
            owner=self.name,
            status=Status.APPROVED,
            content={
                "simulation_mode": True,
                "problem_statement": state.charter.business_goal,
                "scope": state.charter.scope,
                "stakeholders": state.charter.stakeholders,
                "constraints": state.charter.constraints,
                "success_metrics": state.charter.success_metrics,
                "user_stories": stories,
            },
        )
        return AgentResult(
            artifacts=[artifact],
            backlog_items=stories,
            output={"stories_generated": len(stories)},
        )


class SolutionArchitectAgent(Agent):
    name = "sa-agent"
    role = "Solution Architect"

    def run(self, state: State) -> AgentResult:
        architecture = Artifact(
            name="solution_design",
            phase=Phase.DESIGN,
            owner=self.name,
            status=Status.APPROVED,
            content={
                "simulation_mode": True,
                "architecture_style": "modular orchestrator with phase gates and audit trail",
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
                    "recoverable failures",
                    "human approvals before production",
                ],
                "handoffs": [
                    "requirements -> design",
                    "design -> development",
                    "development -> testing",
                    "testing + security -> deployment",
                    "monitoring -> improvement",
                ],
            },
        )
        decision = Decision(
            title="Use a gated orchestration model",
            owner=self.name,
            rationale="A gated model prevents silent promotion into deployment when required reviews or artifacts are missing.",
        )
        return AgentResult(
            artifacts=[architecture],
            decisions=[decision],
            output={"components": 6},
        )


class DeveloperAgent(Agent):
    name = "dev-agent"
    role = "Developer"

    def run(self, state: State) -> AgentResult:
        artifact = Artifact(
            name="development_plan",
            phase=Phase.DEVELOPMENT,
            owner=self.name,
            status=Status.REVIEWED,
            content={
                "simulation_mode": True,
                "modules": [
                    "orchestrator core",
                    "phase manager",
                    "artifact manager",
                    "status reporter",
                    "snapshot persistence",
                ],
                "implementation_status": "implementation_ready",
                "quality_practices": [
                    "typed models",
                    "centralized error handling",
                    "structured logging",
                    "phase gate enforcement",
                ],
            },
        )
        return AgentResult(artifacts=[artifact], output={"modules": 5})


class MLEngineerAgent(Agent):
    name = "mle-agent"
    role = "ML Engineer"

    def run(self, state: State) -> AgentResult:
        artifact = Artifact(
            name="ml_plan",
            phase=Phase.DEVELOPMENT,
            owner=self.name,
            status=Status.REVIEWED,
            content={
                "simulation_mode": True,
                "data_strategy": [
                    "dataset validation",
                    "evaluation dataset curation",
                    "feedback capture",
                ],
                "ml_patterns": [
                    "retrieval-augmented generation",
                    "tool-using agents",
                    "evaluation harness",
                ],
                "operational_metrics": [
                    "accuracy",
                    "groundedness",
                    "latency",
                    "cost_per_run",
                ],
            },
        )
        return AgentResult(artifacts=[artifact], output={"patterns": 3})


class QAAgent(Agent):
    name = "qa-agent"
    role = "QA Engineer"

    def run(self, state: State) -> AgentResult:
        artifact = Artifact(
            name="test_report",
            phase=Phase.TESTING,
            owner=self.name,
            status=Status.APPROVED,
            content={
                "simulation_mode": True,
                "test_types": ["unit", "integration", "workflow", "regression"],
                "passed": 28,
                "failed": 0,
                "coverage_percent": 88.0,
                "quality_gate": "pass",
            },
        )
        return AgentResult(artifacts=[artifact], output={"passed": 28, "failed": 0})


class SecurityAgent(Agent):
    name = "sec-agent"
    role = "Security and Compliance"

    def run(self, state: State) -> AgentResult:
        artifact = Artifact(
            name="security_review",
            phase=Phase.SECURITY,
            owner=self.name,
            status=Status.APPROVED,
            content={
                "simulation_mode": True,
                "security_checks": [
                    "input validation reviewed",
                    "tool access boundaries checked",
                    "audit logging enabled",
                    "secret handling reviewed",
                ],
                "compliance_checks": [
                    "approval gate defined",
                    "risk review recorded",
                    "human oversight checkpoint present",
                ],
                "status": "approved_with_follow_up",
            },
        )
        risk = Risk(
            title="Prompt injection against tool-enabled agents",
            severity=Severity.MEDIUM,
            owner=self.name,
            mitigation="Constrain tools, sanitize inputs, add policy checks, and isolate privileged actions.",
            status=RiskStatus.OPEN,
        )
        return AgentResult(artifacts=[artifact], risks=[risk], output={"status": "approved_with_follow_up"})


class DevOpsAgent(Agent):
    name = "cicd-agent"
    role = "DevOps"

    def run(self, state: State) -> AgentResult:
        event = {
            "timestamp": utc_now(),
            "environment": "staging",
            "status": "success",
            "simulation_mode": True,
        }
        artifact = Artifact(
            name="deployment_report",
            phase=Phase.DEPLOYMENT,
            owner=self.name,
            status=Status.APPROVED,
            content={
                "simulation_mode": True,
                "pipeline_stages": ["lint", "test", "package", "deploy"],
                "environments": ["dev", "qa", "staging", "prod"],
                "deployment_strategy": "staged rollout with manual promotion approval",
                "status": "staging_deployed",
            },
        )
        return AgentResult(artifacts=[artifact], deployment_events=[event], output={"environment": "staging"})


class MonitoringAgent(Agent):
    name = "ops-agent"
    role = "Monitoring"

    def run(self, state: State) -> AgentResult:
        metrics = {
            "availability_percent": 99.8,
            "avg_latency_ms": 740,
            "workflow_success_rate_percent": 96.4,
            "drift_status": "low",
            "recommended_action": "Continue monitoring and collect user feedback for the next optimization cycle.",
            "simulation_mode": True,
        }
        artifact = Artifact(
            name="monitoring_report",
            phase=Phase.MONITORING,
            owner=self.name,
            status=Status.REVIEWED,
            content=metrics,
        )
        return AgentResult(
            artifacts=[artifact],
            monitoring_events=[{"timestamp": utc_now(), "metrics": metrics}],
            output={"avg_latency_ms": metrics["avg_latency_ms"]},
        )


class PMAgent(Agent):
    name = "pm-agent"
    role = "Project Manager"

    def run(self, state: State) -> AgentResult:
        return AgentResult(
            output={
                "phase": state.current_phase.value,
                "artifact_count": len(state.artifacts),
                "open_risks": len([risk for risk in state.risks if risk.status == RiskStatus.OPEN]),
                "backlog_count": len(state.backlog),
            }
        )


class AIOrchestrator:
    PHASE_ORDER = [
        Phase.REQUIREMENTS,
        Phase.DESIGN,
        Phase.DEVELOPMENT,
        Phase.TESTING,
        Phase.SECURITY,
        Phase.DEPLOYMENT,
        Phase.MONITORING,
        Phase.IMPROVEMENT,
    ]

    PHASE_PREREQUISITES: Dict[Phase, List[str]] = {
        Phase.DESIGN: ["requirements_spec"],
        Phase.DEVELOPMENT: ["solution_design"],
        Phase.TESTING: ["development_plan", "ml_plan"],
        Phase.SECURITY: ["test_report"],
        Phase.DEPLOYMENT: ["test_report", "security_review"],
        Phase.MONITORING: ["deployment_report"],
    }

    def __init__(self, charter: ProjectCharter, config: Optional[ExecutionConfig] = None):
        charter.validate()
        self.config = config or ExecutionConfig()
        self.state = State(charter=charter)
        self.logger = logging.getLogger(self.__class__.__name__)
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
        self.state.record_event(
            "orchestrator",
            "run_initialized",
            {
                "run_id": self.state.run_id,
                "project_name": charter.project_name,
                "simulated_agents": self.config.simulated_agents,
            },
        )

    def validate_phase_entry(self, phase: Phase) -> None:
        required_artifacts = self.PHASE_PREREQUISITES.get(phase, [])
        available = self.state.artifact_names()
        missing = [artifact_name for artifact_name in required_artifacts if artifact_name not in available]
        if missing:
            raise PhaseGateError(
                f"Cannot enter phase {phase.value!r}; missing prerequisite artifacts: {', '.join(missing)}"
            )

        if phase == Phase.DEPLOYMENT:
            if self.config.strict_review_gates:
                required_status = {"test_report": Status.APPROVED, "security_review": Status.APPROVED}
                for artifact_name, expected_status in required_status.items():
                    artifact = self.state.find_artifact(artifact_name)
                    if artifact is None or artifact.status != expected_status:
                        raise PhaseGateError(
                            f"Cannot deploy; artifact {artifact_name!r} must be {expected_status.value}."
                        )
            if self.config.fail_on_blocking_risks:
                blocking = self.state.blocking_risks()
                if blocking:
                    titles = ", ".join(risk.title for risk in blocking)
                    raise PhaseGateError(f"Cannot deploy; blocking risks remain open: {titles}")

    def apply_result(self, agent: Agent, result: AgentResult) -> None:
        for artifact in result.artifacts:
            self.state.add_artifact(artifact)
        for risk in result.risks:
            self.state.add_risk(risk)
        for decision in result.decisions:
            self.state.add_decision(decision)
        self.state.backlog.extend(result.backlog_items)
        self.state.deployment_events.extend(result.deployment_events)
        self.state.monitoring_events.extend(result.monitoring_events)
        self.state.record_event(
            agent.name,
            "agent_completed",
            {
                "phase": self.state.current_phase.value,
                "artifacts": len(result.artifacts),
                "risks": len(result.risks),
                "decisions": len(result.decisions),
                "output": result.output,
            },
        )

    def persist_snapshot(self, phase: Phase) -> None:
        if not self.config.output_dir or not self.config.persist_phase_snapshots:
            return
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        snapshot_file = self.config.output_dir / f"{len(self.state.phase_history):02d}_{phase.value}.json"
        snapshot_file.write_text(
            json.dumps(self.state.to_dict(include_logs=self.config.include_logs_in_summary), indent=2),
            encoding="utf-8",
        )
        self.logger.info("Persisted snapshot for phase %s to %s", phase.value, snapshot_file)

    def run_phase(self, phase: Phase) -> None:
        self.validate_phase_entry(phase)
        self.state.current_phase = phase
        self.logger.info("Starting phase %s", phase.value)
        self.state.record_event("orchestrator", "phase_started", {"phase": phase.value})

        for agent in self.phase_agents.get(phase, []):
            try:
                result = agent.run(self.state)
            except Exception as exc:  # noqa: BLE001
                self.state.record_event(
                    agent.name,
                    "agent_failed",
                    {"phase": phase.value, "error": str(exc)},
                    level=EventLevel.ERROR,
                )
                raise AgentExecutionError(f"Agent {agent.name!r} failed in phase {phase.value!r}") from exc
            self.apply_result(agent, result)

        pm_result = self.pm.run(self.state)
        self.state.record_event(self.pm.name, "status_report", pm_result.output)
        self.state.phase_history.append(
            {
                "phase": phase.value,
                "completed_at": utc_now(),
                "artifacts_total": len(self.state.artifacts),
                "open_risks": len([risk for risk in self.state.risks if risk.status == RiskStatus.OPEN]),
            }
        )
        self.persist_snapshot(phase)

    def run_improvement_cycle(self) -> None:
        self.state.current_phase = Phase.IMPROVEMENT
        improvement_item = {
            "title": "Post-production optimization cycle",
            "inputs": ["monitoring metrics", "user feedback", "risk actions"],
            "outputs": ["updated backlog", "retraining candidates", "workflow refinements"],
            "simulation_mode": True,
        }
        self.state.backlog.append(improvement_item)
        self.state.add_decision(
            Decision(
                title="Start continuous improvement loop",
                owner="orchestrator",
                rationale="Monitoring has been established, so feedback should feed the next delivery cycle.",
            )
        )
        self.state.record_event("orchestrator", "improvement_cycle_started", improvement_item)
        pm_result = self.pm.run(self.state)
        self.state.record_event(self.pm.name, "status_report", pm_result.output)
        self.state.phase_history.append(
            {
                "phase": Phase.IMPROVEMENT.value,
                "completed_at": utc_now(),
                "artifacts_total": len(self.state.artifacts),
                "open_risks": len([risk for risk in self.state.risks if risk.status == RiskStatus.OPEN]),
            }
        )
        self.persist_snapshot(Phase.IMPROVEMENT)

    def run(self) -> State:
        for phase in self.PHASE_ORDER:
            if phase == Phase.IMPROVEMENT:
                self.run_improvement_cycle()
            else:
                self.run_phase(phase)
        self.state.current_phase = Phase.CLOSED
        self.state.record_event("orchestrator", "project_closed", {"status": "successful"})
        if self.config.output_dir:
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
            summary_file = self.config.output_dir / "final_summary.json"
            summary_file.write_text(
                json.dumps(self.summary(include_logs=self.config.include_logs_in_summary), indent=2),
                encoding="utf-8",
            )
            self.logger.info("Persisted final summary to %s", summary_file)
        return self.state

    def summary(self, include_logs: bool = True) -> Dict[str, Any]:
        return {
            "project": to_serializable(self.state.charter),
            "run_id": self.state.run_id,
            "current_phase": self.state.current_phase.value,
            "artifact_count": len(self.state.artifacts),
            "open_risks": len([risk for risk in self.state.risks if risk.status == RiskStatus.OPEN]),
            "blocking_risks": [to_serializable(risk) for risk in self.state.blocking_risks()],
            "simulated_agents": self.config.simulated_agents,
            "state": self.state.to_dict(include_logs=include_logs),
        }


def build_sample_charter(project_name: str = "Agentic AI SDLC Orchestrator") -> ProjectCharter:
    return ProjectCharter(
        project_name=project_name,
        business_goal=(
            "Manage an AI SDLC project from requirements through monitoring with clear governance, "
            "delivery traceability, and controlled deployment gates."
        ),
        scope="AI-enabled software delivery lifecycle orchestration for enterprise projects.",
        stakeholders=["Business Sponsor", "PMO", "Architecture", "Engineering", "Security", "Operations"],
        constraints=["Human approval before production", "Security review mandatory", "Traceability across phases"],
        success_metrics=[
            "95% requirement traceability",
            "85%+ automated test coverage",
            "Deployment approval enforced",
            "Continuous monitoring active",
        ],
        timeline="12-16 weeks",
        budget="TBD",
    )


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the production-oriented AI SDLC orchestrator.")
    parser.add_argument("--project-name", default="Agentic AI SDLC Orchestrator", help="Project name for the sample charter.")
    parser.add_argument("--output-dir", help="Optional directory for phase snapshots and final summary JSON.")
    parser.add_argument("--log-level", default="INFO", help="Logging level. Example: INFO, WARNING, DEBUG.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print the final JSON payload.")
    parser.add_argument("--exclude-logs", action="store_true", help="Omit audit logs from the final summary.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)

    config = ExecutionConfig(
        output_dir=Path(args.output_dir).resolve() if args.output_dir else None,
        include_logs_in_summary=not args.exclude_logs,
    )
    charter = build_sample_charter(project_name=args.project_name)

    try:
        orchestrator = AIOrchestrator(charter=charter, config=config)
        final_state = orchestrator.run()
        output = {
            "project_name": final_state.charter.project_name,
            "final_phase": final_state.current_phase.value,
            "artifact_count": len(final_state.artifacts),
            "open_risks": len([risk for risk in final_state.risks if risk.status == RiskStatus.OPEN]),
            "summary": orchestrator.summary(include_logs=config.include_logs_in_summary),
        }
        print(json.dumps(output, indent=2 if args.pretty else None))
        return 0
    except OrchestratorError as exc:
        logging.getLogger("main").error("Orchestrator failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
