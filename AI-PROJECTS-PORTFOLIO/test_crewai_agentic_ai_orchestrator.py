from __future__ import annotations

from pathlib import Path
import sys
import unittest
import uuid


PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from crewai_agentic_ai_orchestrator import (  # noqa: E402
    AIOrchestrator,
    ExecutionConfig,
    Phase,
    PhaseGateError,
    Risk,
    RiskStatus,
    Severity,
    ValidationError,
    build_sample_charter,
)


class AIOrchestratorTests(unittest.TestCase):
    def test_run_completes_and_closes_project(self) -> None:
        orchestrator = AIOrchestrator(
            charter=build_sample_charter(),
            config=ExecutionConfig(include_logs_in_summary=False),
        )

        final_state = orchestrator.run()

        self.assertEqual(final_state.current_phase.value, "closed")
        self.assertEqual(len(final_state.artifacts), 8)
        self.assertEqual(len(final_state.phase_history), 8)

    def test_invalid_charter_fails_fast(self) -> None:
        charter = build_sample_charter(project_name=" ")

        with self.assertRaises(ValidationError):
            AIOrchestrator(charter=charter)

    def test_high_open_risk_blocks_deployment(self) -> None:
        orchestrator = AIOrchestrator(charter=build_sample_charter())
        orchestrator.run_phase(Phase.REQUIREMENTS)
        orchestrator.run_phase(Phase.DESIGN)
        orchestrator.run_phase(Phase.DEVELOPMENT)
        orchestrator.run_phase(Phase.TESTING)
        orchestrator.run_phase(Phase.SECURITY)
        orchestrator.state.add_risk(
            Risk(
                title="Critical model exfiltration path",
                severity=Severity.HIGH,
                owner="security-review",
                mitigation="Block deployment until egress controls are enforced.",
                status=RiskStatus.OPEN,
            )
        )

        with self.assertRaises(PhaseGateError):
            orchestrator.run_phase(Phase.DEPLOYMENT)

    def test_output_directory_receives_final_summary(self) -> None:
        output_dir = PROJECT_DIR / f"test_output_{uuid.uuid4().hex[:8]}"
        orchestrator = AIOrchestrator(
            charter=build_sample_charter(),
            config=ExecutionConfig(output_dir=output_dir, include_logs_in_summary=False),
        )

        orchestrator.run()

        self.assertTrue((output_dir / "final_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
