"""Tests for consensus engine."""

import pytest

from cosilico_validators import (
    BaseValidator,
    ConsensusEngine,
    ConsensusLevel,
    TestCase,
    ValidatorResult,
    ValidatorType,
)


class MockValidator(BaseValidator):
    """Mock validator for testing."""

    def __init__(
        self,
        name: str,
        validator_type: ValidatorType,
        return_value: float | None,
        error: str | None = None,
    ):
        self.name = name
        self.validator_type = validator_type
        self._return_value = return_value
        self._error = error
        self.supported_variables = {"eitc", "ctc", "income_tax"}

    def supports_variable(self, variable: str) -> bool:
        return variable.lower() in self.supported_variables

    def validate(self, test_case: TestCase, variable: str, year: int = 2024) -> ValidatorResult:
        return ValidatorResult(
            validator_name=self.name,
            validator_type=self.validator_type,
            calculated_value=self._return_value,
            error=self._error,
        )


@pytest.fixture
def simple_test_case():
    return TestCase(
        name="EITC basic test",
        inputs={"earned_income": 15000, "filing_status": "SINGLE"},
        expected={"eitc": 600},
        citation="26 USC § 32",
    )


class TestConsensusEngine:
    def test_full_agreement(self, simple_test_case):
        """All validators agree within tolerance."""
        validators = [
            MockValidator("V1", ValidatorType.PRIMARY, 600),
            MockValidator("V2", ValidatorType.REFERENCE, 605),
            MockValidator("V3", ValidatorType.REFERENCE, 598),
        ]
        engine = ConsensusEngine(validators, tolerance=15.0)
        result = engine.validate(simple_test_case, "eitc", 2024)

        assert result.consensus_level == ConsensusLevel.FULL_AGREEMENT
        assert result.consensus_value is not None
        assert abs(result.consensus_value - 601) < 1  # mean of 600, 605, 598
        assert result.reward_signal > 0.5  # High reward for full agreement

    def test_primary_confirmed(self, simple_test_case):
        """Primary validator + majority agree."""
        validators = [
            MockValidator("Primary", ValidatorType.PRIMARY, 600),
            MockValidator("V2", ValidatorType.REFERENCE, 605),
            MockValidator("V3", ValidatorType.REFERENCE, 800),  # Outlier
        ]
        engine = ConsensusEngine(validators, tolerance=15.0)
        result = engine.validate(simple_test_case, "eitc", 2024)

        assert result.consensus_level == ConsensusLevel.PRIMARY_CONFIRMED
        assert result.consensus_value == 600
        assert result.reward_signal > 0.4

    def test_disagreement(self, simple_test_case):
        """No consensus when validators wildly disagree."""
        validators = [
            MockValidator("V1", ValidatorType.REFERENCE, 100),
            MockValidator("V2", ValidatorType.REFERENCE, 500),
            MockValidator("V3", ValidatorType.REFERENCE, 900),
        ]
        engine = ConsensusEngine(validators, tolerance=15.0)
        result = engine.validate(simple_test_case, "eitc", 2024)

        assert result.consensus_level == ConsensusLevel.DISAGREEMENT
        assert result.reward_signal < 0  # Negative reward for disagreement

    def test_potential_upstream_bug(self, simple_test_case):
        """Claude confident but validators disagree with expected."""
        validators = [
            MockValidator("V1", ValidatorType.REFERENCE, 800),  # Different from expected 600
            MockValidator("V2", ValidatorType.REFERENCE, 850),
        ]
        engine = ConsensusEngine(validators, tolerance=15.0)
        result = engine.validate(simple_test_case, "eitc", 2024, claude_confidence=0.95)

        assert result.consensus_level == ConsensusLevel.POTENTIAL_UPSTREAM_BUG
        assert len(result.potential_bugs) > 0
        assert result.potential_bugs[0]["expected"] == 600
        assert result.potential_bugs[0]["actual"] in [800, 850]

    def test_no_validators_succeed(self, simple_test_case):
        """All validators fail."""
        validators = [
            MockValidator("V1", ValidatorType.REFERENCE, None, error="Failed"),
            MockValidator("V2", ValidatorType.REFERENCE, None, error="Also failed"),
        ]
        engine = ConsensusEngine(validators, tolerance=15.0)
        result = engine.validate(simple_test_case, "eitc", 2024)

        assert result.consensus_level == ConsensusLevel.DISAGREEMENT
        assert result.consensus_value is None
        assert result.confidence == 0.0

    def test_reward_signal_bounds(self, simple_test_case):
        """Reward signal stays within [-1, 1]."""
        validators = [
            MockValidator("V1", ValidatorType.PRIMARY, 600),
            MockValidator("V2", ValidatorType.REFERENCE, 600),
            MockValidator("V3", ValidatorType.REFERENCE, 600),
        ]
        engine = ConsensusEngine(validators, tolerance=15.0)
        result = engine.validate(simple_test_case, "eitc", 2024)

        assert -1.0 <= result.reward_signal <= 1.0

    def test_confidence_calculation(self, simple_test_case):
        """Confidence reflects success rate and agreement."""
        validators = [
            MockValidator("V1", ValidatorType.PRIMARY, 600),
            MockValidator("V2", ValidatorType.REFERENCE, 600),
            MockValidator("V3", ValidatorType.REFERENCE, None, error="Failed"),
        ]
        engine = ConsensusEngine(validators, tolerance=15.0)
        result = engine.validate(simple_test_case, "eitc", 2024)

        assert 0.0 <= result.confidence <= 1.0
        # 2/3 success rate + agreement + primary bonus
        assert result.confidence > 0.5

    def test_batch_validate(self, simple_test_case):
        """Batch validation works correctly."""
        validators = [MockValidator("V1", ValidatorType.REFERENCE, 600)]
        engine = ConsensusEngine(validators)

        test_cases = [simple_test_case, simple_test_case]
        results = engine.batch_validate(test_cases, "eitc", 2024)

        assert len(results) == 2
        for r in results:
            assert r.consensus_value is not None


class TestValidationResult:
    def test_matches_expected_within_tolerance(self, simple_test_case):
        """Result matches when within $15 tolerance."""
        validators = [MockValidator("V1", ValidatorType.REFERENCE, 610)]
        engine = ConsensusEngine(validators)
        result = engine.validate(simple_test_case, "eitc", 2024)

        assert result.matches_expected  # 610 vs 600, diff = 10 < 15

    def test_does_not_match_outside_tolerance(self, simple_test_case):
        """Result doesn't match when outside tolerance."""
        validators = [MockValidator("V1", ValidatorType.REFERENCE, 620)]
        engine = ConsensusEngine(validators)
        result = engine.validate(simple_test_case, "eitc", 2024)

        assert not result.matches_expected  # 620 vs 600, diff = 20 > 15

    def test_summary_generation(self, simple_test_case):
        """Summary string is generated correctly."""
        validators = [MockValidator("V1", ValidatorType.REFERENCE, 600)]
        engine = ConsensusEngine(validators)
        result = engine.validate(simple_test_case, "eitc", 2024)

        summary = result.summary()
        assert "EITC basic test" in summary
        assert "$600" in summary
        assert "Reward:" in summary

    def test_matches_expected_none_consensus(self, simple_test_case):
        """matches_expected returns False when consensus_value is None."""
        validators = [
            MockValidator("V1", ValidatorType.REFERENCE, None, error="Failed"),
        ]
        engine = ConsensusEngine(validators)
        result = engine.validate(simple_test_case, "eitc", 2024)
        assert not result.matches_expected

    def test_summary_with_potential_bugs(self, simple_test_case):
        """Summary includes bug count when potential_bugs exist."""
        validators = [
            MockValidator("V1", ValidatorType.REFERENCE, 800),
            MockValidator("V2", ValidatorType.REFERENCE, 850),
        ]
        engine = ConsensusEngine(validators, tolerance=15.0)
        result = engine.validate(simple_test_case, "eitc", 2024, claude_confidence=0.95)
        summary = result.summary()
        assert "Potential bugs:" in summary


class TestConsensusRewardConfidenceEdgeCases:
    def test_compute_reward_empty_results(self):
        """Test _compute_reward with empty results dict returns 0.0."""
        engine = ConsensusEngine(
            validators=[
                MockValidator("V1", ValidatorType.REFERENCE, 500),
            ]
        )
        reward = engine._compute_reward({}, expected=500.0, consensus_level=ConsensusLevel.DISAGREEMENT)
        assert reward == 0.0

    def test_compute_confidence_no_successful_results(self):
        """Test _compute_confidence with consensus_value but no successful validators."""
        engine = ConsensusEngine(
            validators=[
                MockValidator("V1", ValidatorType.REFERENCE, 500),
            ]
        )
        failed_result = ValidatorResult(
            validator_name="test",
            validator_type=ValidatorType.REFERENCE,
            calculated_value=None,
            error="Failed",
        )
        assert not failed_result.success
        confidence = engine._compute_confidence({"test": failed_result}, consensus_value=500.0)
        assert confidence == 0.0


class TestConsensusEdgeCases:
    def test_expected_from_first_value(self):
        """When variable name doesn't match expected keys, use first value."""
        tc = TestCase(
            name="test",
            inputs={"earned_income": 15000},
            expected={"income_tax": 500},  # Key doesn't match "eitc"
        )
        validators = [MockValidator("V1", ValidatorType.REFERENCE, 500)]
        engine = ConsensusEngine(validators)
        result = engine.validate(tc, "eitc", 2024)
        # Should use first expected value (500) since "eitc" not in expected
        assert result.expected_value == 500

    def test_expected_empty(self):
        """When expected is empty, default to 0."""
        tc = TestCase(
            name="test",
            inputs={"earned_income": 15000},
            expected={},
        )
        validators = [MockValidator("V1", ValidatorType.REFERENCE, 0)]
        engine = ConsensusEngine(validators)
        result = engine.validate(tc, "eitc", 2024)
        assert result.expected_value == 0

    def test_majority_agreement(self):
        """Test majority agreement path (not full, not primary confirmed)."""
        tc = TestCase(
            name="test",
            inputs={"earned_income": 15000},
            expected={"eitc": 600},
        )
        validators = [
            MockValidator("V1", ValidatorType.REFERENCE, 700),
            MockValidator("V2", ValidatorType.REFERENCE, 705),
            MockValidator("V3", ValidatorType.REFERENCE, 1000),  # outlier
        ]
        engine = ConsensusEngine(validators, tolerance=15.0)
        result = engine.validate(tc, "eitc", 2024)
        # V1 and V2 agree (within 15), V3 is outlier
        # None is PRIMARY, so skip PRIMARY_CONFIRMED
        # Cluster of [700, 705] is majority (2/3)
        assert result.consensus_level == ConsensusLevel.MAJORITY_AGREEMENT

    def test_detect_potential_bugs_low_confidence(self):
        """No bugs detected when claude_confidence is low."""
        tc = TestCase(
            name="test",
            inputs={"earned_income": 15000},
            expected={"eitc": 600},
        )
        validators = [MockValidator("V1", ValidatorType.REFERENCE, 1000)]
        engine = ConsensusEngine(validators, tolerance=15.0)
        result = engine.validate(tc, "eitc", 2024, claude_confidence=0.5)
        assert len(result.potential_bugs) == 0

    def test_detect_potential_bugs_with_failed_validator(self):
        """Failed validators are skipped in bug detection."""
        tc = TestCase(
            name="test",
            inputs={"earned_income": 15000},
            expected={"eitc": 600},
            citation="26 USC 32",
        )
        validators = [
            MockValidator("V1", ValidatorType.REFERENCE, None, error="Failed"),
            MockValidator("V2", ValidatorType.REFERENCE, 1000),
        ]
        engine = ConsensusEngine(validators, tolerance=15.0)
        result = engine.validate(tc, "eitc", 2024, claude_confidence=0.95)
        # Only V2 should produce a bug (V1 is failed)
        assert len(result.potential_bugs) == 1
        assert result.potential_bugs[0]["validator"] == "V2"

    def test_reward_potential_upstream_bug(self):
        """Reward for POTENTIAL_UPSTREAM_BUG level."""
        tc = TestCase(
            name="test",
            inputs={"earned_income": 15000},
            expected={"eitc": 600},
        )
        validators = [
            MockValidator("V1", ValidatorType.REFERENCE, 800),
            MockValidator("V2", ValidatorType.REFERENCE, 850),
        ]
        engine = ConsensusEngine(validators, tolerance=15.0)
        result = engine.validate(tc, "eitc", 2024, claude_confidence=0.95)
        assert result.consensus_level == ConsensusLevel.POTENTIAL_UPSTREAM_BUG
        # Reward should be slightly positive for this level
        assert result.reward_signal >= 0

    def test_summary_no_consensus(self):
        """Summary with N/A consensus."""
        tc = TestCase(
            name="test",
            inputs={},
            expected={"eitc": 0},
        )
        validators = [
            MockValidator("V1", ValidatorType.REFERENCE, None, error="F"),
        ]
        engine = ConsensusEngine(validators)
        result = engine.validate(tc, "eitc", 2024)
        summary = result.summary()
        assert "N/A" in summary
