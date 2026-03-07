"""Consensus engine - aggregate results from multiple validators."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from cosilico_validators.validators.base import (
    BaseValidator,
    TestCase,
    ValidatorResult,
    ValidatorType,
)


class ConsensusLevel(Enum):
    """Level of agreement across validators."""

    FULL_AGREEMENT = "full_agreement"  # All validators agree
    PRIMARY_CONFIRMED = "primary_confirmed"  # Primary + majority agree
    MAJORITY_AGREEMENT = "majority_agreement"  # >50% agree
    DISAGREEMENT = "disagreement"  # No consensus
    POTENTIAL_UPSTREAM_BUG = "potential_upstream_bug"  # DSL confident, validators disagree


@dataclass
class ValidationResult:
    """Result from multi-system validation."""

    test_case: TestCase
    variable: str
    expected_value: float
    validator_results: dict[str, ValidatorResult]
    consensus_level: ConsensusLevel
    consensus_value: float | None
    reward_signal: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    potential_bugs: list[dict[str, Any]] = field(default_factory=list)

    @property
    def matches_expected(self) -> bool:
        """Check if consensus matches expected value within tolerance."""
        if self.consensus_value is None:
            return False
        return abs(self.consensus_value - self.expected_value) <= 15.0  # $15 tolerance

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Test: {self.test_case.name}",
            f"Variable: {self.variable}",
            f"Expected: ${self.expected_value:,.0f}",
            f"Consensus: ${self.consensus_value:,.0f}" if self.consensus_value else "Consensus: N/A",
            f"Level: {self.consensus_level.value}",
            f"Reward: {self.reward_signal:+.2f}",
            f"Confidence: {self.confidence:.1%}",
        ]
        if self.potential_bugs:
            lines.append(f"Potential bugs: {len(self.potential_bugs)}")
        return "\n".join(lines)


class ConsensusEngine:
    """Engine for aggregating validation results and computing consensus."""

    def __init__(
        self,
        validators: list[BaseValidator],
        tolerance: float = 15.0,
        primary_weight: float = 2.0,
    ):
        """Initialize consensus engine.

        Args:
            validators: List of validators to use
            tolerance: Dollar tolerance for matching ($15 default)
            primary_weight: Weight multiplier for PRIMARY validators
        """
        self.validators = validators
        self.tolerance = tolerance
        self.primary_weight = primary_weight

        # Sort by validator type for consistent ordering
        self.validators.sort(
            key=lambda v: (
                0
                if v.validator_type == ValidatorType.PRIMARY
                else 1
                if v.validator_type == ValidatorType.REFERENCE
                else 2
            )
        )

    def validate(
        self,
        test_case: TestCase,
        variable: str,
        year: int = 2024,
        claude_confidence: float | None = None,
    ) -> ValidationResult:
        """Run validation across all validators and compute consensus.

        Args:
            test_case: Test case with inputs and expected output
            variable: Variable to validate
            year: Tax year
            claude_confidence: Optional confidence score from Claude encoder (0-1)

        Returns:
            ValidationResult with consensus and reward signal
        """
        # Get expected value
        expected_value = None
        for var, value in test_case.expected.items():
            if variable.lower() in var.lower():
                expected_value = value
                break

        if expected_value is None:
            expected_value = list(test_case.expected.values())[0] if test_case.expected else 0

        # Run all validators
        validator_results: dict[str, ValidatorResult] = {}
        for validator in self.validators:
            if validator.supports_variable(variable):
                result = validator.validate(test_case, variable, year)
                validator_results[validator.name] = result

        # Compute consensus
        consensus_value, consensus_level = self._compute_consensus(validator_results, expected_value, claude_confidence)

        # Compute reward signal
        reward_signal = self._compute_reward(validator_results, expected_value, consensus_level)

        # Compute confidence
        confidence = self._compute_confidence(validator_results, consensus_value)

        # Detect potential upstream bugs
        potential_bugs = self._detect_potential_bugs(validator_results, expected_value, claude_confidence, test_case)

        return ValidationResult(
            test_case=test_case,
            variable=variable,
            expected_value=expected_value,
            validator_results=validator_results,
            consensus_level=consensus_level,
            consensus_value=consensus_value,
            reward_signal=reward_signal,
            confidence=confidence,
            potential_bugs=potential_bugs,
        )

    def _compute_consensus(
        self,
        results: dict[str, ValidatorResult],
        expected: float,
        claude_confidence: float | None,
    ) -> tuple[float | None, ConsensusLevel]:
        """Compute consensus value and level from validator results."""
        # Get successful results with values
        values: list[tuple[str, float, ValidatorType]] = []
        for name, result in results.items():
            if result.success and result.calculated_value is not None:
                values.append((name, result.calculated_value, result.validator_type))

        if not values:
            return None, ConsensusLevel.DISAGREEMENT

        # Check for full agreement (within tolerance)
        all_values = [v[1] for v in values]
        mean_value = sum(all_values) / len(all_values)

        all_agree = all(abs(v - mean_value) <= self.tolerance for v in all_values)
        if all_agree:
            return mean_value, ConsensusLevel.FULL_AGREEMENT

        # Check if primary validator agrees with expected
        primary_results = [(n, v, t) for n, v, t in values if t == ValidatorType.PRIMARY]
        if primary_results:
            primary_value = primary_results[0][1]
            if abs(primary_value - expected) <= self.tolerance:
                # Check if majority also agrees with primary
                agreeing = sum(1 for _, v, _ in values if abs(v - primary_value) <= self.tolerance)
                if agreeing > len(values) / 2:
                    return primary_value, ConsensusLevel.PRIMARY_CONFIRMED

        # Check for majority agreement
        # Group values by similarity
        clusters: list[list[float]] = []
        for _, v, _ in values:
            added = False
            for cluster in clusters:
                if abs(v - cluster[0]) <= self.tolerance:
                    cluster.append(v)
                    added = True
                    break
            if not added:
                clusters.append([v])

        # Find largest cluster
        largest_cluster = max(clusters, key=len)
        if len(largest_cluster) > len(values) / 2:
            cluster_mean = sum(largest_cluster) / len(largest_cluster)
            return cluster_mean, ConsensusLevel.MAJORITY_AGREEMENT

        # Check for potential upstream bug
        if claude_confidence and claude_confidence > 0.9:
            # Claude is confident but validators disagree
            return expected, ConsensusLevel.POTENTIAL_UPSTREAM_BUG

        # No consensus
        return mean_value, ConsensusLevel.DISAGREEMENT

    def _compute_reward(
        self,
        results: dict[str, ValidatorResult],
        expected: float,
        consensus_level: ConsensusLevel,
    ) -> float:
        """Compute reward signal (-1.0 to 1.0) for training.

        Higher reward when:
        - More validators agree with expected value
        - Primary validators confirm
        - High consensus level
        """
        if not results:
            return 0.0

        # Base reward from consensus level
        level_rewards = {
            ConsensusLevel.FULL_AGREEMENT: 0.5,
            ConsensusLevel.PRIMARY_CONFIRMED: 0.4,
            ConsensusLevel.MAJORITY_AGREEMENT: 0.2,
            ConsensusLevel.DISAGREEMENT: -0.2,
            ConsensusLevel.POTENTIAL_UPSTREAM_BUG: 0.1,  # Slight positive - investigate
        }
        reward = level_rewards.get(consensus_level, 0.0)

        # Add reward for matching expected value
        matches = 0
        total_weight = 0
        for _name, result in results.items():
            if result.success and result.calculated_value is not None:
                weight = self.primary_weight if result.validator_type == ValidatorType.PRIMARY else 1.0
                total_weight += weight
                if abs(result.calculated_value - expected) <= self.tolerance:
                    matches += weight

        if total_weight > 0:
            match_ratio = matches / total_weight
            reward += match_ratio * 0.5  # Up to +0.5 for all matches

        return max(-1.0, min(1.0, reward))

    def _compute_confidence(self, results: dict[str, ValidatorResult], consensus_value: float | None) -> float:
        """Compute confidence in the validation result (0.0 to 1.0)."""
        if consensus_value is None:
            return 0.0

        successful = [r for r in results.values() if r.success]
        if not successful:
            return 0.0

        # Confidence based on:
        # 1. Number of successful validators
        # 2. Agreement with consensus value
        # 3. Presence of primary validator

        n_successful = len(successful)
        n_total = len(results)

        # Base confidence from success rate
        success_rate = n_successful / n_total if n_total > 0 else 0

        # Agreement with consensus
        agreeing = sum(
            1
            for r in successful
            if r.calculated_value is not None and abs(r.calculated_value - consensus_value) <= self.tolerance
        )
        agreement_rate = agreeing / n_successful if n_successful > 0 else 0

        # Bonus for primary validator
        has_primary = any(r.validator_type == ValidatorType.PRIMARY and r.success for r in results.values())
        primary_bonus = 0.1 if has_primary else 0

        confidence = success_rate * 0.3 + agreement_rate * 0.6 + primary_bonus
        return min(1.0, confidence)

    def _detect_potential_bugs(
        self,
        results: dict[str, ValidatorResult],
        expected: float,
        claude_confidence: float | None,
        test_case: TestCase,
    ) -> list[dict[str, Any]]:
        """Detect potential bugs in upstream systems.

        Returns list of potential bugs when:
        - Claude is highly confident
        - Expected value differs from validator result
        - Citation is clear
        """
        potential_bugs = []

        if not claude_confidence or claude_confidence < 0.9:
            return potential_bugs

        for name, result in results.items():
            if not result.success or result.calculated_value is None:
                continue

            diff = abs(result.calculated_value - expected)
            if diff > self.tolerance:
                # Potential bug in this validator
                potential_bugs.append(
                    {
                        "validator": name,
                        "validator_type": result.validator_type.value,
                        "expected": expected,
                        "actual": result.calculated_value,
                        "difference": diff,
                        "citation": test_case.citation,
                        "test_case": test_case.name,
                        "inputs": test_case.inputs,
                        "claude_confidence": claude_confidence,
                    }
                )

        return potential_bugs

    def batch_validate(
        self,
        test_cases: list[TestCase],
        variable: str,
        year: int = 2024,
    ) -> list[ValidationResult]:
        """Validate multiple test cases."""
        return [self.validate(tc, variable, year) for tc in test_cases]
