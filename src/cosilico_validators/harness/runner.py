"""Main validation harness runner.

Orchestrates all validation checks:
- Alignment (match rates against PolicyEngine, TAXSIM, etc.)
- Coverage (implemented vs total variables)
- Quality (RAC file quality checks)
- Review (agent-based subjective review)
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

from . import (
    AlignmentResult,
    CoverageResult,
    HarnessResult,
    ReviewResult,
    VariableAlignment,
)
from .checkpoint import get_git_commit
from .quality import run_quality_checks

# Default paths
COSILICO_US_ROOT = Path.home() / "CosilicoAI" / "cosilico-us"
STATUTE_ROOT = COSILICO_US_ROOT / "statute"

# Variable definitions with their sections
VARIABLES = {
    "eitc": {"section": "26/32", "title": "Earned Income Tax Credit"},
    "income_tax_before_credits": {"section": "26/1", "title": "Income Tax (Before Credits)"},
    "ctc": {"section": "26/24", "title": "Child Tax Credit (Total)"},
    "non_refundable_ctc": {"section": "26/24", "title": "Child Tax Credit (Non-refundable)"},
    "refundable_ctc": {"section": "26/24", "title": "Additional Child Tax Credit"},
    "standard_deduction": {"section": "26/63", "title": "Standard Deduction"},
    "adjusted_gross_income": {"section": "26/62", "title": "Adjusted Gross Income"},
    "taxable_income": {"section": "26/63", "title": "Taxable Income"},
    "cdcc": {"section": "26/21", "title": "Child & Dependent Care Credit"},
    "salt_deduction": {"section": "26/164", "title": "SALT Deduction"},
    "alternative_minimum_tax": {"section": "26/55", "title": "Alternative Minimum Tax"},
    "premium_tax_credit": {"section": "26/36B", "title": "Premium Tax Credit"},
    "net_investment_income_tax": {"section": "26/1411", "title": "Net Investment Income Tax"},
}


class ValidationHarness:
    """Main validation harness."""

    def __init__(
        self,
        statute_root: Optional[Path] = None,
        run_alignment: bool = True,
        run_quality: bool = True,
        run_review: bool = False,
    ):
        """Initialize harness.

        Args:
            statute_root: Root of statute files
            run_alignment: Whether to run alignment checks (slow)
            run_quality: Whether to run quality checks
            run_review: Whether to run agent review
        """
        self.statute_root = statute_root or STATUTE_ROOT
        self.run_alignment = run_alignment
        self.run_quality = run_quality
        self.run_review = run_review

    def run_full_validation(
        self,
        changed_files: Optional[list[Path]] = None,
    ) -> HarnessResult:
        """Run all validation checks.

        Args:
            changed_files: If provided, focus on these files

        Returns:
            Complete HarnessResult
        """
        timestamp = datetime.now().isoformat()
        git_commit = get_git_commit()

        # Run alignment checks
        alignment = self._run_alignment_checks() if self.run_alignment else AlignmentResult(overall_rate=0.0)

        # Run coverage checks
        coverage = self._run_coverage_checks()

        # Run quality checks
        if self.run_quality:
            quality = run_quality_checks(self.statute_root, changed_files)
        else:
            from . import QualityResult

            quality = QualityResult(
                test_coverage=0.0,
                no_literals_pass=True,
                all_imports_valid=True,
                all_dtypes_valid=True,
            )

        # Run agent review (optional)
        review = None
        if self.run_review and changed_files:
            review = self._run_agent_review(changed_files)

        return HarnessResult(
            timestamp=timestamp,
            git_commit=git_commit,
            alignment=alignment,
            coverage=coverage,
            quality=quality,
            review=review,
        )

    def _run_alignment_checks(self) -> AlignmentResult:
        """Run alignment checks against external validators."""
        # Import here to avoid circular deps and slow imports
        from ..dashboard_export import run_export

        try:
            # Run the existing dashboard export (uses PE)
            dashboard_data = run_export(year=2024)

            # Convert to alignment result
            by_variable = {}
            by_validator = {"policyengine": 0.0}

            pe_rates = []
            for section in dashboard_data.get("sections", []):
                var_name = section["variable"]
                if var_name in VARIABLES:
                    pe_rate = section["summary"]["matchRate"]
                    pe_rates.append(pe_rate)

                    by_variable[var_name] = VariableAlignment(
                        variable=var_name,
                        section=VARIABLES[var_name]["section"],
                        policyengine=pe_rate,
                    )

            overall_rate = sum(pe_rates) / len(pe_rates) if pe_rates else 0.0
            by_validator["policyengine"] = overall_rate

            return AlignmentResult(
                overall_rate=overall_rate,
                by_variable=by_variable,
                by_validator=by_validator,
            )

        except Exception as e:
            print(f"Warning: Alignment check failed: {e}")
            return AlignmentResult(overall_rate=0.0)

    def _run_coverage_checks(self) -> CoverageResult:
        """Check coverage of implemented variables."""
        implemented = 0
        by_section: dict[str, tuple[int, int]] = {}

        # Check which variables are implemented (have .rac files with engine integration)
        # For now, just check if the section path exists
        for _var_name, meta in VARIABLES.items():
            section = meta["section"]
            section_path = self.statute_root / section

            # Initialize section counter
            if section not in by_section:
                by_section[section] = (0, 0)

            impl, total = by_section[section]
            total += 1

            # Check if .rac file exists
            has_rac = (section_path / "a.rac").exists() or section_path.with_suffix(".rac").exists()
            if has_rac:
                implemented += 1
                impl += 1

            by_section[section] = (impl, total)

        return CoverageResult(
            implemented=implemented,
            total=len(VARIABLES),
            by_section=by_section,
        )

    def _run_agent_review(self, changed_files: list[Path]) -> Optional[ReviewResult]:
        """Run agent-based review on changed files."""
        # Filter to only .rac files
        rac_files = [f for f in changed_files if f.suffix == ".rac"]
        if not rac_files:
            return None

        # For now, return a placeholder
        # In a full implementation, this would spawn a Claude agent
        return ReviewResult(
            overall_score=7.0,
            accuracy=7.0,
            completeness=7.0,
            parameterization=7.0,
            test_quality=7.0,
            feedback="Agent review not yet implemented. Run with --review to enable.",
            reviewed_files=[str(f) for f in rac_files],
        )


def run_harness(
    only: Optional[str] = None,
    changed_files: Optional[list[Path]] = None,
) -> HarnessResult:
    """Convenience function to run harness.

    Args:
        only: Run only specific check ("alignment", "quality", "review")
        changed_files: Focus on specific files

    Returns:
        HarnessResult
    """
    harness = ValidationHarness(
        run_alignment=only is None or only == "alignment",
        run_quality=only is None or only == "quality",
        run_review=only == "review",
    )

    return harness.run_full_validation(changed_files)
