"""Test coverage analysis for .rac files.

Checks that variables with formulas have associated tests.
"""

import re
from pathlib import Path

from .. import QualityIssue

# Pattern to find variable declarations
VARIABLE_PATTERN = re.compile(r"^variable\s+(\w+):")
# Pattern to find formula blocks
FORMULA_PATTERN = re.compile(r"^\s+formula:")
# Pattern to find tests blocks
TESTS_PATTERN = re.compile(r"^\s+tests:")
# Pattern to find individual test cases
TEST_CASE_PATTERN = re.compile(r"^\s+-\s+(?:name:|inputs:)")


def check_test_coverage(rac_files: list[Path]) -> tuple[float, list[QualityIssue]]:
    """Check test coverage for .rac files.

    Returns:
        Tuple of (coverage_rate, issues)
        coverage_rate is 0-1 representing % of variables with tests
    """
    issues: list[QualityIssue] = []
    total_variables = 0
    variables_with_tests = 0

    for rac_file in rac_files:
        try:
            content = rac_file.read_text()
            lines = content.split("\n")
        except Exception:
            continue

        # Parse variables and their test status
        current_variable = None
        current_variable_line = None
        has_formula = False
        has_tests = False
        test_count = 0

        for i, line in enumerate(lines, 1):
            # Check for new variable
            var_match = VARIABLE_PATTERN.match(line)
            if var_match:
                # Save previous variable info
                if current_variable and has_formula:
                    total_variables += 1
                    if has_tests and test_count > 0:
                        variables_with_tests += 1
                    else:
                        issues.append(
                            QualityIssue(
                                file=str(rac_file),
                                line=current_variable_line,
                                category="test_coverage",
                                severity="warning",
                                message=f"Variable '{current_variable}' has a formula but no tests.",
                            )
                        )

                # Start tracking new variable
                current_variable = var_match.group(1)
                current_variable_line = i
                has_formula = False
                has_tests = False
                test_count = 0
                continue

            # Check for formula
            if FORMULA_PATTERN.match(line):
                has_formula = True
                continue

            # Check for tests section
            if TESTS_PATTERN.match(line):
                has_tests = True
                continue

            # Count test cases
            if has_tests and TEST_CASE_PATTERN.match(line):
                test_count += 1

        # Don't forget the last variable
        if current_variable and has_formula:
            total_variables += 1
            if has_tests and test_count > 0:
                variables_with_tests += 1
            else:
                issues.append(
                    QualityIssue(
                        file=str(rac_file),
                        line=current_variable_line,
                        category="test_coverage",
                        severity="warning",
                        message=f"Variable '{current_variable}' has a formula but no tests.",
                    )
                )

    coverage_rate = variables_with_tests / total_variables if total_variables > 0 else 1.0

    return coverage_rate, issues
