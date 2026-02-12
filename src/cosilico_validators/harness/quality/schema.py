"""Schema validation for .rac files.

Checks:
- Valid entity types: Person, TaxUnit, Household, Family
- Valid period types: Year, Month, Day
- Valid dtype types: Money, Rate, Boolean, Integer, Enum[...]
- No hardcoded literals (only -1, 0, 1, 2, 3 allowed)
"""

import re
from pathlib import Path

from .. import QualityIssue

# Valid values for schema fields
VALID_ENTITIES = {
    # Core tax/benefit units
    "Person", "TaxUnit", "Household", "Family",
    # Benefit program units
    "TanfUnit", "SnapUnit", "SPMUnit",
    # Business/asset entities (for corporate/capital gains)
    "Corporation", "Business", "Asset",
}
VALID_PERIODS = {"Year", "Month", "Week", "Day"}
VALID_DTYPES = {"Money", "Rate", "Boolean", "Integer", "Count", "String", "Decimal"}
ALLOWED_INTEGERS = {-1, 0, 1, 2, 3}

# Regex patterns
ENTITY_PATTERN = re.compile(r"^\s*entity:\s*(\w+)")
# Period type declarations only - exclude date-like values (2024-01, etc.)
PERIOD_PATTERN = re.compile(r"^\s*period:\s*(Year|Month|Week|Day|[A-Z][a-z]+)$")
DTYPE_PATTERN = re.compile(r"^\s*dtype:\s*(\w+)")
FORMULA_START = re.compile(r"^\s*formula:\s*\|")
FORMULA_LINE = re.compile(r"^\s{4,}")  # Indented lines in formula

# Pattern to find literals in formulas
# Matches integers > 3 or any float (but not digits within larger numbers)
LITERAL_PATTERN = re.compile(
    r"""
    (?<![a-zA-Z_\d])  # Not preceded by identifier char or digit
    (
        \d+\.\d+      # Float like 0.075
        |
        [4-9]         # Single digit 4-9
        |
        [1-9]\d+      # Multi-digit starting with 1-9 (10+)
    )
    (?![a-zA-Z_\d])   # Not followed by identifier char or digit
    """,
    re.VERBOSE,
)


def check_schema(rac_files: list[Path]) -> tuple[list[QualityIssue], bool, bool]:
    """Check schema validity of .rac files.

    Returns:
        Tuple of (issues, no_literals_pass, all_dtypes_valid)
    """
    issues: list[QualityIssue] = []
    has_literal_issues = False
    has_dtype_issues = False

    for rac_file in rac_files:
        try:
            content = rac_file.read_text()
            lines = content.split("\n")
        except Exception as e:
            issues.append(
                QualityIssue(
                    file=str(rac_file),
                    line=None,
                    category="schema",
                    severity="error",
                    message=f"Could not read file: {e}",
                )
            )
            continue

        in_formula = False

        for i, line in enumerate(lines, 1):
            # Track if we're in a formula block
            if FORMULA_START.match(line):
                in_formula = True
                continue
            elif in_formula and not FORMULA_LINE.match(line) and line.strip():
                in_formula = False

            # Check entity
            entity_match = ENTITY_PATTERN.match(line)
            if entity_match:
                entity = entity_match.group(1)
                if entity not in VALID_ENTITIES:
                    issues.append(
                        QualityIssue(
                            file=str(rac_file),
                            line=i,
                            category="schema",
                            severity="error",
                            message=f"Invalid entity '{entity}'. Must be one of: {VALID_ENTITIES}",
                        )
                    )
                    has_dtype_issues = True

            # Check period
            period_match = PERIOD_PATTERN.match(line)
            if period_match:
                period = period_match.group(1)
                if period not in VALID_PERIODS:
                    issues.append(
                        QualityIssue(
                            file=str(rac_file),
                            line=i,
                            category="schema",
                            severity="error",
                            message=f"Invalid period '{period}'. Must be one of: {VALID_PERIODS}",
                        )
                    )
                    has_dtype_issues = True

            # Check dtype
            dtype_match = DTYPE_PATTERN.match(line)
            if dtype_match:
                dtype = dtype_match.group(1)
                # Handle Enum[...] specially
                if not (dtype in VALID_DTYPES or dtype.startswith("Enum")):
                    issues.append(
                        QualityIssue(
                            file=str(rac_file),
                            line=i,
                            category="schema",
                            severity="error",
                            message=f"Invalid dtype '{dtype}'. Must be one of: {VALID_DTYPES} or Enum[...]",
                        )
                    )
                    has_dtype_issues = True

            # Check for hardcoded literals in formulas
            if in_formula:
                # Skip comments and strings
                code_line = re.sub(r"#.*$", "", line)  # Remove comments
                code_line = re.sub(r"['\"].*?['\"]", "", code_line)  # Remove strings

                for match in LITERAL_PATTERN.finditer(code_line):
                    literal = match.group(1)
                    # Check if it's an allowed value (integers -1,0,1,2,3 or their float equivalents)
                    try:
                        val = float(literal)
                        if val in {-1.0, 0.0, 1.0, 2.0, 3.0}:
                            continue
                    except ValueError:  # pragma: no cover – regex only matches numeric literals
                        pass

                    issues.append(
                        QualityIssue(
                            file=str(rac_file),
                            line=i,
                            category="literal",
                            severity="error",
                            message=f"Hardcoded literal '{literal}' found. Use a parameter instead.",
                        )
                    )
                    has_literal_issues = True

    return issues, not has_literal_issues, not has_dtype_issues
