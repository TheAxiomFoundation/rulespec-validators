"""Source grounding validation for .rac files.

Checks that all numeric values in parameter/variable definitions are
present in the source rule text. Encodings must be grounded in the
source — no invented, hallucinated, or cross-referenced values.
"""

import contextlib
import re
from pathlib import Path

from .. import QualityIssue

# Matches numeric values in parameter definitions like:
#   from 2018-01-01: 2000
#   from 1998-01-01: 400
#   value: 0.3540
# But NOT dates (2018-01-01), comments, or description strings.
PARAM_VALUE_PATTERN = re.compile(
    r"^\s*(?:from\s+\d{4}-\d{2}-\d{2}:\s*|value:\s*)"
    r"(-?[\d,]+(?:\.\d+)?)"
)

# Matches scalar variable definitions like:
#   some_var: 1000
#   rate: 0.075
SCALAR_VALUE_PATTERN = re.compile(r"^(\w[\w_]*):\s*(-?[\d,]+(?:\.\d+)?)\s*$")

# Numbers that are always allowed (trivial values)
ALLOWED_VALUES = {-1, 0, 1}

# Metadata keys whose values should not be checked
METADATA_KEYS = {
    "entity",
    "period",
    "dtype",
    "unit",
    "label",
    "description",
    "status",
    "indexed_by",
    "formula",
    "tests",
    "imports",
    "variable",
}


def extract_numeric_values(content: str) -> list[tuple[int, str, float]]:
    """Extract all numeric values from parameter/variable definitions.

    Returns list of (line_number, raw_string, numeric_value).
    """
    values = []
    in_formula = False
    in_tests = False
    in_docstring = False

    lines = content.split("\n")
    for i, line in enumerate(lines, 1):
        stripped = line.strip()

        # Track docstring blocks (triple-quoted)
        if '"""' in stripped:
            if in_docstring:
                in_docstring = False
                continue
            else:
                in_docstring = True
                continue
        if in_docstring:
            continue

        # Skip comments
        if stripped.startswith("#"):
            continue

        # Track formula blocks
        if re.match(r"\s*formula:\s*\|", line):
            in_formula = True
            continue
        if re.match(r"\s*tests:", line):
            in_tests = True
            in_formula = False
            continue
        if in_formula and stripped and not line.startswith(" " * 4):
            in_formula = False
        if in_tests and stripped and not line.startswith(" "):
            in_tests = False

        # Skip formulas and tests
        if in_formula or in_tests:
            continue

        # Skip description strings
        if re.match(r'\s*description:\s*"', line) or re.match(r"\s*description:\s*'", line):
            continue
        if re.match(r'\s*label:\s*"', line) or re.match(r"\s*label:\s*'", line):
            continue

        # Check for parameter values (from DATE: VALUE)
        m = PARAM_VALUE_PATTERN.match(line)
        if m:
            raw = m.group(1).replace(",", "")
            try:
                val = float(raw)
                if val not in ALLOWED_VALUES:
                    values.append((i, raw, val))
            except ValueError:
                pass
            continue

        # Check for scalar values (key: VALUE)
        m = SCALAR_VALUE_PATTERN.match(stripped)
        if m:
            key = m.group(1)
            if key.lower() not in METADATA_KEYS:
                raw = m.group(2).replace(",", "")
                try:
                    val = float(raw)
                    if val not in ALLOWED_VALUES:
                        values.append((i, raw, val))
                except ValueError:
                    pass

    return values


def extract_numbers_from_text(text: str) -> set[float]:
    """Extract all numbers from rule text.

    Handles:
    - Plain numbers: 1000, 2000
    - Comma-separated: 1,000, 200,000
    - Dollar amounts: $1,000, $200,000
    - Percentages: 7.65%, 34%
    - Decimals: 0.075, 0.3540
    - Negative: -1000
    - Written percentages: "15 percent" -> 0.15, 15
    - Fractions: "one-half" -> 0.5
    """
    numbers = set()

    # Find all number-like patterns in the text
    # Matches: optional $, optional -, digits with optional commas, optional decimal
    pattern = re.compile(r"(?:^|(?<=[\s$(\[,]))(-?[\d,]+(?:\.\d+)?)\b")

    for m in pattern.finditer(text):
        raw = m.group(1).replace(",", "")
        with contextlib.suppress(ValueError):
            numbers.add(float(raw))

    # "X percent" / "X per centum" -> also add X/100 as decimal
    percent_pattern = re.compile(
        r"(\d+(?:\.\d+)?)\s+(?:percent|per\s*centum)", re.IGNORECASE
    )
    for m in percent_pattern.finditer(text):
        with contextlib.suppress(ValueError):
            numbers.add(float(m.group(1)) / 100)

    # Common written fractions
    fraction_words = {
        "one-half": 0.5,
        "one half": 0.5,
        "one-third": 1 / 3,
        "one third": 1 / 3,
        "two-thirds": 2 / 3,
        "two thirds": 2 / 3,
        "one-quarter": 0.25,
        "one quarter": 0.25,
        "three-quarters": 0.75,
        "three quarters": 0.75,
    }
    text_lower = text.lower()
    for word, val in fraction_words.items():
        if word in text_lower:
            numbers.add(val)

    return numbers


def check_grounding(
    rac_files: list[Path],
    rule_text: str | None = None,
    rule_text_by_file: dict[str, str] | None = None,
) -> tuple[list[QualityIssue], bool]:
    """Check that all numeric values in .rac files are grounded in the rule text.

    Args:
        rac_files: List of .rac files to check.
        rule_text: The full source rule text. All numbers in encodings
            must appear somewhere in this text.
        rule_text_by_file: Optional per-file rule text mapping (file path -> text).
            If provided, each file is checked against its own rule text.

    Returns:
        Tuple of (issues, all_grounded).
    """
    issues: list[QualityIssue] = []
    all_grounded = True

    if rule_text is None and rule_text_by_file is None:
        return issues, True

    # Extract numbers from the full rule text
    rule_numbers = extract_numbers_from_text(rule_text) if rule_text else set()

    for rac_file in rac_files:
        try:
            content = rac_file.read_text()
        except Exception:
            continue

        # Get file-specific rule text if available
        file_key = str(rac_file)
        file_rule_numbers = rule_numbers.copy()
        if rule_text_by_file and file_key in rule_text_by_file:
            file_rule_numbers |= extract_numbers_from_text(rule_text_by_file[file_key])

        if not file_rule_numbers:
            continue

        numeric_values = extract_numeric_values(content)

        for line_num, raw_str, val in numeric_values:
            if val not in file_rule_numbers:
                issues.append(
                    QualityIssue(
                        file=str(rac_file),
                        line=line_num,
                        category="grounding",
                        severity="error",
                        message=(
                            f"Value '{raw_str}' not found in rule text. "
                            f"All numeric values must be grounded in the source."
                        ),
                    )
                )
                all_grounded = False

    return issues, all_grounded
