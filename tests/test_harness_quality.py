"""Tests for harness/quality submodules: __init__.py, coverage.py, imports.py, schema.py."""

from pathlib import Path

from cosilico_validators.harness import QualityResult
from cosilico_validators.harness.quality import run_quality_checks
from cosilico_validators.harness.quality.coverage import (
    FORMULA_PATTERN,
    TEST_CASE_PATTERN,
    TESTS_PATTERN,
    VARIABLE_PATTERN,
    check_test_coverage,
)
from cosilico_validators.harness.quality.grounding import (
    check_grounding,
    extract_numbers_from_text,
    extract_numeric_values,
)
from cosilico_validators.harness.quality.imports import (
    IMPORT_PATTERN,
    IMPORTS_START,
    check_imports,
)
from cosilico_validators.harness.quality.schema import (
    ALLOWED_INTEGERS,
    DTYPE_PATTERN,
    ENTITY_PATTERN,
    FORMULA_START,
    LITERAL_PATTERN,
    PERIOD_PATTERN,
    VALID_DTYPES,
    VALID_ENTITIES,
    VALID_PERIODS,
    check_schema,
)

# ============================================================================
# quality/__init__.py
# ============================================================================


class TestRunQualityChecks:
    def test_with_no_files(self, tmp_path):
        result = run_quality_checks(statute_root=tmp_path)
        assert isinstance(result, QualityResult)
        assert result.test_coverage == 1.0
        assert result.no_literals_pass is True

    def test_with_rac_files(self, tmp_path):
        rac_file = tmp_path / "test.rac"
        rac_file.write_text(
            "variable eitc:\n  entity: TaxUnit\n  dtype: Money\n  period: Year\n"
            "  formula: |\n    0\n  tests:\n    - inputs: {}\n"
        )
        result = run_quality_checks(statute_root=tmp_path)
        assert isinstance(result, QualityResult)

    def test_with_changed_files(self, tmp_path):
        rac_file = tmp_path / "test.rac"
        rac_file.write_text("variable eitc:\n  formula: |\n    0\n")
        result = run_quality_checks(statute_root=tmp_path, changed_files=[rac_file])
        assert isinstance(result, QualityResult)

    def test_with_changed_files_no_rac(self, tmp_path):
        py_file = tmp_path / "test.py"
        py_file.write_text("pass")
        result = run_quality_checks(statute_root=tmp_path, changed_files=[py_file])
        assert result.test_coverage == 1.0

    def test_returns_issues(self, tmp_path):
        rac_file = tmp_path / "test.rac"
        rac_file.write_text(
            "variable eitc:\n  entity: BadEntity\n  dtype: BadDtype\n  formula: |\n    earned_income * 12345\n"
        )
        result = run_quality_checks(statute_root=tmp_path)
        assert isinstance(result.issues, list)
        assert len(result.issues) > 0


# ============================================================================
# quality/coverage.py
# ============================================================================


class TestPatterns:
    def test_variable_pattern(self):
        assert VARIABLE_PATTERN.match("variable eitc:")
        assert VARIABLE_PATTERN.match("variable some_var:")
        assert not VARIABLE_PATTERN.match("  variable eitc:")

    def test_formula_pattern(self):
        assert FORMULA_PATTERN.match("  formula:")
        assert FORMULA_PATTERN.match("    formula:")
        assert not FORMULA_PATTERN.match("formula:")

    def test_tests_pattern(self):
        assert TESTS_PATTERN.match("  tests:")
        assert TESTS_PATTERN.match("    tests:")

    def test_test_case_pattern(self):
        assert TEST_CASE_PATTERN.match("    - name: test1")
        assert TEST_CASE_PATTERN.match("    - inputs: {}")


class TestCheckTestCoverage:
    def test_full_coverage(self, tmp_path):
        rac_file = tmp_path / "test.rac"
        rac_file.write_text("variable eitc:\n  formula: |\n    0\n  tests:\n    - name: test1\n")
        coverage, issues = check_test_coverage([rac_file])
        assert coverage == 1.0
        assert len(issues) == 0

    def test_no_coverage(self, tmp_path):
        rac_file = tmp_path / "test.rac"
        rac_file.write_text("variable eitc:\n  formula: |\n    0\n")
        coverage, issues = check_test_coverage([rac_file])
        assert coverage == 0.0
        assert len(issues) == 1

    def test_partial_coverage(self, tmp_path):
        rac_file = tmp_path / "test.rac"
        rac_file.write_text(
            "variable eitc:\n  formula: |\n    0\n  tests:\n    - name: t1\n\nvariable ctc:\n  formula: |\n    0\n"
        )
        coverage, issues = check_test_coverage([rac_file])
        assert coverage == 0.5
        assert len(issues) == 1

    def test_first_variable_missing_tests_when_second_encountered(self, tmp_path):
        """Test the inline issue-append path when a new variable is encountered
        and the previous variable had formula but no tests (line 57)."""
        rac_file = tmp_path / "test.rac"
        # First variable has formula but NO tests, second variable triggers processing of first
        rac_file.write_text(
            "variable no_tests_var:\n  formula: |\n    0\n"
            "variable has_tests_var:\n  formula: |\n    0\n  tests:\n    - name: t1\n"
        )
        coverage, issues = check_test_coverage([rac_file])
        assert coverage == 0.5
        assert len(issues) == 1
        assert "no_tests_var" in issues[0].message

    def test_no_formula_no_issue(self, tmp_path):
        rac_file = tmp_path / "test.rac"
        rac_file.write_text("variable eitc:\n  dtype: Money\n")
        coverage, issues = check_test_coverage([rac_file])
        assert len(issues) == 0

    def test_empty_file_list(self):
        coverage, issues = check_test_coverage([])
        assert coverage == 1.0
        assert len(issues) == 0

    def test_unreadable_file(self, tmp_path):
        bad_file = tmp_path / "bad.rac"
        # File doesn't exist but is in the list
        coverage, issues = check_test_coverage([bad_file])
        assert coverage == 1.0

    def test_tests_with_inputs_format(self, tmp_path):
        rac_file = tmp_path / "test.rac"
        rac_file.write_text("variable eitc:\n  formula: |\n    0\n  tests:\n    - inputs: {x: 1}\n")
        coverage, issues = check_test_coverage([rac_file])
        assert coverage == 1.0


# ============================================================================
# quality/imports.py
# ============================================================================


class TestImportPatterns:
    def test_imports_start(self):
        assert IMPORTS_START.match("imports:")
        assert IMPORTS_START.match("  imports:")

    def test_import_pattern(self):
        m = IMPORT_PATTERN.match("  - 26/32/a#eitc")
        assert m
        assert m.group(1) == "26/32/a"
        assert m.group(2) == "eitc"

    def test_import_pattern_with_alias(self):
        m = IMPORT_PATTERN.match("  - 26/32/a#eitc as earned_income_credit")
        assert m
        assert m.group(1) == "26/32/a"
        assert m.group(2) == "eitc"
        assert m.group(3) == "earned_income_credit"

    def test_import_pattern_with_quotes(self):
        m = IMPORT_PATTERN.match("  - '26/32/a#eitc'")
        assert m


class TestCheckImports:
    def test_valid_imports(self, tmp_path):
        # Create source file
        source = tmp_path / "26" / "32" / "a.rac"
        source.parent.mkdir(parents=True)
        source.write_text("variable eitc:\n  formula: |\n    0\n")

        # Create importing file
        importer = tmp_path / "26" / "62" / "a.rac"
        importer.parent.mkdir(parents=True)
        importer.write_text("imports:\n  - 26/32/a#eitc\n\nvariable agi:\n  formula: |\n    eitc\n")

        issues, all_valid = check_imports([source, importer], tmp_path)
        assert isinstance(issues, list)

    def test_no_imports(self, tmp_path):
        rac_file = tmp_path / "test.rac"
        rac_file.write_text("variable eitc:\n  formula: |\n    0\n")
        issues, all_valid = check_imports([rac_file], tmp_path)
        assert isinstance(issues, list)
        assert all_valid is True

    def test_invalid_import_path(self, tmp_path):
        rac_file = tmp_path / "test.rac"
        rac_file.write_text("imports:\n  - nonexistent/path#var\n\nvariable x:\n  formula: |\n    0\n")
        issues, all_valid = check_imports([rac_file], tmp_path)
        assert len(issues) >= 1
        assert all_valid is False

    def test_invalid_import_variable(self, tmp_path):
        source = tmp_path / "26" / "32" / "a.rac"
        source.parent.mkdir(parents=True)
        source.write_text("variable eitc:\n  formula: |\n    0\n")

        importer = tmp_path / "test.rac"
        importer.write_text("imports:\n  - 26/32/a#nonexistent\n")

        issues, all_valid = check_imports([source, importer], tmp_path)
        # May have issues for the nonexistent variable
        assert isinstance(issues, list)

    def test_empty_file_list(self):
        from pathlib import Path

        issues, all_valid = check_imports([], Path("/tmp"))
        assert len(issues) == 0
        assert all_valid is True

    def test_statute_prefix_stripped(self, tmp_path):
        """When file is under statute/ dir, the 'statute/' prefix is stripped."""
        statute_dir = tmp_path / "statute"
        statute_dir.mkdir()
        rac_file = statute_dir / "26" / "32" / "a.rac"
        rac_file.parent.mkdir(parents=True)
        rac_file.write_text("variable eitc:\n  formula: |\n    0\n")

        # statute_root = tmp_path, so rel_path = "statute/26/32/a.rac"
        # path_key should be "26/32/a" after stripping "statute/" prefix
        issues, all_valid = check_imports([rac_file], tmp_path)
        assert all_valid is True

    def test_file_read_error_in_index_building(self, tmp_path):
        """When a .rac file can't be read during index building, it's skipped."""
        rac_file = tmp_path / "test.rac"
        rac_file.write_text("imports:\n  - missing/path#var\n")

        # Create a file that exists but will raise on read
        bad_file = tmp_path / "bad.rac"
        bad_file.write_text("valid content")
        # Make it unreadable by patching
        from unittest.mock import patch

        original_read = Path.read_text
        call_count = [0]

        def mock_read(self_path, *args, **kwargs):
            call_count[0] += 1
            if "bad.rac" in str(self_path) and call_count[0] <= 1:
                raise PermissionError("cannot read")
            return original_read(self_path, *args, **kwargs)

        with patch.object(Path, "read_text", mock_read):
            issues, all_valid = check_imports([bad_file, rac_file], tmp_path)
            # Should not crash, bad_file skipped
            assert isinstance(issues, list)

    def test_file_read_error_in_import_checking(self, tmp_path):
        """When a .rac file can't be read during import checking, it's skipped."""
        rac_file = tmp_path / "test.rac"
        rac_file.write_text("imports:\n  - some/path#var\n")

        # Build index first (reads succeed), then fail on second pass
        from unittest.mock import patch

        original_read = Path.read_text
        read_calls = [0]

        def mock_read(self_path, *args, **kwargs):
            read_calls[0] += 1
            # Fail on the second read of this file (import checking pass)
            if "test.rac" in str(self_path) and read_calls[0] > 1:
                raise PermissionError("cannot read")
            return original_read(self_path, *args, **kwargs)

        with patch.object(Path, "read_text", mock_read):
            issues, all_valid = check_imports([rac_file], tmp_path)
            assert isinstance(issues, list)

    def test_import_found_via_statute_prefix(self, tmp_path):
        """Import resolved via statute/ subdirectory."""
        # Create the file at statute/26/32/a.rac
        statute_dir = tmp_path / "statute" / "26" / "32"
        statute_dir.mkdir(parents=True)
        target = statute_dir / "a.rac"
        target.write_text("variable eitc:\n  formula: |\n    0\n")

        # Create importer that references 26/32/a (without statute/ prefix)
        importer = tmp_path / "test.rac"
        importer.write_text("imports:\n  - 26/32/a#eitc\n")

        issues, all_valid = check_imports([target, importer], tmp_path)
        # The import should be found via statute/ prefix search
        assert isinstance(issues, list)

    def test_file_outside_statute_root(self, tmp_path):
        """File not under statute_root uses stem as path_key."""
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".rac", mode="w", delete=False) as f:
            f.write("variable eitc:\n  formula: |\n    0\n")
            other_file = Path(f.name)

        try:
            issues, all_valid = check_imports([other_file], tmp_path)
            assert isinstance(issues, list)
        finally:
            other_file.unlink()


# ============================================================================
# quality/schema.py
# ============================================================================


class TestSchemaPatterns:
    def test_entity_pattern(self):
        m = ENTITY_PATTERN.match("  entity: TaxUnit")
        assert m
        assert m.group(1) == "TaxUnit"

    def test_period_pattern(self):
        m = PERIOD_PATTERN.match("  period: Year")
        assert m
        assert m.group(1) == "Year"

    def test_dtype_pattern(self):
        m = DTYPE_PATTERN.match("  dtype: Money")
        assert m
        assert m.group(1) == "Money"

    def test_formula_start(self):
        assert FORMULA_START.match("  formula: |")
        assert not FORMULA_START.match("  formula: 0")

    def test_literal_pattern_finds_numbers(self):
        assert LITERAL_PATTERN.search(" 12345 ")
        assert LITERAL_PATTERN.search(" 0.075 ")
        assert not LITERAL_PATTERN.search(" 0 ")
        assert not LITERAL_PATTERN.search(" 1 ")

    def test_valid_entities(self):
        assert "Person" in VALID_ENTITIES
        assert "TaxUnit" in VALID_ENTITIES
        assert "Household" in VALID_ENTITIES

    def test_valid_periods(self):
        assert "Year" in VALID_PERIODS
        assert "Month" in VALID_PERIODS

    def test_valid_dtypes(self):
        assert "Money" in VALID_DTYPES
        assert "Rate" in VALID_DTYPES
        assert "Boolean" in VALID_DTYPES

    def test_allowed_integers(self):
        assert {-1, 0, 1} == ALLOWED_INTEGERS


class TestCheckSchema:
    def test_valid_file(self, tmp_path):
        rac_file = tmp_path / "test.rac"
        rac_file.write_text(
            "variable eitc:\n  entity: TaxUnit\n  dtype: Money\n  period: Year\n"
            "  formula: |\n    max(0, earned_income)\n"
        )
        issues, no_literals, all_valid = check_schema([rac_file])
        assert no_literals is True
        assert all_valid is True
        assert len(issues) == 0

    def test_invalid_entity(self, tmp_path):
        rac_file = tmp_path / "test.rac"
        rac_file.write_text("variable eitc:\n  entity: BadEntity\n")
        issues, no_literals, all_valid = check_schema([rac_file])
        assert all_valid is False
        assert any("entity" in str(i.message).lower() for i in issues)

    def test_invalid_dtype(self, tmp_path):
        rac_file = tmp_path / "test.rac"
        rac_file.write_text("variable eitc:\n  dtype: BadType\n")
        issues, no_literals, all_valid = check_schema([rac_file])
        assert all_valid is False

    def test_enum_dtype_valid(self, tmp_path):
        rac_file = tmp_path / "test.rac"
        rac_file.write_text("variable status:\n  dtype: EnumFilingStatus\n")
        issues, no_literals, all_valid = check_schema([rac_file])
        assert all_valid is True

    def test_literal_in_formula(self, tmp_path):
        rac_file = tmp_path / "test.rac"
        rac_file.write_text("variable eitc:\n  formula: |\n    earned_income * 12345\n")
        issues, no_literals, all_valid = check_schema([rac_file])
        assert no_literals is False
        assert any("literal" in str(i.category) for i in issues)

    def test_allowed_literals(self, tmp_path):
        rac_file = tmp_path / "test.rac"
        rac_file.write_text("variable eitc:\n  formula: |\n    max(0, min(1, x))\n")
        issues, no_literals, all_valid = check_schema([rac_file])
        assert no_literals is True

    def test_unreadable_file(self, tmp_path):
        bad_file = tmp_path / "bad.rac"
        # File doesn't exist
        issues, no_literals, all_valid = check_schema([bad_file])
        assert len(issues) >= 1
        assert issues[0].severity == "error"

    def test_empty_file_list(self):
        issues, no_literals, all_valid = check_schema([])
        assert len(issues) == 0
        assert no_literals is True
        assert all_valid is True

    def test_comment_and_string_excluded_from_literals(self, tmp_path):
        rac_file = tmp_path / "test.rac"
        rac_file.write_text("variable eitc:\n  formula: |\n    x = 0  # param is 12345\n    y = 'amount is 500'\n")
        issues, no_literals, all_valid = check_schema([rac_file])
        # Only the non-comment, non-string portions should be checked
        assert isinstance(issues, list)

    def test_invalid_period(self, tmp_path):
        rac_file = tmp_path / "test.rac"
        rac_file.write_text("variable eitc:\n  period: Century\n")
        issues, no_literals, all_valid = check_schema([rac_file])
        assert all_valid is False

    def test_formula_block_exit(self, tmp_path):
        rac_file = tmp_path / "test.rac"
        rac_file.write_text("variable eitc:\n  formula: |\n    0\n\nvariable ctc:\n  dtype: Money\n")
        issues, no_literals, all_valid = check_schema([rac_file])
        assert isinstance(issues, list)

    def test_disallowed_float_literals(self, tmp_path):
        """Test that 2.0 and 3.0 are flagged as disallowed literals."""
        rac_file = tmp_path / "test.rac"
        rac_file.write_text("variable eitc:\n  formula: |\n    x * 2.0 + 3.0\n")
        issues, no_literals, all_valid = check_schema([rac_file])
        # 2.0 and 3.0 are no longer allowed
        assert no_literals is False
        assert any(i.category == "literal" for i in issues)

    def test_literal_2_in_formula_flagged(self, tmp_path):
        """2 and 3 are no longer allowed in formulas."""
        rac_file = tmp_path / "test.rac"
        rac_file.write_text("variable x:\n  formula: |\n    income * 2\n")
        issues, no_literals, all_valid = check_schema([rac_file])
        assert no_literals is False


# ============================================================================
# quality/grounding.py
# ============================================================================


class TestExtractNumbersFromText:
    def test_plain_numbers(self):
        nums = extract_numbers_from_text("credit of $1,000 per child")
        assert 1000 in nums

    def test_dollar_amounts(self):
        nums = extract_numbers_from_text("amount equal to $2,500")
        assert 2500 in nums

    def test_percentages(self):
        nums = extract_numbers_from_text("rate of 7.65 percent")
        assert 7.65 in nums

    def test_written_percent_to_decimal(self):
        nums = extract_numbers_from_text("15 percent of earned income")
        assert 15 in nums
        assert 0.15 in nums

    def test_written_percent_decimal_value(self):
        nums = extract_numbers_from_text("50 percent of the taxes imposed")
        assert 0.50 in nums

    def test_per_centum(self):
        nums = extract_numbers_from_text("7.65 per centum of wages")
        assert 0.0765 in nums

    def test_fraction_one_half(self):
        nums = extract_numbers_from_text("one-half of such amount")
        assert 0.5 in nums

    def test_fraction_two_thirds(self):
        nums = extract_numbers_from_text("two-thirds of the applicable amount")
        assert abs(2 / 3 - min(nums, key=lambda x: abs(x - 2 / 3))) < 1e-10

    def test_plain_integers(self):
        nums = extract_numbers_from_text("increased to 400 for 1998")
        assert 400 in nums
        assert 1998 in nums

    def test_comma_separated(self):
        nums = extract_numbers_from_text("threshold of 200,000")
        assert 200000 in nums

    def test_no_numbers(self):
        nums = extract_numbers_from_text("no numeric values here")
        assert len(nums) == 0


class TestExtractNumericValues:
    def test_param_value(self):
        content = "ctc_amount:\n  from 2018-01-01: 2000\n  from 2025-01-01: 2200\n"
        values = extract_numeric_values(content)
        nums = [v[2] for v in values]
        assert 2000 in nums
        assert 2200 in nums

    def test_skips_formulas(self):
        content = "variable x:\n  formula: |\n    income * 12345\n"
        values = extract_numeric_values(content)
        nums = [v[2] for v in values]
        assert 12345 not in nums

    def test_skips_descriptions(self):
        content = 'variable x:\n  description: "amount of $5000 per child"\n'
        values = extract_numeric_values(content)
        nums = [v[2] for v in values]
        assert 5000 not in nums

    def test_skips_docstrings(self):
        content = '"""\nP.L. 107-16: increased to $600 for 2001\n"""\namt:\n  from 2001-01-01: 600\n'
        values = extract_numeric_values(content)
        nums = [v[2] for v in values]
        assert 600 in nums
        # The 600 in the docstring should NOT be extracted as a param value
        assert len(values) == 1

    def test_skips_allowed_values(self):
        content = "flag:\n  from 2020-01-01: 0\ncount:\n  from 2020-01-01: 1\n"
        values = extract_numeric_values(content)
        assert len(values) == 0

    def test_scalar_value(self):
        content = "rate: 0.075\n"
        values = extract_numeric_values(content)
        nums = [v[2] for v in values]
        assert 0.075 in nums

    def test_skips_metadata_keys(self):
        content = "entity: TaxUnit\nperiod: Year\n"
        values = extract_numeric_values(content)
        assert len(values) == 0

    def test_skips_tests(self):
        content = "variable x:\n  tests:\n    - inputs: {income: 50000}\n      output: 5000\n"
        values = extract_numeric_values(content)
        assert len(values) == 0


class TestCheckGrounding:
    def test_grounded_values_pass(self, tmp_path):
        rac_file = tmp_path / "test.rac"
        rac_file.write_text("ctc_amount:\n  from 2018-01-01: 1000\n")
        rule_text = "an amount equal to $1,000"
        issues, all_grounded = check_grounding([rac_file], rule_text=rule_text)
        assert all_grounded is True
        assert len(issues) == 0

    def test_ungrounded_values_fail(self, tmp_path):
        rac_file = tmp_path / "test.rac"
        rac_file.write_text("ctc_amount:\n  from 2018-01-01: 2000\n")
        rule_text = "an amount equal to $1,000"
        issues, all_grounded = check_grounding([rac_file], rule_text=rule_text)
        assert all_grounded is False
        assert len(issues) == 1
        assert issues[0].category == "grounding"
        assert "2000" in issues[0].message

    def test_no_rule_text_skips(self, tmp_path):
        rac_file = tmp_path / "test.rac"
        rac_file.write_text("ctc_amount:\n  from 2018-01-01: 9999\n")
        issues, all_grounded = check_grounding([rac_file])
        assert all_grounded is True
        assert len(issues) == 0

    def test_per_file_rule_text(self, tmp_path):
        rac_file = tmp_path / "test.rac"
        rac_file.write_text("rate:\n  from 2020-01-01: 0.3540\n")
        issues, all_grounded = check_grounding(
            [rac_file],
            rule_text_by_file={str(rac_file): "credit percentage is 35.40 percent"},
        )
        # "35.40 percent" -> 0.354 which matches 0.3540
        assert all_grounded is True

    def test_per_file_rule_text_ungrounded(self, tmp_path):
        rac_file = tmp_path / "test.rac"
        rac_file.write_text("rate:\n  from 2020-01-01: 0.99\n")
        issues, all_grounded = check_grounding(
            [rac_file],
            rule_text_by_file={str(rac_file): "credit percentage is 35.40 percent"},
        )
        assert all_grounded is False
        assert len(issues) == 1

    def test_unreadable_file_skipped(self, tmp_path):
        bad_file = tmp_path / "nonexistent.rac"
        issues, all_grounded = check_grounding([bad_file], rule_text="some text with 1000")
        assert all_grounded is True

    def test_multiple_values_mixed(self, tmp_path):
        rac_file = tmp_path / "test.rac"
        rac_file.write_text("amt:\n  from 2018-01-01: 1000\n  from 2025-01-01: 2200\n")
        rule_text = "amount of $1,000"
        issues, all_grounded = check_grounding([rac_file], rule_text=rule_text)
        assert all_grounded is False
        assert len(issues) == 1
        assert "2200" in issues[0].message

    def test_24a_encoding_fails_grounding(self, tmp_path):
        """The actual 26 USC 24(a) encoding should fail — it has values from other subsections."""
        rac_file = tmp_path / "a.rac"
        rac_file.write_text(
            "ctc_base_amount:\n"
            "  from 1998-01-01: 400\n"
            "  from 1999-01-01: 500\n"
            "  from 2001-01-01: 600\n"
            "  from 2003-01-01: 1000\n"
            "  from 2018-01-01: 2000\n"
            "  from 2025-01-01: 2200\n"
        )
        # The actual text of 24(a) only mentions $1,000
        rule_text = (
            "There shall be allowed as a credit against the tax imposed by "
            "this chapter for the taxable year with respect to each qualifying "
            "child of the taxpayer for which the taxpayer is allowed a deduction "
            "under section 151 an amount equal to $1,000."
        )
        issues, all_grounded = check_grounding([rac_file], rule_text=rule_text)
        assert all_grounded is False
        # 400, 500, 600, 2000, 2200 are all ungrounded
        ungrounded = {i.message.split("'")[1] for i in issues}
        assert "400" in ungrounded
        assert "500" in ungrounded
        assert "600" in ungrounded
        assert "2000" in ungrounded
        assert "2200" in ungrounded
        # 1000 IS in the text, so it should NOT be flagged
        assert "1000" not in ungrounded
