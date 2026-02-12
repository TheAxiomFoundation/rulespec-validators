"""Record-by-record Cosilico vs PolicyEngine comparison."""

from .aligned import (
    CommonDataset,
    ComparisonResult,
    compare_variable,
    load_common_dataset,
    run_aligned_comparison,
)
from .core import (
    compare_records,
    generate_dashboard_json,
    load_cosilico_values,
    load_pe_values,
    run_full_comparison,
    run_variable_comparison,
)

__all__ = [
    # Core comparison
    "compare_records",
    "load_pe_values",
    "load_cosilico_values",
    "run_variable_comparison",
    "run_full_comparison",
    "generate_dashboard_json",
    # Aligned comparison (common dataset)
    "CommonDataset",
    "ComparisonResult",
    "load_common_dataset",
    "compare_variable",
    "run_aligned_comparison",
]
