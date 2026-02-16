from __future__ import annotations

"""Deprecated shim.

Architecture-search logging helpers were consolidated into `src.experiment_logging`.
This module remains as a compatibility import path.
"""

from src.experiment_logging import (  # noqa: F401
    ARCH_SEARCH_COLUMNS as CSV_COLUMNS,
    append_arch_search_result as append_result,
    compare_arch_to_best as compare_to_best,
    ensure_arch_search_csv as ensure_csv,
    load_best_arch_result as load_best,
)

