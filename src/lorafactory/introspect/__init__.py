"""Extended weight introspection: what a trained LoRA actually did to the base.

CPU-only, checkpoint-only, offline. Everything here reads a `.safetensors`
adapter and reports per-module spectra — Frobenius norm, effective rank, the
top singular values, and (SD3, with a base checkpoint) how many of the adapter's
directions fall outside the base weight's dominant subspace.

    from lorafactory.introspect import introspect_checkpoint, write_csv
    report = introspect_checkpoint("adapter.safetensors", k=10)
    write_csv(report, "introspect.csv", adapter_id="L-A")

ΔW = B @ A is never materialised anywhere in this package; see `linalg.py`.
"""

from lorafactory.introspect.base_cache import (
    SD3_BASE_KEY_MAP,
    BaseKeyRef,
    BaseSubspaceCache,
    StaleSubspaceCacheError,
    base_fingerprint,
    base_key_for,
    load_base_weight,
)
from lorafactory.introspect.linalg import (
    LoraSVD,
    effective_rank,
    intruder_stats,
    lora_frobenius_norm,
    lora_singular_values,
)
from lorafactory.introspect.report import (
    ADAPTER_DIRECTIONS,
    IntrospectionReport,
    ModuleStats,
    Settings,
    block_of,
    csv_header,
    detect_lora_layout,
    introspect_checkpoint,
    module_stats,
    write_config_json,
    write_csv,
)

__all__ = [
    "ADAPTER_DIRECTIONS",
    "BaseKeyRef",
    "BaseSubspaceCache",
    "IntrospectionReport",
    "LoraSVD",
    "ModuleStats",
    "SD3_BASE_KEY_MAP",
    "Settings",
    "StaleSubspaceCacheError",
    "base_fingerprint",
    "base_key_for",
    "block_of",
    "csv_header",
    "detect_lora_layout",
    "effective_rank",
    "introspect_checkpoint",
    "intruder_stats",
    "load_base_weight",
    "lora_frobenius_norm",
    "lora_singular_values",
    "module_stats",
    "write_config_json",
    "write_csv",
]
