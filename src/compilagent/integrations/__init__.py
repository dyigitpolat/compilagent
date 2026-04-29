"""Phase-2 integration namespace.

Each integration lives as `compilagent.integrations.<name>` and self-registers
into the appropriate registry (`backend_registry`, `harness_registry`, or
`workload_registry`) at import time.

Phase-1 ships zero integrations. Adding one is purely additive — the core
never imports from this package.
"""
