"""Claude Agent SDK harness integration.

Self-registers a `ClaudeAgentSdkHarness` instance under id
`"claude_agent_sdk"` at import time. Lazy-imports `claude_agent_sdk` so the
package loads on machines without the SDK installed; only fails when
`run()` actually executes.
"""

from __future__ import annotations

from compilagent.harness.registry import harness_registry

from .harness import ClaudeAgentSdkHarness

if "claude_agent_sdk" not in harness_registry.ids():
    harness_registry.register("claude_agent_sdk", ClaudeAgentSdkHarness)

__all__ = ["ClaudeAgentSdkHarness"]
