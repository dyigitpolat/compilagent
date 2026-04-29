"""Console entry point for `compilagent-acp`."""

from __future__ import annotations

from .server import run_acp_server


def main() -> None:
    run_acp_server()


if __name__ == "__main__":  # pragma: no cover
    main()
