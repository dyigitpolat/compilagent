"""Inter-process GPU lease pool for episode-parallel sandbox runs.

Episodes are LLM-latency-dominated: with one pinned worker per GPU the
devices idle at ~0-3% utilization while workers wait on chat completions.
The only GPU-bound unit of work is the `triton_source` sandbox subprocess
(it contains compile + correctness gates + CUDA-event timing of candidate
AND reference), so timing validity requires exactly one invariant: no two
sandboxes share a device simultaneously. This module enforces that
invariant ACROSS PROCESSES, which lets N >> #GPUs episode workers run
concurrently and lease a device only for the sandbox's lifetime.

  - Pool membership: the ``COMPILAGENT_GPU_POOL`` env var — comma-separated
    *physical* CUDA device indices, e.g. ``"1,2,3"``. Unset/empty → no pool
    (callers fall back to the inherited ``CUDA_VISIBLE_DEVICES`` pinning).
  - One lock file per device at
    ``${COMPILAGENT_GPU_LOCK_DIR:-/tmp/compilagent_gpu_locks}/gpu<idx>.lock``,
    held via ``fcntl.flock(LOCK_EX | LOCK_NB)``.
  - Crash safety: the kernel releases an flock automatically when its fd is
    closed — which includes process death by ANY path (crash, OOM-kill,
    SIGKILL) — so a dead leaseholder can never wedge the pool and no
    stale-lock cleanup is ever needed.
  - Fairness (non-strict, by design): `acquire` sweeps the pool round-robin
    with a small jittered sleep between sweeps. flock has no FIFO queue, so
    a long waiter can lose a freed device to a newer arrival. That is fair
    enough here: sandbox runs are short relative to episode wallclock, and
    the jitter de-synchronizes contenders so nobody starves in practice.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import random
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path

POOL_ENV = "COMPILAGENT_GPU_POOL"
LOCK_DIR_ENV = "COMPILAGENT_GPU_LOCK_DIR"
BUSY_CHECK_ENV = "COMPILAGENT_GPU_BUSY_CHECK"
DEFAULT_LOCK_DIR = "/tmp/compilagent_gpu_locks"  # noqa: S108 (deliberate)

#: Jittered pause between full sweeps of a busy pool (seconds).
_SWEEP_SLEEP_RANGE_S = (0.05, 0.15)


def _busy_check_enabled() -> bool:
    return os.environ.get(BUSY_CHECK_ENV, "1").strip() != "0"


def foreign_occupants(device: str) -> tuple[str, ...]:
    """PIDs of compute processes currently resident on `device`.

    Our own sandboxes only ever run while their parent holds the device's
    flock, so any compute process found on a device whose flock was just
    acquired belongs to a FOREIGN job (another framework, a manual run, a
    training job). Leasing such a device would corrupt both jobs' timings —
    "only empty GPUs are allocatable" is therefore part of the lease
    contract, not a scheduling convention.

    Query failures (no nvidia-smi, driver hiccup) return ``()`` — the check
    degrades to flock-only rather than wedging the pool. Disable explicitly
    with ``COMPILAGENT_GPU_BUSY_CHECK=0``.
    """

    try:
        out = subprocess.run(  # noqa: S603, S607
            [
                "nvidia-smi",
                "--query-compute-apps=pid",
                "--format=csv,noheader",
                "-i",
                str(device),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:  # noqa: BLE001 — fail-open is this check's contract
        return ()
    if out.returncode != 0:
        return ()
    # Only numeric PID lines count: nvidia-smi mixes warnings into stdout on
    # some drivers, and anything non-PID-shaped must not mark a device busy
    # forever (fail-open is this check's contract).
    return tuple(
        line.strip()
        for line in out.stdout.splitlines()
        if line.strip().isdigit()
    )


class GpuLeaseTimeoutError(TimeoutError):
    """No pool device could be leased within the caller's timeout."""


class GpuLease:
    """One exclusively leased device.

    Context manager: `.device` is the physical CUDA index (string, as it
    appears in the pool), `.wait_seconds` is how long `acquire` waited.
    Exit (or `release()`) closes the lock fd, which releases the flock; the
    kernel does the same on process death, so a crashed holder frees its
    device automatically.
    """

    def __init__(self, device: str, fd: int, wait_seconds: float) -> None:
        self.device = device
        self.wait_seconds = wait_seconds
        self._fd: int | None = fd

    def release(self) -> None:
        fd, self._fd = self._fd, None
        if fd is None:
            return
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)

    def __enter__(self) -> GpuLease:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()


def pool_devices() -> tuple[str, ...]:
    """Devices in the configured pool (empty tuple → pool mode is off)."""

    raw = os.environ.get(POOL_ENV, "") or ""
    return tuple(d.strip() for d in raw.split(",") if d.strip())


def _lock_dir() -> Path:
    return Path(os.environ.get(LOCK_DIR_ENV) or DEFAULT_LOCK_DIR)


def _try_lock(directory: Path, device: str) -> int | None:
    """Non-blocking flock attempt on one device's lock file; fd or None."""

    fd = os.open(directory / f"gpu{device}.lock", os.O_RDWR | os.O_CREAT, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return None
    return fd


def acquire(
    timeout: float | None,
    *,
    devices: Sequence[str] | None = None,
) -> GpuLease:
    """Lease one device from the pool, waiting up to `timeout` seconds.

    `timeout=None` waits indefinitely. `devices` overrides the
    ``COMPILAGENT_GPU_POOL`` env var (used by tests). Raises
    `GpuLeaseTimeoutError` once the total wait exceeds `timeout`, and
    `ValueError` when no pool is configured at all.
    """

    pool = tuple(devices) if devices is not None else pool_devices()
    if not pool:
        raise ValueError(
            f"GPU lease pool is empty — set {POOL_ENV} to a comma-separated "
            'list of physical device indices (e.g. "1,2,3").'
        )
    directory = _lock_dir()
    directory.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    busy_check = _busy_check_enabled()
    foreign_seen: dict[str, tuple[str, ...]] = {}
    # Occupancy is re-queried at most once per device per TTL within this
    # acquire() call: sweeps run every ~0.1s while nvidia-smi costs ~0.1-1s,
    # so an uncached check would dominate the sweep (and hammer the driver
    # under contention). The TTL race window is inherent to any
    # occupancy check and small next to a sandbox lifetime.
    occupancy_ttl_s = 1.0
    occupancy_cache: dict[str, tuple[float, tuple[str, ...]]] = {}

    def _occupants(device: str) -> tuple[str, ...]:
        now = time.monotonic()
        hit = occupancy_cache.get(device)
        if hit is not None and now - hit[0] < occupancy_ttl_s:
            return hit[1]
        result = foreign_occupants(device)
        occupancy_cache[device] = (now, result)
        return result
    # Random starting offset (advanced each sweep) de-synchronizes
    # contending processes so they don't all hammer the pool in the same
    # order; see the module docstring for the non-strict-fairness argument.
    offset = random.randrange(len(pool))
    while True:
        for i in range(len(pool)):
            device = pool[(offset + i) % len(pool)]
            fd = _try_lock(directory, device)
            if fd is None:
                continue
            if busy_check:
                occupants = _occupants(device)
                if occupants:
                    # Foreign compute process resident on an unleased device:
                    # NOT allocatable. Release the flock and keep sweeping —
                    # the device re-enters the pool when it empties.
                    foreign_seen[device] = occupants
                    with contextlib.suppress(OSError):
                        fcntl.flock(fd, fcntl.LOCK_UN)
                    os.close(fd)
                    continue
            return GpuLease(device, fd, time.monotonic() - started)
        waited = time.monotonic() - started
        if timeout is not None and waited >= timeout:
            foreign_note = (
                " Foreign compute processes were seen on: "
                + "; ".join(
                    f"gpu{d} (pids {', '.join(p)})"
                    for d, p in sorted(foreign_seen.items())
                )
                + f" — only empty GPUs are allocatable (override: {BUSY_CHECK_ENV}=0)."
                if foreign_seen
                else ""
            )
            raise GpuLeaseTimeoutError(
                f"no GPU lease acquired after {waited:.1f}s (pool "
                f"[{','.join(pool)}], timeout {timeout:.0f}s) — every device "
                "stayed busy; raise the lease timeout, reduce the episode "
                f"worker count, or grow {POOL_ENV}.{foreign_note}"
            )
        offset += 1
        time.sleep(random.uniform(*_SWEEP_SLEEP_RANGE_S))
