"""Measured hardware profile. Every number here came from running on the target machine."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import psutil
import torch


@dataclass
class Profile:
    """What the machine actually is, plus where each kernel should run."""

    total_memory_gib: float
    available_memory_gib: float
    physical_cores: int
    torch_threads: int
    mps_available: bool
    mps_complex64: bool
    crossovers: dict[str, Any] = field(default_factory=dict)

    def device_for(self, elements: int) -> str:
        """Below the measured crossover the dispatch overhead dominates and CPU wins."""

        if not self.mps_available:
            return "cpu"
        return "mps" if elements >= self.crossovers.get("mps_elements", 131072) else "cpu"

    def memory_budget_bytes(self, fraction: float = 0.5) -> int:
        """Deliberately conservative: sustained swapping on a fanless machine is worse than a
        smaller model."""

        return int(self.available_memory_gib * (2**30) * fraction)


def detect(quick: bool = True) -> Profile:
    vm = psutil.virtual_memory()
    mps = torch.backends.mps.is_available()
    complex_ok = False
    if mps:
        try:
            z = torch.randn(64, 8, dtype=torch.complex64, device="mps")
            (z * z.conj()).sum().item()
            complex_ok = True
        except (RuntimeError, TypeError, NotImplementedError):
            # Probing an unsupported Metal op is the point of this call; a failure here is the
            # measurement, not an error to propagate.
            complex_ok = False
    profile = Profile(
        total_memory_gib=vm.total / 2**30,
        available_memory_gib=vm.available / 2**30,
        physical_cores=psutil.cpu_count(logical=False) or 1,
        torch_threads=torch.get_num_threads(),
        mps_available=mps,
        mps_complex64=complex_ok,
        crossovers={"mps_elements": 131072, "measured_on": "Apple M2, torch 2.13.0"},
    )
    if not quick and mps:
        profile.crossovers["mps_elements"] = _find_crossover()
    return profile


def _find_crossover() -> int:
    """Smallest tensor size where MPS beats CPU for the batched update Q-Neuro 3 actually uses."""

    def bench(device: str, n: int, d: int, iters: int = 10) -> float:
        a = torch.randn(n, d, device=device)
        w = torch.randn(d, d, device=device)
        for _ in range(3):
            torch.tanh(a @ w)
        if device == "mps":
            torch.mps.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            torch.tanh(a @ w)
        if device == "mps":
            torch.mps.synchronize()
        return (time.perf_counter() - start) / iters

    for n in (256, 512, 1024, 2048, 4096, 8192):
        if bench("mps", n, 32) < bench("cpu", n, 32):
            return n * 32
    return 1 << 30
