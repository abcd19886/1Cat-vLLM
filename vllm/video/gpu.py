# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Coordinate H3 with other cooperating V100 jobs before workers allocate memory."""

import fcntl
import os
from pathlib import Path

LOCK_ROOT = Path("/tmp")


def _available_gpu_groups(tp: int) -> list[tuple[int, ...]]:
    import pynvml as nvml

    groups = []
    nvml.nvmlInit()
    try:
        for start in (0, 4):
            indices = tuple(range(start, start + tp))
            if indices[-1] >= nvml.nvmlDeviceGetCount():
                continue
            available = True
            for index in indices:
                handle = nvml.nvmlDeviceGetHandleByIndex(index)
                processes = nvml.nvmlDeviceGetComputeRunningProcesses(handle)
                # Ignore a small desktop CUDA allocation; never displace jobs.
                if any(p.usedGpuMemory > 256 * 1024**2 for p in processes):
                    available = False
                if nvml.nvmlDeviceGetMemoryInfo(handle).free < 30 * 1024**3:
                    available = False
            if available:
                groups.append(indices)
        return groups
    finally:
        nvml.nvmlShutdown()


def select_gpu_group(tp: int) -> tuple[int, ...]:
    """Read-only capacity probe; launchers should hold acquire_gpu_group instead."""
    groups = _available_gpu_groups(tp)
    if groups:
        return groups[0]
    raise RuntimeError("Neither configured GPU group has enough free memory")


class GPUGroupLease:
    def __init__(self, gpu_ids: tuple[int, ...]):
        self.gpu_ids = gpu_ids
        self._files = []
        names = [*(f"gpu{i}" for i in gpu_ids), "gpus" + "".join(map(str, gpu_ids))]
        try:
            for name in names:
                file = (LOCK_ROOT / f"1cat-vllm-v100-{name}.lock").open("a+")
                self._files.append(file)
                fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self.close()
            raise

    def close(self):
        for file in self._files:
            file.close()
        self._files.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def acquire_gpu_group(tp: int) -> GPUGroupLease:
    for indices in _available_gpu_groups(tp):
        try:
            lease = GPUGroupLease(indices)
        except BlockingIOError:
            continue
        try:
            # Recheck after locking: a capacity probe alone races other loaders.
            if indices not in _available_gpu_groups(tp):
                lease.close()
                continue
            for file in lease._files:
                file.seek(0)
                file.truncate()
                file.write(f"pid={os.getpid()} task=native-h3 gpus={indices}\n")
                file.flush()
            return lease
        except BaseException:
            lease.close()
            raise
    raise RuntimeError("No free, unleased H3 GPU group; existing jobs remain active")
