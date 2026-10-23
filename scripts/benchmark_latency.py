"""
benchmark_latency.py
====================
Edge hardware compliance benchmark for candidate fatigue detection architectures.

Evaluates
---------
1. Trainable Parameter Count (< 250k edge budget)
2. Estimated Memory & Checkpoint Disk Footprint (< 2 MB)
3. Inference Latency (CPU single-stream, 1000 iterations, target <= 15-30ms)
4. Inference Latency (GPU Tensor Core, 1000 iterations, target <= 2-5ms)
5. Throughput (Frames Per Second)

Saves results to logs/phase2/latency_benchmark.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from core.model import build_model


def benchmark_device(
    model: torch.nn.Module,
    device: torch.device,
    iterations: int = 1000,
    warmup: int = 100,
    seq_len: int = 60,
    n_features: int = 8,
) -> dict:
    model = model.to(device)
    model.eval()

    # Single-window input: batch_size=1 representing real-time edge frame stream
    x = torch.randn(1, seq_len, n_features, device=device)

    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()

    # Timing iterations
    timings = []
    with torch.no_grad():
        for _ in range(iterations):
            t0 = time.perf_counter()
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000.0)  # milliseconds

    timings = np.array(timings)
    return {
        "mean_ms": round(float(np.mean(timings)), 3),
        "median_ms": round(float(np.median(timings)), 3),
        "p95_ms": round(float(np.percentile(timings, 95)), 3),
        "p99_ms": round(float(np.percentile(timings, 99)), 3),
        "min_ms": round(float(np.min(timings)), 3),
        "fps": round(1000.0 / float(np.mean(timings)), 1),
    }


def run_benchmarks(
    architectures: list[str] = ["bigru", "tcn", "cnn_gru"],
    output_path: Path = _PROJECT_ROOT / "logs" / "phase2" / "latency_benchmark.json",
    iterations: int = 1000,
) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = {}

    print("=" * 80)
    print("ALERT — Edge Architecture Latency & Parameter Benchmark")
    print(f"Iterations per test: {iterations:,} single-sample forward passes (B=1, T=60, F=8)")
    print("=" * 80)

    header = f"{'Architecture':<15} {'Params':>10} {'Disk (MB)':>11} {'CPU Mean (ms)':>15} {'CPU FPS':>10} {'GPU Mean (ms)':>15} {'Status':>10}"
    print(header)
    print("-" * len(header))

    cuda_available = torch.cuda.is_available()

    for arch in architectures:
        model = build_model(arch=arch)
        params = model.count_parameters()
        disk_mb = round((params * 4) / (1024 * 1024), 3)  # FP32 raw weights

        # CPU Benchmark
        cpu_metrics = benchmark_device(model, torch.device("cpu"), iterations=iterations)

        # GPU Benchmark (if available)
        gpu_metrics = (
            benchmark_device(model, torch.device("cuda"), iterations=iterations)
            if cuda_available
            else {"mean_ms": 0.0, "fps": 0.0}
        )

        is_edge_compliant = (
            params < 250_000
            and cpu_metrics["mean_ms"] <= 30.0
            and disk_mb < 2.0
        )
        status = "COMPLIANT" if is_edge_compliant else "NON-COMPLIANT"

        print(
            f"{arch.upper():<15} {params:>10,d} {disk_mb:>11.2f} "
            f"{cpu_metrics['mean_ms']:>15.3f} {cpu_metrics['fps']:>10.1f} "
            f"{gpu_metrics['mean_ms']:>15.3f} {status:>10}"
        )

        results[arch] = {
            "parameters": params,
            "estimated_disk_mb": disk_mb,
            "edge_compliant": is_edge_compliant,
            "cpu_benchmark": cpu_metrics,
            "gpu_benchmark": gpu_metrics,
        }

    print("-" * len(header))
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\n[Saved Artifact] Benchmark report saved to: {output_path}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    run_benchmarks()
