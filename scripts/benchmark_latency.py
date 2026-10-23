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
from utils.config import Config


def benchmark_device(
    model: torch.nn.Module,
    device: torch.device,
    iterations: int = 1000,
    warmup: int = 100,
    seq_len: int = Config.WINDOW_SIZE,
    n_features: int = Config.N_FEATURES,
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


def benchmark_ensemble_device(
    models: list[torch.nn.Module],
    device: torch.device,
    iterations: int = 500,
    warmup: int = 50,
    seq_len: int = Config.WINDOW_SIZE,
    n_features: int = Config.N_FEATURES,
) -> dict:
    """Benchmark full soft-voting pass across all ensemble models."""
    for m in models:
        m.to(device)
        m.eval()

    x = torch.randn(1, seq_len, n_features, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            for m in models:
                _ = m(x)
            if device.type == "cuda":
                torch.cuda.synchronize()

    timings = []
    with torch.no_grad():
        for _ in range(iterations):
            t0 = time.perf_counter()
            for m in models:
                _ = m(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000.0)

    timings = np.array(timings)
    return {
        "mean_ms": round(float(np.mean(timings)), 3),
        "median_ms": round(float(np.median(timings)), 3),
        "p95_ms": round(float(np.percentile(timings, 95)), 3),
        "fps": round(1000.0 / float(np.mean(timings)), 1),
    }


def run_benchmarks(
    architectures: list[str] = ["bigru", "tcn", "cnn_gru"],
    output_path: Path = _PROJECT_ROOT / "logs" / "phase4" / "latency_benchmark.json",
    iterations: int = 1000,
) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = {}

    print("=" * 80)
    print("ALERT - Edge Architecture Latency & Parameter Benchmark")
    print(f"Input specification: B=1, T={Config.WINDOW_SIZE} frames, F={Config.N_FEATURES} kinematic channels")
    print("=" * 80)

    header = f"{'Architecture':<15} {'Params':>10} {'Disk (MB)':>11} {'CPU Mean (ms)':>15} {'CPU FPS':>10} {'GPU Mean (ms)':>15} {'Status':>10}"
    print(header)
    print("-" * len(header))

    cuda_available = torch.cuda.is_available()

    models_for_ensemble = []
    for arch in architectures:
        model = build_model(arch=arch)
        models_for_ensemble.append(model)
        params = model.count_parameters()
        disk_mb = round((params * 4) / (1024 * 1024), 3)

        # CPU Benchmark
        cpu_metrics = benchmark_device(model, torch.device("cpu"), iterations=iterations)

        # GPU Benchmark
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

    # Benchmark 12-model ensemble (4 replicates of each arch)
    full_ensemble_suite = models_for_ensemble * 4
    total_ensemble_params = sum(m.count_parameters() for m in full_ensemble_suite)
    ens_cpu = benchmark_ensemble_device(full_ensemble_suite, torch.device("cpu"), iterations=300)
    ens_gpu = benchmark_ensemble_device(full_ensemble_suite, torch.device("cuda"), iterations=300) if cuda_available else {"mean_ms": 0.0, "fps": 0.0}

    ens_compliant = ens_cpu["mean_ms"] <= 30.0
    ens_status = "COMPLIANT" if ens_compliant else "NON-COMPLIANT"

    print("-" * len(header))
    print(
        f"{'12-ENSEMBLE':<15} {total_ensemble_params:>10,d} {total_ensemble_params*4/(1024*1024):>11.2f} "
        f"{ens_cpu['mean_ms']:>15.3f} {ens_cpu['fps']:>10.1f} "
        f"{ens_gpu['mean_ms']:>15.3f} {ens_status:>10}"
    )

    results["12_model_ensemble"] = {
        "total_parameters": total_ensemble_params,
        "cpu_benchmark": ens_cpu,
        "gpu_benchmark": ens_gpu,
        "edge_compliant": ens_compliant,
    }

    print("-" * len(header))
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\n[Saved Artifact] Benchmark report saved to: {output_path}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    run_benchmarks()
