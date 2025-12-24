"""Utility functions for benchmarks."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, stdev
from typing import TYPE_CHECKING, Callable


if TYPE_CHECKING:
    from optyx.problem import Problem
    from optyx.solution import Solution


# Results directory
RESULTS_DIR = Path(__file__).parent / "results"


@dataclass
class TimingResult:
    """Result of a timing benchmark."""

    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    n_runs: int

    def __str__(self) -> str:
        return f"{self.mean_ms:.3f} ± {self.std_ms:.3f} ms (n={self.n_runs})"


@dataclass
class ScalingData:
    """Data for scaling analysis across problem sizes."""

    sizes: list[int] = field(default_factory=list)
    optyx_times: list[float] = field(default_factory=list)
    optyx_stds: list[float] = field(default_factory=list)
    scipy_times: list[float] = field(default_factory=list)
    scipy_stds: list[float] = field(default_factory=list)
    label: str = ""

    def add_point(
        self,
        n: int,
        optyx_ms: float,
        optyx_std: float,
        scipy_ms: float = 0.0,
        scipy_std: float = 0.0,
    ) -> None:
        """Add a data point."""
        self.sizes.append(n)
        self.optyx_times.append(optyx_ms)
        self.optyx_stds.append(optyx_std)
        self.scipy_times.append(scipy_ms)
        self.scipy_stds.append(scipy_std)

    def overhead_ratios(self) -> list[float]:
        """Compute overhead ratios at each size."""
        return [
            o / s if s > 0 else float("inf")
            for o, s in zip(self.optyx_times, self.scipy_times)
        ]


def time_function(
    func: Callable[[], object],
    n_warmup: int = 2,
    n_runs: int = 10,
) -> TimingResult:
    """Time a function with warmup and multiple runs.

    Args:
        func: Function to time (should take no arguments).
        n_warmup: Number of warmup calls (not timed).
        n_runs: Number of timed runs.

    Returns:
        TimingResult with statistics.
    """
    # Warmup
    for _ in range(n_warmup):
        func()

    # Timed runs
    times_ms: list[float] = []
    for _ in range(n_runs):
        start = time.perf_counter()
        func()
        elapsed = (time.perf_counter() - start) * 1000
        times_ms.append(elapsed)

    return TimingResult(
        mean_ms=mean(times_ms),
        std_ms=stdev(times_ms) if len(times_ms) > 1 else 0.0,
        min_ms=min(times_ms),
        max_ms=max(times_ms),
        n_runs=n_runs,
    )


def time_solve(
    problem: Problem,
    method: str = "auto",
    n_warmup: int = 2,
    n_runs: int = 10,
    **kwargs,
) -> tuple[TimingResult, Solution]:
    """Time problem.solve() with statistics.

    Args:
        problem: Problem to solve.
        method: Solver method.
        n_warmup: Number of warmup solves.
        n_runs: Number of timed solves.
        **kwargs: Additional solver arguments.

    Returns:
        Tuple of (TimingResult, last Solution).
    """
    solution = None

    def solve_once():
        nonlocal solution
        solution = problem.solve(method=method, **kwargs)

    timing = time_function(solve_once, n_warmup=n_warmup, n_runs=n_runs)
    assert solution is not None
    return timing, solution


@dataclass
class ComparisonResult:
    """Result comparing two approaches."""

    optyx_timing: TimingResult
    baseline_timing: TimingResult
    overhead_ratio: float

    def __str__(self) -> str:
        return (
            f"Optyx: {self.optyx_timing}\n"
            f"Baseline: {self.baseline_timing}\n"
            f"Overhead: {self.overhead_ratio:.2f}x"
        )


def compare_timing(
    optyx_func: Callable[[], object],
    baseline_func: Callable[[], object],
    n_warmup: int = 2,
    n_runs: int = 10,
) -> ComparisonResult:
    """Compare timing of Optyx vs baseline.

    Args:
        optyx_func: Optyx implementation.
        baseline_func: Baseline implementation.
        n_warmup: Number of warmup calls.
        n_runs: Number of timed runs.

    Returns:
        ComparisonResult with overhead ratio.
    """
    optyx_timing = time_function(optyx_func, n_warmup=n_warmup, n_runs=n_runs)
    baseline_timing = time_function(baseline_func, n_warmup=n_warmup, n_runs=n_runs)

    overhead = (
        optyx_timing.mean_ms / baseline_timing.mean_ms
        if baseline_timing.mean_ms > 0
        else float("inf")
    )

    return ComparisonResult(
        optyx_timing=optyx_timing,
        baseline_timing=baseline_timing,
        overhead_ratio=overhead,
    )


def check_solution_accuracy(
    solution: Solution,
    expected_values: dict[str, float],
    expected_objective: float,
    value_tol: float = 1e-4,
    objective_tol: float = 1e-4,
) -> tuple[bool, str]:
    """Check if solution matches expected values.

    Args:
        solution: Optyx solution to check.
        expected_values: Expected variable values.
        expected_objective: Expected objective value.
        value_tol: Tolerance for variable values.
        objective_tol: Tolerance for objective value.

    Returns:
        Tuple of (is_accurate, message).
    """
    if not solution.is_optimal:
        return False, f"Solution not optimal: {solution.status}"

    # Check objective
    obj_error = abs(solution.objective_value - expected_objective)
    if obj_error > objective_tol:
        return (
            False,
            f"Objective error: {obj_error:.2e} (expected {expected_objective}, got {solution.objective_value})",
        )

    # Check variable values
    for var_name, expected in expected_values.items():
        actual = solution.values.get(var_name)
        if actual is None:
            return False, f"Variable {var_name} not in solution"
        error = abs(actual - expected)
        if error > value_tol:
            return (
                False,
                f"Variable {var_name} error: {error:.2e} (expected {expected}, got {actual})",
            )

    return True, "OK"


def skip_if_missing(module_name: str):
    """Decorator to skip test if module is not installed."""
    import pytest

    try:
        __import__(module_name)
        return lambda f: f
    except ImportError:
        return pytest.mark.skip(reason=f"{module_name} not installed")


# =============================================================================
# Plotting utilities
# =============================================================================


def plot_scaling_comparison(
    data: ScalingData,
    title: str = "Optyx vs SciPy Scaling",
    save_path: Path | str | None = None,
    show: bool = False,
) -> None:
    """Plot scaling comparison between Optyx and SciPy.

    Args:
        data: ScalingData with timing results.
        title: Plot title.
        save_path: Path to save the plot (optional).
        show: Whether to display the plot.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Absolute times
    ax1 = axes[0]
    ax1.errorbar(
        data.sizes,
        data.optyx_times,
        yerr=data.optyx_stds,
        marker="o",
        capsize=3,
        label="Optyx",
        linewidth=2,
    )
    if any(t > 0 for t in data.scipy_times):
        ax1.errorbar(
            data.sizes,
            data.scipy_times,
            yerr=data.scipy_stds,
            marker="s",
            capsize=3,
            label="SciPy",
            linewidth=2,
        )
    ax1.set_xlabel("Problem Size (n)", fontsize=11)
    ax1.set_ylabel("Time (ms)", fontsize=11)
    ax1.set_title(f"{title} - Absolute Time", fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale("log")
    ax1.set_yscale("log")

    # Right: Overhead ratio
    ax2 = axes[1]
    if any(t > 0 for t in data.scipy_times):
        ratios = data.overhead_ratios()
        ax2.plot(data.sizes, ratios, marker="o", linewidth=2, color="green")
        ax2.axhline(y=1.0, color="red", linestyle="--", label="Parity (1x)")
        ax2.axhline(
            y=2.0, color="orange", linestyle="--", alpha=0.5, label="2x overhead"
        )
        ax2.set_xlabel("Problem Size (n)", fontsize=11)
        ax2.set_ylabel("Overhead Ratio (Optyx / SciPy)", fontsize=11)
        ax2.set_title(f"{title} - Overhead Ratio", fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale("log")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_cache_benefit(
    sizes: list[int],
    cold_times: list[float],
    warm_times: list[float],
    title: str = "Cache Benefit Analysis",
    save_path: Path | str | None = None,
    show: bool = False,
) -> None:
    """Plot cache benefit (cold vs warm solve times).

    Args:
        sizes: Problem sizes.
        cold_times: First solve times (cache miss).
        warm_times: Subsequent solve times (cache hit).
        title: Plot title.
        save_path: Path to save the plot.
        show: Whether to display the plot.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Times comparison
    ax1 = axes[0]
    ax1.plot(sizes, cold_times, marker="o", label="Cold (first solve)", linewidth=2)
    ax1.plot(sizes, warm_times, marker="s", label="Warm (cached)", linewidth=2)
    ax1.set_xlabel("Problem Size (n)", fontsize=11)
    ax1.set_ylabel("Time (ms)", fontsize=11)
    ax1.set_title(f"{title} - Solve Times", fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale("log")
    ax1.set_yscale("log")

    # Right: Speedup
    ax2 = axes[1]
    speedups = [c / w if w > 0 else 0 for c, w in zip(cold_times, warm_times)]
    ax2.bar(range(len(sizes)), speedups, color="green", alpha=0.7)
    ax2.set_xticks(range(len(sizes)))
    ax2.set_xticklabels([str(s) for s in sizes])
    ax2.set_xlabel("Problem Size (n)", fontsize=11)
    ax2.set_ylabel("Speedup (Cold / Warm)", fontsize=11)
    ax2.set_title(f"{title} - Cache Speedup", fontsize=12)
    ax2.axhline(y=1.0, color="red", linestyle="--")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_overhead_breakdown(
    categories: list[str],
    overheads: list[float],
    title: str = "Overhead by Problem Type",
    save_path: Path | str | None = None,
    show: bool = False,
) -> None:
    """Plot overhead breakdown by category.

    Args:
        categories: Category names.
        overheads: Overhead ratios for each category.
        title: Plot title.
        save_path: Path to save the plot.
        show: Whether to display the plot.
    """
    import matplotlib.pyplot as plt

    colors = ["green" if o < 1.5 else "orange" if o < 2.0 else "red" for o in overheads]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(categories, overheads, color=colors, alpha=0.7)
    ax.axvline(x=1.0, color="green", linestyle="--", linewidth=2, label="Parity")
    ax.axvline(x=2.0, color="red", linestyle="--", linewidth=2, label="2x target")
    ax.set_xlabel("Overhead Ratio (Optyx / SciPy)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="x")

    # Add value labels
    for bar, overhead in zip(bars, overheads):
        ax.text(
            bar.get_width() + 0.05,
            bar.get_y() + bar.get_height() / 2,
            f"{overhead:.2f}x",
            va="center",
            fontsize=10,
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_multi_scaling(
    datasets: list[ScalingData],
    title: str = "Scaling Comparison",
    save_path: Path | str | None = None,
    show: bool = False,
) -> None:
    """Plot multiple scaling datasets on one chart.

    Args:
        datasets: List of ScalingData objects.
        title: Plot title.
        save_path: Path to save the plot.
        show: Whether to display the plot.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    markers = ["o", "s", "^", "D", "v", "<", ">", "p"]

    for i, data in enumerate(datasets):
        marker = markers[i % len(markers)]
        ax.errorbar(
            data.sizes,
            data.optyx_times,
            yerr=data.optyx_stds,
            marker=marker,
            capsize=3,
            label=data.label,
            linewidth=2,
        )

    ax.set_xlabel("Problem Size (n)", fontsize=11)
    ax.set_ylabel("Time (ms)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()
