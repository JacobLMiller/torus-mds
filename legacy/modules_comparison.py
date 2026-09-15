from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Literal, Sequence

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from standalone_toruslayout.toruslayout import TorusLayoutResult, wrap_python, wrap_ts

from . import geometry, metrics, visualization
from .projector import MDSTorusProjector

StandaloneBackend = Literal["wrap_python", "wrap_ts"]
METRIC_COLUMNS = ("stress", "distortion", "goodness", "NP", "alpha")
LOWER_IS_BETTER = {"alpha": False, "stress": True, "distortion": True, "goodness": False, "NP": False, "runtime": True}


@dataclass
class GraphComparisonSpec:
    name: str
    graph: nx.Graph
    distances: np.ndarray
    group: str = "default"
    colors: Any = None
    plot_kwargs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LayoutComparisonResult:
    spec: GraphComparisonSpec
    standalone_backend: StandaloneBackend
    standalone_layout: TorusLayoutResult
    standalone_runtime_s: float
    standalone_metrics: dict[str, float]
    mdstorus_projector: MDSTorusProjector
    mdstorus_layout: np.ndarray
    mdstorus_runtime_s: float
    mdstorus_metrics: dict[str, float]

    def to_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "graph": self.spec.name,
            "group": self.spec.group,
            "standalone_backend": self.standalone_backend,
            "nodes": int(self.spec.graph.number_of_nodes()),
            "edges": int(self.spec.graph.number_of_edges()),
            "iterations_standalone": int(self.standalone_layout.iterations),
            "runtime_standalone_s": float(self.standalone_runtime_s),
            "runtime_mdstorus_s": float(self.mdstorus_runtime_s),
        }
        for metric in METRIC_COLUMNS:
            row[f"{metric}_standalone"] = float(self.standalone_metrics[metric])
            row[f"{metric}_mdstorus"] = float(self.mdstorus_metrics[metric])
        row.update(self.spec.metadata)
        return row


def _scaled_torus_distance(alpha: float):
    return lambda p, q: alpha * geometry.torus_distance(p, q)


def _safe_percent(numerator: float, denominator: float) -> float:
    denominator = float(denominator)
    if denominator == 0.0 or not np.isfinite(denominator):
        return float("nan")
    return float(100.0 * float(numerator) / denominator)


def _compute_layout_metrics(X: np.ndarray, D: np.ndarray) -> dict[str, float]:
    alpha = metrics.estimate_alpha(X, D)
    geod = _scaled_torus_distance(alpha)
    return {
        "alpha": float(alpha),
        "stress": float(metrics.geodesic_stress(X, D, geod)),
        "distortion": float(metrics.geodesic_distortion(X, D, geod)),
        "goodness": float(metrics.SGS(X, D, geod)),
        "NP": float(metrics.geodesic_NP(X, D, geod)),
    }


def _run_standalone_layout(
    spec: GraphComparisonSpec,
    *,
    backend: StandaloneBackend,
    standalone_kwargs: dict[str, Any] | None,
) -> tuple[TorusLayoutResult, float]:
    kwargs = dict(standalone_kwargs or {})
    t0 = time.perf_counter()
    if backend == "wrap_python":
        kwargs.pop("node_memory_mb", None)
        layout = wrap_python(spec.graph, shortest_path_lengths=spec.distances, **kwargs)
    elif backend == "wrap_ts":
        layout = wrap_ts(spec.graph, shortest_path_lengths=spec.distances, **kwargs)
    else:
        raise ValueError(f"Unknown standalone backend: {backend}")
    return layout, time.perf_counter() - t0


def run_standalone_vs_mdstorus(
    spec: GraphComparisonSpec,
    *,
    standalone_backend: StandaloneBackend = "wrap_python",
    standalone_kwargs: dict[str, Any] | None = None,
    mdstorus_kwargs: dict[str, Any] | None = None,
) -> LayoutComparisonResult:
    standalone_layout, standalone_runtime_s = _run_standalone_layout(spec, backend=standalone_backend, standalone_kwargs=standalone_kwargs)
    standalone_metrics = _compute_layout_metrics(standalone_layout.positions, spec.distances)

    projector = MDSTorusProjector()
    t0 = time.perf_counter()
    mdstorus_layout = projector.fit_transform(spec.distances, **(mdstorus_kwargs or {}))
    mdstorus_runtime_s = time.perf_counter() - t0
    mdstorus_metrics = _compute_layout_metrics(mdstorus_layout, spec.distances)

    return LayoutComparisonResult(
        spec=spec,
        standalone_backend=standalone_backend,
        standalone_layout=standalone_layout,
        standalone_runtime_s=standalone_runtime_s,
        standalone_metrics=standalone_metrics,
        mdstorus_projector=projector,
        mdstorus_layout=mdstorus_layout,
        mdstorus_runtime_s=mdstorus_runtime_s,
        mdstorus_metrics=mdstorus_metrics,
    )


def _plot_layout_pair(
    result: LayoutComparisonResult,
    axes: np.ndarray,
    *,
    standalone_label: str,
    mdstorus_label: str,
    edge_alpha: float,
) -> None:
    plot_kwargs = {"edge_alpha": edge_alpha}
    plot_kwargs.update(result.spec.plot_kwargs)
    visualization.plot_embedding_with_torus_edges(
        result.standalone_layout.positions,
        result.spec.graph,
        ax=axes[0],
        colors=result.spec.colors,
        **plot_kwargs,
    )
    visualization.plot_embedding_with_torus_edges(
        result.mdstorus_layout,
        result.spec.graph,
        ax=axes[1],
        colors=result.spec.colors,
        **plot_kwargs,
    )
    axes[0].invert_yaxis()
    axes[1].invert_yaxis() # flip standalone layout to match website, not really necessary for MDS torus layout
    axes[0].set_title(f"{result.spec.name} - {standalone_label} ({result.standalone_runtime_s:.3f}s)")
    axes[1].set_title(f"{result.spec.name} - {mdstorus_label} ({result.mdstorus_runtime_s:.3f}s)")


def compare_standalone_vs_mdstorus_graphs(
    specs: Sequence[GraphComparisonSpec],
    *,
    standalone_backend: StandaloneBackend = "wrap_python",
    standalone_kwargs: dict[str, Any] | None = None,
    mdstorus_kwargs: dict[str, Any] | None = None,
    edge_alpha: float = 0.5,
    figsize_per_row: tuple[float, float] = (10.0, 4.0),
) -> tuple[plt.Figure, pd.DataFrame] | tuple[plt.Figure, pd.DataFrame, list[LayoutComparisonResult]]:
    if not specs:
        raise ValueError("specs must contain at least one graph")

    results = [
        run_standalone_vs_mdstorus(
            spec,
            standalone_backend=standalone_backend,
            standalone_kwargs=standalone_kwargs,
            mdstorus_kwargs=mdstorus_kwargs,
        )
        for spec in specs
    ]

    fig, axes = plt.subplots(
        len(results),
        2,
        figsize=(figsize_per_row[0], figsize_per_row[1] * len(results)),
        squeeze=False,
    )
    for row_axes, result in zip(axes, results):
        _plot_layout_pair(
            result,
            row_axes,
            standalone_label=standalone_backend,
            mdstorus_label="MDSTorusProjector",
            edge_alpha=edge_alpha,
        )
    fig.tight_layout()

    df = pd.DataFrame([result.to_row() for result in results])
    return fig, df, results


def summarize_standalone_vs_mdstorus(
    df: pd.DataFrame,
    *,
    groupby: str | None = "group",
) -> dict[str, Any]:
    analysis_df = df.copy()
    summary_rows: list[dict[str, Any]] = []

    for metric in (*METRIC_COLUMNS, "runtime"):
        left_col = f"{metric}_standalone" if metric != "runtime" else "runtime_standalone_s"
        right_col = f"{metric}_mdstorus" if metric != "runtime" else "runtime_mdstorus_s"
        if left_col not in analysis_df.columns or right_col not in analysis_df.columns:
            continue

        delta_col = f"{metric}_delta"
        pct_col = f"{metric}_delta_pct"
        analysis_df[delta_col] = analysis_df[left_col] - analysis_df[right_col]
        denom = analysis_df[right_col].replace(0, np.nan)
        analysis_df[pct_col] = 100.0 * analysis_df[delta_col] / denom

        mean_standalone = float(analysis_df[left_col].mean())
        mean_mdstorus = float(analysis_df[right_col].mean())
        mean_delta = float(analysis_df[delta_col].mean())
        median_mdstorus = float(analysis_df[right_col].median())
        median_delta = float(analysis_df[delta_col].median())
        mean_abs_delta = float(analysis_df[delta_col].abs().mean())

        if LOWER_IS_BETTER[metric]:
            better_mask = analysis_df[left_col] < analysis_df[right_col]
            worse_mask = analysis_df[left_col] > analysis_df[right_col]
        else:
            better_mask = analysis_df[left_col] > analysis_df[right_col]
            worse_mask = analysis_df[left_col] < analysis_df[right_col]
        tie_mask = ~(better_mask | worse_mask)

        summary_rows.append(
            {
                "metric": metric,
                "mean_standalone": mean_standalone,
                "mean_mdstorus": mean_mdstorus,
                "mean_delta": mean_delta,
                "mean_delta_pct": _safe_percent(mean_delta, mean_mdstorus),
                "median_delta": median_delta,
                "median_delta_pct": _safe_percent(median_delta, median_mdstorus),
                "mean_abs_delta": mean_abs_delta,
                "mean_abs_delta_pct": _safe_percent(mean_abs_delta, abs(mean_mdstorus)),
                "standalone_better_count": int(better_mask.sum()),
                "mdstorus_better_count": int(worse_mask.sum()),
                "tie_count": int(tie_mask.sum()),
            }
        )

    overall_summary = pd.DataFrame(summary_rows)

    group_summary = pd.DataFrame()
    if groupby is not None and groupby in analysis_df.columns:
        aggregated_rows: list[dict[str, Any]] = []
        for group_value, group_df in analysis_df.groupby(groupby, dropna=False):
            row: dict[str, Any] = {
                groupby: group_value,
                "graphs": int(len(group_df)),
            }
            for metric in (*METRIC_COLUMNS, "runtime"):
                left_col = f"{metric}_standalone" if metric != "runtime" else "runtime_standalone_s"
                right_col = f"{metric}_mdstorus" if metric != "runtime" else "runtime_mdstorus_s"
                if left_col not in group_df.columns or right_col not in group_df.columns:
                    continue
                delta = group_df[left_col] - group_df[right_col]
                mean_left = float(group_df[left_col].mean())
                mean_right = float(group_df[right_col].mean())
                mean_delta = float(delta.mean())
                median_right = float(group_df[right_col].median())
                median_delta = float(delta.median())
                row[f"mean_{left_col}"] = mean_left
                row[f"mean_{right_col}"] = mean_right
                row[f"mean_{metric}_delta"] = mean_delta
                row[f"mean_{metric}_delta_pct"] = _safe_percent(mean_delta, mean_right)
                row[f"median_{metric}_delta_pct"] = _safe_percent(median_delta, median_right)
            aggregated_rows.append(row)
        group_summary = pd.DataFrame(aggregated_rows)

    report_lines = [
        f"Graphs compared: {len(analysis_df)}",
    ]
    if "standalone_backend" in analysis_df.columns and not analysis_df.empty:
        backends = ", ".join(sorted(map(str, analysis_df["standalone_backend"].unique())))
        report_lines.append(f"Standalone backend: {backends}")
    if not overall_summary.empty:
        for _, row in overall_summary.iterrows():
            report_lines.append(
                f"{row['metric']}: standalone={row['mean_standalone']:.4f}, "
                f"mdstorus={row['mean_mdstorus']:.4f}, delta={row['mean_delta']:.4f} "
                f"({row['mean_delta_pct']:.2f}%)"
            )

    return {
        "analysis": analysis_df,
        "overall_summary": overall_summary,
        "group_summary": group_summary,
        "report": "\n".join(report_lines),
    }
