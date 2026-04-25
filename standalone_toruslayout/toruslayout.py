from __future__ import annotations

import json
import math
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np


WEBSITE_SVG_WIDTH = 1300.0
WEBSITE_SVG_HEIGHT = 1300.0
WEBSITE_EPSILON = 0.1
WEBSITE_NUMBER_OF_ADJUSTMENT_ITERATIONS = 80
WEBSITE_DELTA = 0.03
WEBSITE_MAX_STEPS = 200
WEBSITE_SEED = 1


@dataclass(frozen=True)
class TorusLayoutResult:
    nodes: list[Any]
    positions: np.ndarray
    raw_positions: np.ndarray
    iterations: int
    seed: int
    config: dict[str, Any]


@dataclass(frozen=True)
class TorusLayoutComparison:
    wrapper: TorusLayoutResult
    direct_positions: np.ndarray
    direct_raw_positions: np.ndarray
    direct_iterations: int
    max_abs_position_diff: float
    max_abs_raw_position_diff: float


def _wrap01(values: np.ndarray) -> np.ndarray:
    wrapped = np.mod(values, 1.0)
    wrapped[wrapped < 0.0] += 1.0
    return wrapped


def _windows_to_wsl_path(path: Path) -> str:
    resolved = path.resolve()
    path_str = resolved.as_posix()
    drive, rest = os.path.splitdrive(str(resolved))
    if drive:
        return f"/mnt/{drive[0].lower()}{rest.replace('\\', '/')}"
    return path_str


def _escape_wsl_shell_arg(value: str) -> str:
    escaped = value.replace("\\", "\\\\")
    escaped = escaped.replace(" ", "\\ ")
    escaped = escaped.replace("(", "\\(")
    escaped = escaped.replace(")", "\\)")
    escaped = escaped.replace("&", "\\&")
    return escaped


def _all_edge_weights_are_one(G: nx.Graph, weight: str) -> bool:
    for _, _, data in G.edges(data=True):
        edge_weight = data.get(weight, 1.0)
        if not math.isclose(float(edge_weight), 1.0):
            return False
    return True


def _graph_diameter(
    G: nx.Graph,
    weight: str,
    shortest_path_lengths: np.ndarray | None,
) -> float:
    if shortest_path_lengths is not None:
        matrix = np.asarray(shortest_path_lengths, dtype=np.float64)
        finite = matrix[np.isfinite(matrix)]
        if finite.size == 0:
            return 0.0
        return float(np.max(finite))

    if _all_edge_weights_are_one(G, weight):
        diameter = 0
        for _, distances in nx.all_pairs_shortest_path_length(G):
            if distances:
                diameter = max(diameter, max(distances.values()))
        return float(diameter)

    diameter = 0.0
    for _, distances in nx.all_pairs_dijkstra_path_length(G, weight=weight):
        if distances:
            diameter = max(diameter, max(float(distance) for distance in distances.values()))
    return float(diameter)


def _website_link_length(svg_width: float, graph_diameter: float) -> float:
    divisor = graph_diameter + 1.0
    link_length = svg_width / 3.0 / divisor
    target = svg_width / 9.0
    if target - 5.0 < link_length < target + 5.0:
        divisor += 1.0
        link_length = svg_width / 3.0 / divisor
    return float(link_length)


def _distance_matrix(G: nx.Graph, weight: str) -> np.ndarray:
    nodes = list(G.nodes())
    index_by_node = {node: index for index, node in enumerate(nodes)}
    matrix = np.full((len(nodes), len(nodes)), np.inf, dtype=np.float64)
    np.fill_diagonal(matrix, 0.0)
    for source, distances in nx.all_pairs_dijkstra_path_length(G, weight=weight):
        source_index = index_by_node[source]
        for target, distance in distances.items():
            matrix[source_index, index_by_node[target]] = float(distance)
    return matrix


def _coerce_positions(initial_positions: Any, node_count: int) -> list[list[float]] | None:
    if initial_positions is None:
        return None
    positions = np.asarray(initial_positions, dtype=float)
    if positions.shape != (node_count, 2):
        raise ValueError(
            f"initial_positions must have shape {(node_count, 2)}, got {positions.shape}"
        )
    return _wrap01(positions).tolist()


def _build_payload(
    G: nx.Graph,
    *,
    weight: str,
    seed: int,
    max_iters: int,
    epsilon: float,
    svg_width: float,
    svg_height: float,
    link_length: float | None,
    number_of_adjustment_iterations: int,
    delta: float,
    initial_positions: Any,
    shortest_path_lengths: np.ndarray | None,
) -> tuple[dict[str, Any], list[Any]]:
    node_labels = list(G.nodes())
    node_index = {node: index for index, node in enumerate(node_labels)}
    links = []
    for source, target, data in G.edges(data=True):
        link: dict[str, Any] = {
            "source": node_index[source],
            "target": node_index[target],
        }
        if weight in data:
            link["weight"] = float(data[weight])
        links.append(link)

    if shortest_path_lengths is not None:
        distance_matrix = np.asarray(shortest_path_lengths, dtype=np.float64)
    elif _all_edge_weights_are_one(G, weight):
        distance_matrix = None
    else:
        distance_matrix = _distance_matrix(G, weight)

    if distance_matrix is not None and distance_matrix.shape != (len(node_labels), len(node_labels)):
        raise ValueError(
            "shortest_path_lengths must have shape "
            f"{(len(node_labels), len(node_labels))}, got {distance_matrix.shape}"
        )

    if link_length is None:
        graph_diameter = _graph_diameter(G, weight, distance_matrix)
        link_length = _website_link_length(svg_width, graph_diameter)

    payload: dict[str, Any] = {
        "graph": {
            "nodes": [{"id": index} for index in range(len(node_labels))],
            "links": links,
        },
        "config": {
            "svgWidth": float(svg_width),
            "svgHeight": float(svg_height),
            "epsilon": float(epsilon),
            "linkLength": float(link_length),
            "numberOfAdjustmentIterations": int(number_of_adjustment_iterations),
            "maxSteps": int(max_iters),
            "delta": float(delta),
            "bEnableAnimation": False,
        },
        "seed": int(seed),
        "initialPositions": _coerce_positions(initial_positions, len(node_labels)),
    }
    if distance_matrix is not None:
        payload["graph"]["shortestPathLengths"] = distance_matrix.tolist()

    return payload, node_labels


def build_layout_payload(
    G: nx.Graph,
    weight: str = "weight",
    seed: int = WEBSITE_SEED,
    max_iters: int = WEBSITE_MAX_STEPS,
    epsilon: float = WEBSITE_EPSILON,
    svg_width: float = WEBSITE_SVG_WIDTH,
    svg_height: float = WEBSITE_SVG_HEIGHT,
    link_length: float | None = None,
    number_of_adjustment_iterations: int | None = WEBSITE_NUMBER_OF_ADJUSTMENT_ITERATIONS,
    delta: float = WEBSITE_DELTA,
    initial_positions: Any = None,
    shortest_path_lengths: np.ndarray | None = None,
) -> tuple[dict[str, Any], list[Any]]:
    if number_of_adjustment_iterations is None:
        number_of_adjustment_iterations = WEBSITE_NUMBER_OF_ADJUSTMENT_ITERATIONS
    return _build_payload(
        G,
        weight=weight,
        seed=seed,
        max_iters=max_iters,
        epsilon=epsilon,
        svg_width=svg_width,
        svg_height=svg_height,
        link_length=link_length,
        number_of_adjustment_iterations=number_of_adjustment_iterations,
        delta=delta,
        initial_positions=initial_positions,
        shortest_path_lengths=shortest_path_lengths,
    )


def _run_node(payload: dict[str, Any], *, node_memory_mb: int | None) -> dict[str, Any]:
    package_dir = Path(__file__).resolve().parent
    runner_path = package_dir / "toruslayout_runner.mjs"
    input_text = json.dumps(payload)

    def run_host() -> subprocess.CompletedProcess[str]:
        host_command = ["node"]
        if node_memory_mb is not None:
            host_command.append(f"--max-old-space-size={int(node_memory_mb)}")
        host_command.append(str(runner_path))
        return subprocess.run(
            host_command,
            input=input_text,
            text=True,
            capture_output=True,
            check=True,
            cwd=package_dir,
        )

    def run_wsl() -> subprocess.CompletedProcess[str]:
        wsl_runner = _escape_wsl_shell_arg(_windows_to_wsl_path(runner_path))
        node_prefix = "node"
        if node_memory_mb is not None:
            node_prefix = f"node --max-old-space-size={int(node_memory_mb)}"
        command = (
            'export NVM_DIR="$HOME/.nvm"; '
            '[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"; '
            f"{node_prefix} {wsl_runner}"
        )
        return subprocess.run(
            ["wsl.exe", "-e", "bash", "-lc", command],
            input=input_text,
            text=True,
            capture_output=True,
            check=True,
            cwd=package_dir,
        )

    def run_bash_with_nvm() -> subprocess.CompletedProcess[str]:
        runner_arg = shlex.quote(str(runner_path))
        node_prefix = "node"
        if node_memory_mb is not None:
            node_prefix = f"node --max-old-space-size={int(node_memory_mb)}"
        command = (
            'export NVM_DIR="$HOME/.nvm"; '
            '[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"; '
            f"{node_prefix} {runner_arg}"
        )
        return subprocess.run(
            ["bash", "-lc", command],
            input=input_text,
            text=True,
            capture_output=True,
            check=True,
            cwd=package_dir,
        )

    completed: subprocess.CompletedProcess[str]
    try:
        if os.name == "nt":
            completed = run_wsl()
        else:
            try:
                completed = run_host()
            except FileNotFoundError:
                completed = run_bash_with_nvm()
    except FileNotFoundError as exc:
        if os.name == "nt":
            raise RuntimeError("Could not find wsl.exe to run the TypeScript torus layout") from exc
        raise RuntimeError("Could not find node to run the TypeScript torus layout") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(exc.stderr.strip() or exc.stdout.strip() or "Node runner failed") from exc

    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Failed to parse torus layout runner output: {completed.stdout[:500]!r}"
        ) from exc


def layout_graph(
    G: nx.Graph,
    weight: str = "weight",
    seed: int = WEBSITE_SEED,
    max_iters: int = WEBSITE_MAX_STEPS,
    epsilon: float = WEBSITE_EPSILON,
    svg_width: float = WEBSITE_SVG_WIDTH,
    svg_height: float = WEBSITE_SVG_HEIGHT,
    link_length: float | None = None,
    number_of_adjustment_iterations: int | None = WEBSITE_NUMBER_OF_ADJUSTMENT_ITERATIONS,
    delta: float = WEBSITE_DELTA,
    initial_positions: Any = None,
    shortest_path_lengths: np.ndarray | None = None,
    node_memory_mb: int | None = 4096,
) -> TorusLayoutResult:
    payload, node_labels = build_layout_payload(
        G,
        weight=weight,
        seed=seed,
        max_iters=max_iters,
        epsilon=epsilon,
        svg_width=svg_width,
        svg_height=svg_height,
        link_length=link_length,
        number_of_adjustment_iterations=number_of_adjustment_iterations,
        delta=delta,
        initial_positions=initial_positions,
        shortest_path_lengths=shortest_path_lengths,
    )
    try:
        result = _run_node(payload, node_memory_mb=node_memory_mb)
    except RuntimeError as e:
        import sys
        print(f"ERROR: {e}", file=sys.stderr)
        raise
    positions = np.asarray(result["positions"], dtype=float)
    raw_positions = np.asarray(result["rawPositions"], dtype=float)

    return TorusLayoutResult(
        nodes=node_labels,
        positions=_wrap01(positions),
        raw_positions=raw_positions,
        iterations=int(result["iterations"]),
        seed=int(seed),
        config=dict(result["config"]),
    )


def compare_with_direct_ts(
    G: nx.Graph,
    weight: str = "weight",
    seed: int = WEBSITE_SEED,
    max_iters: int = WEBSITE_MAX_STEPS,
    epsilon: float = WEBSITE_EPSILON,
    svg_width: float = WEBSITE_SVG_WIDTH,
    svg_height: float = WEBSITE_SVG_HEIGHT,
    link_length: float | None = None,
    number_of_adjustment_iterations: int | None = WEBSITE_NUMBER_OF_ADJUSTMENT_ITERATIONS,
    delta: float = WEBSITE_DELTA,
    initial_positions: Any = None,
    shortest_path_lengths: np.ndarray | None = None,
    node_memory_mb: int | None = 4096,
) -> TorusLayoutComparison:
    payload, _ = build_layout_payload(
        G,
        weight=weight,
        seed=seed,
        max_iters=max_iters,
        epsilon=epsilon,
        svg_width=svg_width,
        svg_height=svg_height,
        link_length=link_length,
        number_of_adjustment_iterations=number_of_adjustment_iterations,
        delta=delta,
        initial_positions=initial_positions,
        shortest_path_lengths=shortest_path_lengths,
    )
    wrapper = layout_graph(
        G,
        weight=weight,
        seed=seed,
        max_iters=max_iters,
        epsilon=epsilon,
        svg_width=svg_width,
        svg_height=svg_height,
        link_length=link_length,
        number_of_adjustment_iterations=number_of_adjustment_iterations,
        delta=delta,
        initial_positions=initial_positions,
        shortest_path_lengths=shortest_path_lengths,
        node_memory_mb=node_memory_mb,
    )
    direct = _run_node(payload, node_memory_mb=node_memory_mb)
    direct_positions = _wrap01(np.asarray(direct["positions"], dtype=float))
    direct_raw_positions = np.asarray(direct["rawPositions"], dtype=float)
    return TorusLayoutComparison(
        wrapper=wrapper,
        direct_positions=direct_positions,
        direct_raw_positions=direct_raw_positions,
        direct_iterations=int(direct["iterations"]),
        max_abs_position_diff=float(np.max(np.abs(wrapper.positions - direct_positions))),
        max_abs_raw_position_diff=float(np.max(np.abs(wrapper.raw_positions - direct_raw_positions))),
    )


__all__ = [
    "TorusLayoutComparison",
    "TorusLayoutResult",
    "build_layout_payload",
    "compare_with_direct_ts",
    "layout_graph",
]
