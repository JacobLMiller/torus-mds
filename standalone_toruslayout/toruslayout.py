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

from numba import njit

#Harc-coded constants from the "its-a-wrap" website
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
    wrap_python: TorusLayoutResult
    wrap_ts: TorusLayoutResult
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


def wrap_ts(
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
def _build_adjacency(node_count: int, links: list[dict[str, Any]]) -> list[list[dict[str, float]]]:
    adjacency: list[list[dict[str, float]]] = [[] for _ in range(node_count)]
    for link in links:
        source = int(link["source"]) if isinstance(link["source"], (int, np.integer)) else int(link["source"]["index"])
        target = int(link["target"]) if isinstance(link["target"], (int, np.integer)) else int(link["target"]["index"])
        weight_value = float(link.get("weight", 1.0))
        weight = weight_value if math.isfinite(weight_value) else 1.0
        adjacency[source].append({"target": target, "weight": weight})
        adjacency[target].append({"target": source, "weight": weight})
    return adjacency


def _all_adjacency_weights_are_one(adjacency: list[list[dict[str, float]]]) -> bool:
    return all(edge["weight"] == 1 for neighbors in adjacency for edge in neighbors)


def _bfs_distances(adjacency: list[list[dict[str, float]]], source: int) -> list[float]:
    distances = [math.inf] * len(adjacency)
    queue = [source]
    queue_index = 0
    distances[source] = 0.0
    while queue_index < len(queue):
        node = queue[queue_index]
        queue_index += 1
        base_distance = distances[node]
        for edge in adjacency[node]:
            target = int(edge["target"])
            if distances[target] != math.inf:
                continue
            distances[target] = base_distance + 1.0
            queue.append(target)
    return distances


def _dijkstra_distances(adjacency: list[list[dict[str, float]]], source: int) -> list[float]:
    visited = [False] * len(adjacency)
    distances = [math.inf] * len(adjacency)
    distances[source] = 0.0

    for _ in range(len(adjacency)):
        best_node = -1
        best_distance = math.inf
        for node, visited_node in enumerate(visited):
            if not visited_node and distances[node] < best_distance:
                best_distance = distances[node]
                best_node = node
        if best_node < 0:
            break
        visited[best_node] = True
        for edge in adjacency[best_node]:
            candidate = best_distance + float(edge["weight"])
            target = int(edge["target"])
            if candidate < distances[target]:
                distances[target] = candidate
    return distances


def _compute_shortest_path_lengths_for_payload(graph: dict[str, Any]) -> list[list[float]]:
    if "shortestPathLengths" in graph and graph["shortestPathLengths"] is not None:
        return [
            [float(distance) for distance in row]
            for row in graph["shortestPathLengths"]
        ]

    adjacency = _build_adjacency(len(graph["nodes"]), graph["links"])
    algorithm = _bfs_distances if _all_adjacency_weights_are_one(adjacency) else _dijkstra_distances
    return [algorithm(adjacency, source) for source in range(len(adjacency))]


def _compute_shortest_path_matrix(graph: dict[str, Any]) -> np.ndarray:
    shortest_path_lengths = graph.get("shortestPathLengths")
    if shortest_path_lengths is None:
        shortest_path_lengths = _compute_shortest_path_lengths_for_payload(graph)
    matrix = np.asarray(shortest_path_lengths, dtype=np.float64)
    node_count = len(graph["nodes"])
    if matrix.shape != (node_count, node_count):
        raise ValueError(
            "shortestPathLengths must have shape "
            f"{(node_count, node_count)}, got {matrix.shape}"
        )
    return np.ascontiguousarray(matrix)


def _build_param_w_matrix(
    shortest_path_matrix: np.ndarray,
    link_length: float,
) -> tuple[np.ndarray, float, float]:
    distances = shortest_path_matrix * float(link_length)
    weights = np.zeros_like(distances)
    valid = np.isfinite(distances) & (distances > 0.0)
    weights[valid] = 1.0 / np.square(distances[valid])
    np.fill_diagonal(weights, 0.0)

    positive = weights[weights > 0.0]
    minimum = float(np.min(positive)) if positive.size else math.inf
    maximum = float(np.max(positive)) if positive.size else 0.0
    return np.ascontiguousarray(weights.reshape(-1), dtype=np.float64), minimum, maximum



@njit(cache=True)
def _numba_get_next(seed: int) -> tuple[int, float]:
    seed = (seed * 214013 + 2531011) % 2147483648
    return seed, (seed >> 16) / 32767.0


@njit(cache=True)
def _numba_fill_random_order(seed: int, temp_array: np.ndarray) -> int:
    node_count = temp_array.shape[0]
    for index in range(node_count):
        temp_array[index] = -1
    value = 0
    while value < node_count:
        seed, random_value = _numba_get_next(seed)
        slot = int(math.floor(random_value * node_count))
        if slot < 0 or slot >= node_count:
            continue
        if temp_array[slot] < 0:
            temp_array[slot] = value
            value += 1
    return seed


def _advance_seed_for_ts_run_setup(seed: int, node_count: int) -> int:
    scratch_order = np.empty(node_count, dtype=np.int64)
    for _ in range(node_count + 1):
        seed = _numba_fill_random_order(seed, scratch_order)
    return seed


@njit(cache=True)
def _numba_wrap_coordinate(value: float, lower: float, upper: float, period: float) -> float:
    if value > upper:
        value -= period
        while value > upper:
            value -= period
    elif value < lower:
        value += period
        while value < lower:
            value += period
    return value


@njit(cache=True)
def _numba_wrap_dirty_coordinates(
    x: np.ndarray,
    y: np.ndarray,
    dirty_count: int,
    dirty_a: int,
    dirty_b: int,
    lower_x: float,
    upper_x: float,
    period_x: float,
    lower_y: float,
    upper_y: float,
    period_y: float,
) -> None:
    if dirty_count > 0:
        x[dirty_a] = _numba_wrap_coordinate(x[dirty_a], lower_x, upper_x, period_x)
        y[dirty_a] = _numba_wrap_coordinate(y[dirty_a], lower_y, upper_y, period_y)
    if dirty_count > 1:
        x[dirty_b] = _numba_wrap_coordinate(x[dirty_b], lower_x, upper_x, period_x)
        y[dirty_b] = _numba_wrap_coordinate(y[dirty_b], lower_y, upper_y, period_y)


@njit(cache=True)
def _numba_wrap_all_coordinates(
    x: np.ndarray,
    y: np.ndarray,
    lower_x: float,
    upper_x: float,
    period_x: float,
    lower_y: float,
    upper_y: float,
    period_y: float,
) -> None:
    for index in range(x.shape[0]):
        x[index] = _numba_wrap_coordinate(x[index], lower_x, upper_x, period_x)
        y[index] = _numba_wrap_coordinate(y[index], lower_y, upper_y, period_y)


@njit(cache=True)
def _numba_compute_shortest_distance_over_context_fast(
    source_x: float,
    source_y: float,
    target_x: float,
    target_y: float,
    ideal_distance: float,
    mapping_period: float,
) -> tuple[float, float, float]:
    best_edge = math.inf
    best_diff = math.inf
    best_dx = 0.0
    best_dy = 0.0

    for mapped_x, mapped_y in (
        (target_x - mapping_period, target_y),
        (target_x, target_y - mapping_period),
        (target_x + mapping_period, target_y),
        (target_x, target_y + mapping_period),
        (target_x - mapping_period, target_y - mapping_period),
        (target_x + mapping_period, target_y - mapping_period),
        (target_x + mapping_period, target_y + mapping_period),
        (target_x - mapping_period, target_y + mapping_period),
    ):
        dx = source_x - mapped_x
        dy = source_y - mapped_y
        distance = math.sqrt(dx * dx + dy * dy)
        diff = abs(distance - ideal_distance)
        if best_diff > diff:
            best_diff = diff
            best_edge = distance
            best_dx = dx
            best_dy = dy

    direct_dx = source_x - target_x
    direct_dy = source_y - target_y
    direct_distance = math.sqrt(direct_dx * direct_dx + direct_dy * direct_dy)
    direct_diff = abs(direct_distance - ideal_distance)
    if best_diff > direct_diff:
        return direct_distance, direct_dx, direct_dy
    return best_edge, best_dx, best_dy


@njit(cache=True)
def _numba_run_core(
    seed: int,
    x: np.ndarray,
    y: np.ndarray,
    shortest_path_flat: np.ndarray,
    param_w_matrix: np.ndarray,
    fixed_iterations: np.ndarray,
    max_steps: int,
    number_of_adjustment_iterations: int,
    delta: float,
    link_length: float,
    min_x: float,
    max_x: float,
    period_x: float,
    min_y: float,
    max_y: float,
    period_y: float,
    unlimited_base: float,
    unlimited_lambda: float,
) -> int:
    node_count = x.shape[0]
    visited_stamp = np.zeros(node_count * node_count, dtype=np.int64)
    random_order_node_index = np.empty(node_count, dtype=np.int64)
    random_order_array = np.empty(node_count, dtype=np.int64)
    step = 0
    max_displacement = math.inf
    dirty_count = 0
    dirty_a = 0
    dirty_b = 0

    while max_displacement >= delta:
        if step > max_steps:
            break

        max_displacement = -1.0
        seed = _numba_fill_random_order(seed, random_order_array)
        for index in range(node_count):
            random_order_node_index[index] = random_order_array[index]

        if step < number_of_adjustment_iterations:
            fixed_iteration = fixed_iterations[step]
            unlimited_scale = 0.0
        else:
            fixed_iteration = 0.0
            unlimited_scale = unlimited_base / (1.0 + unlimited_lambda * step)

        stamp = step + 1
        for order_index in range(node_count):
            a = random_order_node_index[order_index]
            seed = _numba_fill_random_order(seed, random_order_array)
            a_row = a * node_count
            for inner_index in range(node_count):
                b = random_order_array[inner_index]
                visited_index = a_row + b
                if visited_stamp[visited_index] == stamp:
                    continue
                if a == b:
                    visited_stamp[visited_index] = stamp
                    continue

                distance_factor = shortest_path_flat[visited_index]
                if not np.isfinite(distance_factor):
                    visited_stamp[visited_index] = stamp
                    continue
                ideal_distance = link_length * distance_factor
                if ideal_distance <= 0.0:
                    visited_stamp[visited_index] = stamp
                    continue

                if dirty_count > 0:
                    _numba_wrap_dirty_coordinates(
                        x,
                        y,
                        dirty_count,
                        dirty_a,
                        dirty_b,
                        min_x,
                        max_x,
                        period_x,
                        min_y,
                        max_y,
                        period_y,
                    )
                    dirty_count = 0

                xa = x[a]
                ya = y[a]
                xb = x[b]
                yb = y[b]
                distance, dx, dy = _numba_compute_shortest_distance_over_context_fast(
                    xa,
                    ya,
                    xb,
                    yb,
                    ideal_distance,
                    period_y,
                )
                unit_dx = dx / distance
                unit_dy = dy / distance
                temp = (distance - ideal_distance) / 2.0
                vector_rx = unit_dx * temp
                vector_ry = unit_dy * temp

                weight = param_w_matrix[visited_index]
                if step < number_of_adjustment_iterations:
                    temp_meu = weight * fixed_iteration
                    if temp_meu > 1.0:
                        temp_meu = 1.0
                else:
                    temp_meu = unlimited_scale * weight

                new_x = xa - temp_meu * vector_rx
                new_y = ya - temp_meu * vector_ry
                force_to_converge = False
                if step >= number_of_adjustment_iterations:
                    if new_x > max_x or new_x < min_x or new_y > max_y or new_y < min_y:
                        new_x = xa
                        new_y = ya
                        force_to_converge = True
                tmp_movement = math.sqrt((xa - new_x) * (xa - new_x) + (ya - new_y) * (ya - new_y))
                if not force_to_converge:
                    x[a] = new_x
                    y[a] = new_y
                    dirty_a = a
                    dirty_count = 1
                if max_displacement < tmp_movement:
                    max_displacement = tmp_movement

                new_x = xb + temp_meu * vector_rx
                new_y = yb + temp_meu * vector_ry
                force_to_converge = False
                if step >= number_of_adjustment_iterations:
                    if new_x > max_x or new_x < min_x or new_y > max_y or new_y < min_y:
                        new_x = xb
                        new_y = yb
                        force_to_converge = True
                tmp_movement = math.sqrt((xb - new_x) * (xb - new_x) + (yb - new_y) * (yb - new_y))
                if not force_to_converge:
                    x[b] = new_x
                    y[b] = new_y
                    if dirty_count == 0:
                        dirty_a = b
                        dirty_count = 1
                    else:
                        dirty_b = b
                        dirty_count = 2
                if max_displacement < tmp_movement:
                    max_displacement = tmp_movement

                visited_stamp[visited_index] = stamp
                visited_stamp[b * node_count + a] = stamp

        step += 1

    if dirty_count > 0:
        _numba_wrap_dirty_coordinates(
            x,
            y,
            dirty_count,
            dirty_a,
            dirty_b,
            min_x,
            max_x,
            period_x,
            min_y,
            max_y,
            period_y,
        )
    _numba_wrap_all_coordinates(x, y, min_x, max_x, period_x, min_y, max_y, period_y)
    return step


def _run_numba(payload: dict[str, Any]) -> dict[str, Any]:
    graph = payload["graph"]
    configuration = payload["config"]
    node_count = len(graph["nodes"])
    period_x = float(configuration["svgWidth"]) / 3.0
    period_y = float(configuration["svgHeight"]) / 3.0
    min_x = period_x
    min_y = period_y
    max_x = float(configuration["svgWidth"]) * 2.0 / 3.0
    max_y = float(configuration["svgHeight"]) * 2.0 / 3.0
    link_length = float(configuration["linkLength"])
    max_steps = int(configuration["maxSteps"])
    number_of_adjustment_iterations = int(configuration["numberOfAdjustmentIterations"])
    delta = float(configuration["delta"])

    if node_count == 0:
        return {
            "positions": np.empty((0, 2), dtype=float),
            "rawPositions": np.empty((0, 2), dtype=float),
            "iterations": 0,
            "config": dict(configuration),
        }

    shortest_path_matrix = _compute_shortest_path_matrix(graph)
    shortest_path_flat = np.ascontiguousarray(shortest_path_matrix.reshape(-1), dtype=np.float64)
    param_w_matrix, param_w_min, param_w_max = _build_param_w_matrix(shortest_path_matrix, link_length)

    seed = int(payload.get("seed", WEBSITE_SEED))
    x = np.empty(node_count, dtype=np.float64)
    y = np.empty(node_count, dtype=np.float64)
    initial_positions = payload.get("initialPositions")
    if initial_positions is not None:
        initial_positions_array = np.asarray(initial_positions, dtype=np.float64)
        if initial_positions_array.shape != (node_count, 2):
            raise ValueError(
                f"Expected initial positions with shape {(node_count, 2)}, got "
                f"{initial_positions_array.shape}"
            )
        wrapped_initial_positions = _wrap01(initial_positions_array)
        x[:] = min_x + wrapped_initial_positions[:, 0] * period_x
        y[:] = min_y + wrapped_initial_positions[:, 1] * period_y
    else:
        center_x = float(configuration["svgWidth"]) / 2.0 - 0.5
        center_y = float(configuration["svgHeight"]) / 2.0 - 0.5
        for index in range(node_count):
            seed, random_value = _numba_get_next(seed)
            x[index] = center_x + random_value
            seed, random_value = _numba_get_next(seed)
            y[index] = center_y + random_value

    seed = _advance_seed_for_ts_run_setup(seed, node_count)

    param_l_max = 1.0 / param_w_min
    param_l_min = float(configuration["epsilon"]) / param_w_max
    param_l_lambda = -math.log(param_l_min / param_l_max) / (number_of_adjustment_iterations - 1)
    fixed_iterations = np.fromiter(
        (
            (1.0 / param_w_min) * math.exp(-param_l_lambda * iteration)
            for iteration in range(max_steps)
        ),
        dtype=np.float64,
        count=max_steps,
    )

    small_epsilon = 0.001
    unlimited_min = small_epsilon / param_w_max
    unlimited_lambda = (param_l_max / unlimited_min - 1.0) / (
        number_of_adjustment_iterations - 1
    )
    unlimited_base = 1.0 / param_w_max

    step = _numba_run_core(
        seed,
        x,
        y,
        shortest_path_flat,
        param_w_matrix,
        fixed_iterations,
        max_steps,
        number_of_adjustment_iterations,
        delta,
        link_length,
        min_x,
        max_x,
        period_x,
        min_y,
        max_y,
        period_y,
        unlimited_base,
        unlimited_lambda,
    )

    raw_positions = np.empty((node_count, 2), dtype=np.float64)
    raw_positions[:, 0] = x
    raw_positions[:, 1] = y
    positions = np.empty((node_count, 2), dtype=np.float64)
    positions[:, 0] = np.mod((x - min_x) / period_x, 1.0)
    positions[:, 1] = np.mod((y - min_y) / period_y, 1.0)

    return {
        "positions": positions,
        "rawPositions": raw_positions,
        "iterations": int(step),
        "config": dict(configuration),
    }



def wrap_python(
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
    result = _run_numba(payload)
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


def compare_python_with_direct_ts(
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

    ts_result = wrap_ts(
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
    python_result = wrap_python(
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

    return TorusLayoutComparison(
        wrap_python=python_result,
        wrap_ts=ts_result,
        max_abs_position_diff=float(np.max(np.abs(python_result.positions - ts_result.positions))),
        max_abs_raw_position_diff=float(np.max(np.abs(python_result.raw_positions - ts_result.raw_positions))),
    )


__all__ = [
    "TorusLayoutComparison",
    "TorusLayoutResult",
    "build_layout_payload",
    "compare_python_with_direct_ts",
    "wrap_python",
    "wrap_ts",
]
