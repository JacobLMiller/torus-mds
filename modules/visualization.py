from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon as MplPolygon


from .geometry import torus_edge_segments
from .metrics import geodesic_matrix, subsample


def plot_embedding_with_torus_edges(X=None, G=None, outpath="output.png",
                                   s=10, node_alpha=0.9,
                                   edge_alpha=0.10, edge_lw=0.4,
                                   colors=None,
                                   order=None,
                                   torus=None,
                                   ax=None):
    """
    Scatter + edges drawn along shortest torus geodesics, displayed in physical space.

    X:     (N,2) embedding in [0,1)^2 parameter space (optional if torus is provided).
    G:     networkx graph
    torus: optional projector object. Reads torus_embedding_, alpha_, r0_, r1_, theta_
           to recover the physical geometry. theta_ defaults to pi/2 (rectangular).

    Physical lattice vectors (columns of M):
      e1 = (alpha*r0, 0)
      e2 = (alpha*r1*cos(theta), alpha*r1*sin(theta))
    All coordinates are mapped through M before plotting. Normal axes are hidden;
    tick marks with physical-unit labels are drawn directly on the parallelogram edges.
    """
    # Resolve embedding
    if X is None:
        if torus is None or torus.torus_embedding_ is None:
            raise ValueError("Provide either X or a fitted torus object with torus_embedding_.")
        X = torus.torus_embedding_

    if order is not None:
        X = X[order]
        if colors is not None:
            colors = colors[order]

    X = np.asarray(X, dtype=np.float64) % 1.0
    idx = {n: i for i, n in enumerate(G.nodes())} if G is not None else {}

    # Physical side lengths: alpha * r; fall back to 1 if not available
    alpha = torus.alpha_ if (torus is not None and torus.alpha_ is not None) else 1.0
    r0 = alpha * (torus.r0_ if (torus is not None and torus.r0_ is not None) else 1.0)
    r1 = alpha * (torus.r1_ if (torus is not None and torus.r1_ is not None) else 1.0)
    theta = torus.theta_ if (torus is not None and hasattr(torus, 'theta_') and torus.theta_ is not None) else np.pi / 2

    # Lattice matrix: maps parameter coords (u,v) -> physical coords x = M @ [u,v]
    # e1 = (r0, 0),  e2 = (r1*cos(theta), r1*sin(theta))
    M = np.array([[r0, r1 * np.cos(theta)],
                  [0,  r1 * np.sin(theta)]])

    if ax is None:
        fig, ax = plt.subplots()
    # edges — segments computed in [0,1)^2, then mapped to physical space
    for u, v in (G.edges() if G is not None else []):
        i, j = idx[u], idx[v]
        p, q = X[i], X[j]
        for a, b in torus_edge_segments(p, q):
            a_phys = M @ a
            b_phys = M @ b
            ax.plot([a_phys[0], b_phys[0]], [a_phys[1], b_phys[1]],
                     color="k", alpha=edge_alpha, lw=edge_lw, zorder=1)
    # points
    X_phys = X @ M.T
    ax.scatter(X_phys[:, 0], X_phys[:, 1], s=s, alpha=node_alpha, zorder=2,
               c=colors if colors is not None else "blue")

    # parallelogram boundary of the fundamental domain
    corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]) @ M.T
    ax.add_patch(MplPolygon(corners, closed=True, fill=False,
                            edgecolor='gray', lw=1, zorder=3))

    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    # --- Tick marks with physical-unit labels on parallelogram edges ---
    n_ticks   = 5
    tick_ts   = np.linspace(0, 1, n_ticks)
    tick_size = 0.025 * max(r0, r1)   # tick length in physical units
    pad       = 0.03  * max(r0, r1)   # gap between tick tip and label

    e1_perp = np.array([0.0, -1.0])                      # outward normal to bottom side
    e2_end  = M @ np.array([0.0, 1.0])                   # physical tip of e2
    e2_perp = np.array([-np.sin(theta), np.cos(theta)])  # outward normal to left side

    # Bottom side (e1): 5 ticks with labels showing physical distance from 0 to r0
    for t in tick_ts:
        pt = np.array([t * r0, 0.0])
        tip = pt + tick_size * e1_perp
        ax.plot([pt[0], tip[0]], [pt[1], tip[1]], color='dimgray', lw=0.8, zorder=4)
        lbl_pos = tip + pad * e1_perp
        ax.text(lbl_pos[0], lbl_pos[1], f"{t * r0:.3g}",
                ha='center', va='top', fontsize=8, color='dimgray')

    # Left side (e2): 5 ticks with labels showing physical distance from 0 to r1.
    # Skip the t=0 label — already covered by the "0" on the bottom side.
    for t in tick_ts:
        pt = t * e2_end
        tip = pt + tick_size * e2_perp
        ax.plot([pt[0], tip[0]], [pt[1], tip[1]], color='dimgray', lw=0.8, zorder=4)
        if t > 0:
            lbl_pos = tip + pad * e2_perp
            ax.text(lbl_pos[0], lbl_pos[1], f"{t * r1:.3g}",
                    ha='center', va='center', rotation=np.degrees(theta),
                    fontsize=8, color='dimgray')
    return ax


LIFT_CMAP = "tab10"


def lift_styles(p, q, offsets, shortest_offset=None, cmap=LIFT_CMAP):
    """
    Line kwargs for every lift of a point pair, keyed by offset.

    Colour separates the lifts from each other, which is what makes the folded view readable:
    a geodesic that crosses a boundary is drawn as several pieces, and sharing a colour is the
    only cue that those pieces are one line. Colours come from cmap in the order the offsets are
    given, so an offset keeps its colour as long as the offset list does not change. A listed
    (qualitative) cmap contributes its colours directly, a continuous one is sampled evenly with
    the washed-out ends left off.

    Line style says how far the lift wraps: solid inside the fundamental domain, dotted for up
    to one wrap per axis, dashed beyond that. The lift at shortest_offset is thickened and drawn
    on top of the others.

    Returns a dict from offset to a record with keys offset, length and kwargs, the last ready
    to splat into ax.plot or Line2D.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    offsets = [(int(m), int(n)) for m, n in offsets]
    short = None if shortest_offset is None else (int(shortest_offset[0]), int(shortest_offset[1]))

    cm = plt.get_cmap(cmap)
    if isinstance(cm, mcolors.ListedColormap) and cm.N <= 20:
        colors = [cm(i % cm.N) for i in range(len(offsets))]
    else:
        colors = [cm(v) for v in np.linspace(0.1, 0.9, len(offsets))]

    styles = {}
    for offset, color in zip(offsets, colors):
        reach = max(abs(offset[0]), abs(offset[1]))
        if reach == 0:
            linestyle, lw = "-", 1.6
        elif reach == 1:
            linestyle, lw = ":", 1.1
        else:
            linestyle, lw = "--", 1.3
        zorder = 2
        if offset == short:
            lw *= 2.0
            zorder = 3
        length = float(np.linalg.norm(q + np.asarray(offset, dtype=np.float64) - p))
        styles[offset] = dict(offset=offset, length=length,
                              kwargs=dict(color=color, linestyle=linestyle, lw=lw, zorder=zorder))
    return styles


def _annotate_pair(ax, p, q, names=("p", "q"), pad=6.0, fontsize=8):
    """
    Label two points, each label pushed away from the centre of the tile its point sits in.

    pad is in points rather than data units, so the label clears the marker by the same amount
    whether the axes span one tile or a whole patch of the cover.
    """
    for point, name in zip((p, q), names):
        away = point - np.floor(point) - 0.5
        norm = np.linalg.norm(away)
        direction = away / norm if norm > 1e-9 else np.array([0.0, -1.0])
        # White backing keeps the label readable where lifts bunch up around the point.
        ax.annotate(name, xy=point, xytext=pad * direction, textcoords="offset points",
                    fontsize=fontsize, ha="center", va="center", zorder=6)


def plot_torus_lifts(p, q, offsets, ax=None, shortest_offset=None, cmap=LIFT_CMAP,
                     label_points=True, s=45, pad=0.0):
    """
    Draw many straight torus geodesics between two points inside the fundamental domain.

    p, q:    points in [0,1)^2, in parameter space (unit square, square torus).
    offsets: iterable of integer pairs (m, n). Each one lifts q into the tile (m, n) of the
             universal cover, so each offset contributes one geodesic, drawn broken at every
             boundary it crosses. On a torus these are all the geodesics joining p and q.
    shortest_offset: offset whose lift is highlighted, typically the shortest one.
    cmap:    colormap the per-lift colours are taken from, see lift_styles.
    """
    p = np.asarray(p, dtype=np.float64) % 1.0
    q = np.asarray(q, dtype=np.float64) % 1.0

    if ax is None:
        _, ax = plt.subplots()

    styles = lift_styles(p, q, offsets, shortest_offset=shortest_offset, cmap=cmap)
    for offset, style in styles.items():
        for a, b in torus_edge_segments(p, q, offset=offset):
            ax.plot([a[0], b[0]], [a[1], b[1]], **style["kwargs"])

    corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    ax.add_patch(MplPolygon(corners, closed=True, fill=False,
                            edgecolor="gray", lw=1.2, zorder=1))

    ax.scatter([p[0], q[0]], [p[1], q[1]], s=s, c="black", zorder=5,
               edgecolors="white", linewidths=0.8)
    if label_points:
        _annotate_pair(ax, p, q)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-pad, 1 + pad)
    ax.set_ylim(-pad, 1 + pad)
    ax.axis("off")
    return ax


def plot_cover_lifts(p, q, offsets, ax=None, shortest_offset=None, cmap=LIFT_CMAP,
                     label_points=True, label_offsets=True, s=45, pad=0.0):
    """
    Draw the same geodesics unbroken in the universal cover of the torus.

    Each offset (m, n) puts an image of q in the tile (m, n), and the lift is then a single
    straight line from p to that image. The fundamental domain is outlined darker than the
    surrounding tiles. Arguments match plot_torus_lifts, which draws the folded view.
    """
    p = np.asarray(p, dtype=np.float64) % 1.0
    q = np.asarray(q, dtype=np.float64) % 1.0
    offsets = [(int(m), int(n)) for m, n in offsets]

    # Tile range of the cover, always including the fundamental domain itself.
    m_lo, m_hi = min([m for m, _ in offsets] + [0]), max([m for m, _ in offsets] + [0]) + 1
    n_lo, n_hi = min([n for _, n in offsets] + [0]), max([n for _, n in offsets] + [0]) + 1

    if ax is None:
        _, ax = plt.subplots()

    for x in range(m_lo, m_hi + 1):
        ax.plot([x, x], [n_lo, n_hi], color="lightgray", lw=0.8, zorder=0)
    for y in range(n_lo, n_hi + 1):
        ax.plot([m_lo, m_hi], [y, y], color="lightgray", lw=0.8, zorder=0)
    corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    ax.add_patch(MplPolygon(corners, closed=True, fill=False,
                            edgecolor="gray", lw=1.2, zorder=1))

    styles = lift_styles(p, q, offsets, shortest_offset=shortest_offset, cmap=cmap)
    images = np.array([q + np.asarray(o, dtype=np.float64) for o in offsets])
    for offset, target in zip(offsets, images):
        ax.plot([p[0], target[0]], [p[1], target[1]], **styles[offset]["kwargs"])
        # The (0, 0) image is q itself, which already carries its own label.
        if label_offsets and offset != (0, 0):
            ax.text(target[0] + 0.07, target[1] - 0.05, f"q",
                    fontsize=7, color=styles[offset]["kwargs"]["color"], ha="left", va="top", zorder=6)

    # Ring each image in its own lift's colour, so the two panels are keyed the same way.
    outer = np.array([o != (0, 0) for o in offsets])
    ax.scatter(images[outer, 0], images[outer, 1], s=s,
               color=[styles[o]["kwargs"]["color"] for o in offsets if o != (0, 0)],
               linewidths=0.0, zorder=4)
    ax.scatter([p[0], q[0]], [p[1], q[1]], s=s, c="black", zorder=5,
               edgecolors="white", linewidths=0.8)
    if label_points:
        _annotate_pair(ax, p, q)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(m_lo - pad, m_hi + pad)
    ax.set_ylim(n_lo - pad, n_hi + pad)
    ax.axis("off")
    return ax


def plot_embedding(
    X,
    G,
    s=10,
    node_alpha=0.9,
    edge_alpha=0.10,
    edge_lw=0.4,
    colors=None,
    ax=None,
):
    """
    Scatter plot of an embedding with straight edges (no torus wrapping).

    X : (N, 2) embedding — wrapped to [0,1]^2 for display.
    G : networkx graph whose node order matches rows of X.
    """
    X = np.asarray(X, dtype=np.float64) % 1.0
    idx = {n: i for i, n in enumerate(G.nodes())}

    if ax is None:
        _, ax = plt.subplots()

    for u, v in G.edges():
        i, j = idx[u], idx[v]
        p, q = X[i], X[j]
        ax.plot([p[0], q[0]], [p[1], q[1]],
                color="k", alpha=edge_alpha, lw=edge_lw, zorder=1)

    ax.scatter(X[:, 0], X[:, 1], s=s, alpha=node_alpha, zorder=2,
               c=colors if colors is not None else "blue")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    return ax


def shepard_diagram(
    X: np.ndarray,
    D: np.ndarray,
    geod,
    ax=None,
    s: float = 1,
    alpha: float = 0.2,
    label=None,
    max_pairs: int = 5000,
    seed: int = 42,
):
    """
    Scatter plot of input dissimilarities vs embedding distances (Shepard diagram).
    Points on the y=x diagonal indicate perfect distance preservation.
    Subsampled to max_pairs for readability.
    """
    hd = D[np.triu_indices_from(D, k=1)]
    ld = geodesic_matrix(X, geod)[np.triu_indices_from(D, k=1)]
    hd, ld = subsample(max_pairs, hd, ld, seed=seed)

    if ax is None:
        _, ax = plt.subplots()

    ax.scatter(hd, ld, s=s, alpha=alpha, label=label)
    lim_max = max(hd.max(), ld.max())
    ax.plot([0, lim_max], [0, lim_max], color="k", lw=1, linestyle="--")
    ax.set_xlabel("Input distance")
    ax.set_ylabel("Embedding distance")
    return ax
