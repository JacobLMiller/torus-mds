from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import LineCollection


from .geometry import torus_edge_segments
from .metrics import geodesic_matrix, subsample


def plot_embedding_with_torus_edges(X=None, G=None, outpath="output.png",
                                   s=10, node_alpha=0.9,
                                   edge_alpha=0.10, edge_lw=0.4,
                                   colors=None, cmap=None, vmin=None, vmax=None,
                                   torus=None,
                                   ax=None,
                                   n_ticks=3, tick_fontsize=6, show_tick_labels=True,
                                   rasterized=False):
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

    n_ticks / tick_fontsize / show_tick_labels control those edge ticks. Small panels
    need few, small labels: 5 labels at fontsize 8 collide below roughly 1.5 inches
    of panel width. show_tick_labels=False keeps the tick marks but drops the numbers.

    rasterized=True draws only the two heavy artists (the edge LineCollection and the
    scatter) into a raster layer, keeping the domain boundary, ticks and labels as
    vectors. Use it for pdf/svg output of large graphs, where tens of thousands of
    vector segments make the file slow to open. Their resolution is then set by the
    savefig dpi, so save with a print dpi (e.g. fig.savefig(..., dpi=600)).
    """
    # Resolve embedding
    if X is None:
        if torus is None or torus.torus_embedding_ is None:
            raise ValueError("Provide either X or a fitted torus object with torus_embedding_.")
        X = torus.torus_embedding_

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

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots()

    # edges — geodesic segments in [0,1)^2, split at torus-boundary crossings, mapped to
    # physical space, drawn as one batched LineCollection (vectorized torus_edge_segments)
    if G is not None and G.number_of_edges() > 0:
        ij = np.array([(idx[u], idx[v]) for u, v in G.edges()])
        p = X[ij[:, 0]]
        q = X[ij[:, 1]]
        d = ((q - p + 0.5) % 1.0) - 0.5          # shortest displacement per coordinate
        r = p + d
        E = len(p)
        # boundary-crossing parameter t in [0,1] per axis (0 = no crossing -> degenerate piece)
        cross = np.zeros((E, 2))
        for kk in (0, 1):
            up = (d[:, kk] > 0) & (r[:, kk] > 1.0)
            dn = (d[:, kk] < 0) & (r[:, kk] < 0.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                cross[up, kk] = np.clip(((1.0 - p[:, kk]) / d[:, kk])[up], 0.0, 1.0)
                cross[dn, kk] = np.clip(((0.0 - p[:, kk]) / d[:, kk])[dn], 0.0, 1.0)
        cross.sort(axis=1)
        bp = np.concatenate([np.zeros((E, 1)), cross, np.ones((E, 1))], axis=1)  # breakpoints
        segs = []
        for a_t, b_t in ((0, 1), (1, 2), (2, 3)):
            t0, t1 = bp[:, a_t], bp[:, b_t]
            keep = (t1 - t0) > 1e-9              # drop degenerate pieces
            if not keep.any():
                continue
            a = p[keep] + t0[keep, None] * d[keep]
            b = p[keep] + t1[keep, None] * d[keep]
            tile = np.floor(0.5 * (a + b))       # shift each piece into the fundamental domain
            a = np.clip(a - tile, 0.0, 1.0) @ M.T
            b = np.clip(b - tile, 0.0, 1.0) @ M.T
            segs.append(np.stack([a, b], axis=1))
        if segs:
            ax.add_collection(LineCollection(np.concatenate(segs, axis=0), colors="k",
                                             alpha=edge_alpha, linewidths=edge_lw, zorder=1,
                                             rasterized=rasterized))

    # points. Pass scalar `colors` with a `cmap` (not pre-mapped RGBA) so the
    # returned PathCollection is a proper ScalarMappable and plt.colorbar works.
    X_phys = X @ M.T
    ax.scatter(X_phys[:, 0], X_phys[:, 1], s=s, alpha=node_alpha, zorder=2,
               c=colors if colors is not None else "blue",
               cmap=cmap, vmin=vmin, vmax=vmax, rasterized=rasterized)

    # parallelogram boundary of the fundamental domain
    corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]) @ M.T
    ax.add_patch(MplPolygon(corners, closed=True, fill=False,
                            edgecolor='gray', lw=1, zorder=3))

    # 'box' (not 'datalim') so the axes box shrinks to the aspect of the torus instead
    # of inflating the data range, which would pad the panel with whitespace.
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')

    # --- Tick marks with physical-unit labels on parallelogram edges ---
    tick_ts   = np.linspace(0, 1, n_ticks)
    tick_size = 0.025 * max(r0, r1)   # tick length in physical units
    pad       = 0.03  * max(r0, r1)   # gap between tick tip and label

    e1_perp = np.array([0.0, -1.0])                      # outward normal to bottom side
    e2_end  = M @ np.array([0.0, 1.0])                   # physical tip of e2
    e2_perp = np.array([-np.sin(theta), np.cos(theta)])  # outward normal to left side

    # Bottom side (e1): ticks with labels showing physical distance from 0 to r0
    for t in tick_ts:
        pt = np.array([t * r0, 0.0])
        tip = pt + tick_size * e1_perp
        ax.plot([pt[0], tip[0]], [pt[1], tip[1]], color='dimgray', lw=0.8, zorder=4)
        if show_tick_labels:
            lbl_pos = tip + pad * e1_perp
            ax.text(lbl_pos[0], lbl_pos[1], f"{t * r0:.3g}",
                    ha='center', va='top', fontsize=tick_fontsize, color='dimgray')

    # Left side (e2): same ticks, labels showing physical distance from 0 to r1.
    # Skip the t=0 label — already covered by the "0" on the bottom side.
    for t in tick_ts:
        pt = t * e2_end
        tip = pt + tick_size * e2_perp
        ax.plot([pt[0], tip[0]], [pt[1], tip[1]], color='dimgray', lw=0.8, zorder=4)
        if show_tick_labels and t > 0:
            lbl_pos = tip + pad * e2_perp
            ax.text(lbl_pos[0], lbl_pos[1], f"{t * r1:.3g}",
                    ha='center', va='center', rotation=np.degrees(theta),
                    fontsize=tick_fontsize, color='dimgray')

    # Explicit limits: the tick labels are drawn with ax.text and do not enter the
    # data limits, so autoscaling would clip them. Extra room on the bottom/left,
    # where the labels sit.
    lo, hi = corners.min(axis=0), corners.max(axis=0)
    m = tick_size + pad
    label_room = 2.5 * m if show_tick_labels else 0.0
    ax.set_xlim(lo[0] - m - label_room, hi[0] + m)
    ax.set_ylim(lo[1] - m - label_room, hi[1] + m)

    # only lay out the figure when this call owns it; calling tight_layout on a
    # caller-supplied figure overrides its layout engine (e.g. constrained_layout)
    # and blows the spacing between panels apart.
    if own_fig:
        fig.tight_layout()
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
