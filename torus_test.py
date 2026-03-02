import numpy as np 
import pylab as plt 
import networkx as nx

from projector import MDSTorusProjector, UMAPTorusProjector

def torus_delta(p, q):
    """Shortest displacement from p to q on unit square torus (per-coordinate in [-0.5, 0.5))."""
    return ((q - p + 0.5) % 1.0) - 0.5

def torus_edge_segments(p, q, eps=1e-12):
    """
    Return list of (a_plot, b_plot) segments to draw the shortest torus geodesic between p and q,
    plotted in the fundamental domain [0,1]x[0,1], split at boundary crossings.

    p, q are assumed to be in [0,1)^2.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    d = torus_delta(p, q)
    r = p + d  # endpoint in covering space (may lie outside [0,1) )

    ts = [0.0, 1.0]

    # boundary crossings for each coordinate
    for k in (0, 1):
        dk = d[k]
        if abs(dk) < eps:
            continue

        # Determine which boundary is crossed (if any) based on direction and endpoint r
        if dk > 0 and r[k] > 1.0 + eps:
            b = 1.0
        elif dk < 0 and r[k] < 0.0 - eps:
            b = 0.0
        else:
            continue

        t = (b - p[k]) / dk
        if eps < t < 1.0 - eps:
            ts.append(float(t))

    ts = sorted(set(ts))

    segs = []
    for t0, t1 in zip(ts[:-1], ts[1:]):
        a = p + t0 * d
        b = p + t1 * d

        # Choose a single tile shift so BOTH endpoints land in the same plotted tile.
        mid = 0.5 * (a + b)
        tile = np.floor(mid)  # integer vector, typically in {-1,0,1}^2 here

        a_plot = a - tile
        b_plot = b - tile

        # Numerical guard: keep within [0,1] for plotting
        a_plot = np.clip(a_plot, 0.0, 1.0)
        b_plot = np.clip(b_plot, 0.0, 1.0)

        segs.append((a_plot, b_plot))

    return segs

def plot_embedding_with_torus_edges(X, G, outpath="output.png",
                                   s=10, node_alpha=0.9,
                                   edge_alpha=0.10, edge_lw=0.4, 
                                   ax=None):
    """
    Scatter + edges drawn along shortest torus paths in [0,1)^2.

    X: (N,2) embedding (can be unwrapped; we'll wrap for plotting)
    nodes: list of nodes matching row order of X
    G: networkx graph
    """
    X = np.asarray(X, dtype=np.float64) % 1.0
    idx = {n: i for i, n in enumerate(G.nodes())}

    if ax == None: 
        fig, ax = plt.subplots()
    

    # edges
    for u, v in G.edges():
        i, j = idx[u], idx[v]
        p, q = X[i], X[j]
        for a, b in torus_edge_segments(p, q):
            ax.plot([a[0], b[0]], [a[1], b[1]],
                     color="k", alpha=edge_alpha, lw=edge_lw, zorder=1)

    # points
    ax.scatter(X[:, 0], X[:, 1], s=s, alpha=node_alpha, zorder=2)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    # plt.savefig(outpath, dpi=200)
    # plt.close()
    return ax

def periodic_lattice_graph(nx_size: int, ny_size: int, diagonal: bool = False) -> nx.Graph:
    """
    Create a 2D periodic lattice (torus) graph with NetworkX.

    Nodes are (i, j) tuples. Periodic boundaries wrap in both directions.
    If diagonal=True, add 4 diagonal neighbors (8-neighborhood total).
    """
    G = nx.grid_2d_graph(nx_size, ny_size, periodic=True)

    if diagonal:
        # Add diagonal wrap-around edges
        for i in range(nx_size):
            for j in range(ny_size):
                G.add_edge((i, j), ((i + 1) % nx_size, (j + 1) % ny_size))
                G.add_edge((i, j), ((i + 1) % nx_size, (j - 1) % ny_size))

    # Unit weights (optional but explicit)
    nx.set_edge_attributes(G, 1.0, "weight")
    return G

def apsp_distance_matrix(G: nx.Graph, weight: str = "weight") -> tuple[np.ndarray, list]:
    """
    Compute all-pairs shortest path (APSP) distance matrix using NetworkX.

    Returns:
    D: (N,N) float64 distance matrix
    nodes: list of nodes in the order used for D
    """
    nodes = list(G.nodes())
    idx = {n: k for k, n in enumerate(nodes)}
    n = len(nodes)
    D = np.zeros((n, n), dtype=np.float64)

    # all_pairs_dijkstra_path_length works for unit weights too
    for src, dist_dict in nx.all_pairs_dijkstra_path_length(G, weight=weight):
        i = idx[src]
        for dst, d in dist_dict.items():
            D[i, idx[dst]] = float(d)

    return D, nodes

def get_periodic_lattice(nx_size=20,ny_size=20):

    G = periodic_lattice_graph(nx_size, ny_size, diagonal=False)
    D, nodes = apsp_distance_matrix(G)
    return G,D

def MDS_test():
    from sklearn.manifold import MDS

    G,D = get_periodic_lattice(20,20)

    # D /= np.max(D)
    print(np.max(D))

    X_torus = MDSTorusProjector().fit_transform(D)
    X_euc   = MDS(dissimilarity="precomputed").fit_transform(D)

    fig,(ax1,ax2) = plt.subplots(1,2)

    plot_embedding_with_torus_edges(X_torus, G, ax=ax1)

    idx = {n: i for i, n in enumerate(G.nodes())}    
    for u, v in G.edges():
        i, j = idx[u], idx[v]
        p, q = X_euc[i], X_euc[j]
        ax2.plot([p[0], q[0]], [p[1], q[1]],
                    color="k", alpha=0.1, lw=0.4, zorder=1)
    ax2.scatter(X_euc[:, 0], X_euc[:, 1], s=10, alpha=0.9, zorder=2)

    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title("Toroidal MDS (1/t+1 lr)")


    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title("Euclidean MDS (sklearn)")

    fig.savefig("mds_out.png")

def UMAP_test(): 
    from umap import UMAP 

    G,D = get_periodic_lattice()
    # D /= np.max(D)

    X_torus = UMAPTorusProjector().fit_transform(D)
    X_euc   = UMAP(metric="precomputed").fit_transform(D)


    fig,(ax1,ax2) = plt.subplots(1,2)

    plot_embedding_with_torus_edges(X_torus, G, ax=ax1)

    idx = {n: i for i, n in enumerate(G.nodes())}    
    for u, v in G.edges():
        i, j = idx[u], idx[v]
        p, q = X_euc[i], X_euc[j]
        ax2.plot([p[0], q[0]], [p[1], q[1]],
                    color="k", alpha=0.1, lw=0.4, zorder=1)
    ax2.scatter(X_euc[:, 0], X_euc[:, 1], s=10, alpha=0.9, zorder=2)

    fig.savefig("umap_out.png")

import os
import glob
import numpy as np
from PIL import Image

def load_coil(root, size=(64, 64), as_gray=True):
    """
    root: folder containing the COIL images, e.g. coil-20-proc/ or coil-100/
    Returns:
      X: (N, H*W) float32 in [0,1]
      y: (N,) object id
      angle: (N,) rotation index (derived from filename)
      paths: list of file paths
    """
    exts = ("*.png", "*.jpg", "*.ppm", "*.pgm", "*.tif", "*.bmp")
    paths = []
    for e in exts:
        paths += glob.glob(os.path.join(root, "**", e), recursive=True)

    if not paths:
        raise FileNotFoundError(f"No images found under: {root}")

    paths = sorted(paths)

    X = []
    y = []
    angle = []

    for p in paths:
        fn = os.path.basename(p).lower()


        if "obj" in fn and "__" in fn:
            left, right = fn.split("__", 1)
            obj_id = int(left.replace("obj", ""))
            ang = int("".join([c for c in right.split(".", 1)[0] if c.isdigit()]))
        else:
            obj_id = -1
            ang = -1

        img = Image.open(p)
        if as_gray:
            img = img.convert("L")
        img = img.resize(size, Image.BILINEAR)

        arr = np.asarray(img, dtype=np.float32) / 255.0
        X.append(arr.reshape(-1))
        y.append(obj_id)
        angle.append(ang)

    return np.stack(X), np.array(y), np.array(angle), paths

def UMAP_coil():
    from umap import UMAP
    X, y, angle, paths = load_coil("coil-20-proc")

    from sklearn.metrics import pairwise_distances
    D = pairwise_distances(X)
    D /= np.max(D)

    X_torus = UMAPTorusProjector().fit_transform(D)
    X_euc   = UMAP(metric="precomputed").fit_transform(D)


    fig,(ax1,ax2) = plt.subplots(1,2)

    ax1.scatter(X_torus[:,0], X_torus[:,1], s=10, alpha=0.9,c=y)
    ax2.scatter(X_euc[:, 0], X_euc[:, 1], s=10, alpha=0.9, c=y)

    ax2.legend()
    fig.savefig("umap_coil.png")

def UMAP_mnist():
    from umap import UMAP
    from sklearn.datasets import load_digits
    X,y = load_digits(return_X_y=True)

    from sklearn.metrics import pairwise_distances
    D = pairwise_distances(X)
    D /= np.max(D)

    X_torus = UMAPTorusProjector().fit_transform(D)
    X_euc   = UMAP(metric="precomputed").fit_transform(D)


    fig,(ax1,ax2) = plt.subplots(1,2,sharex=None, sharey=None)

    ax2.scatter(X_euc[:,0], X_euc[:,1], alpha=0.1)
    for i in range(X_torus.shape[0]):
        ax1.text(X_torus[i,0], X_torus[i,1], str(int(y[i])))
        ax2.text(X_euc[i,0], X_euc[i,1], str(int(y[i])))
    # sc = ax2.scatter(X_euc[:, 0], X_euc[:, 1], s=10, alpha=0.9, c=y)



    # ax2.legend()
    fig.savefig("umap_mnist.png")


if __name__ == "__main__": 
    import os 
    from sklearn.metrics import pairwise_distances
    from sklearn.manifold import MDS
    from umap import UMAP


    # MDS_test()
    # data = np.load("grid_cells_4800.npz")['data']
    # # data = a['data']
    # # spikes = a['spikes']
    # print(data.shape)
    # # from sklearn.decomposition import PCA

    # # pca = PCA(n_components=10).fit_transform(data)

    # D = pairwise_distances(data)

    # # from sklearn.decomposition import PCA
    # # pca = PCA(2).fit_transform(data)

    # # c1 = pca[:, 0]
    # # c2 = pca[:, 1]

    # # def norm01(v):
    # #     return (v - v.min()) / (v.max() - v.min())

    # # r = norm01(c1)          
    # # b = norm01(c2)          
    # # g = np.zeros_like(r) + 0.2

    # # colors = np.stack([r, g, b], axis=1)  

    # # X_torus = UMAPTorusProjector().fit_transform(D)

    # # np.save("yoda_umap.npy",X_torus)
    # X_torus = MDSTorusProjector().fit_transform(D)
    # np.save("rat_mds.npy",X_torus)



    #Toroidal random graphs 
    #Chen's block graph datasets
    #Find some real-world small-world