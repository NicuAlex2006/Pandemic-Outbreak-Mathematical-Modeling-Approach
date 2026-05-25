"""
Discrete stochastic optimal stopping for pandemic control via Snell envelope
on a binomial tree of transmission rates.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Binomial tree construction
# ---------------------------------------------------------------------------

def build_beta_tree(beta_0, u, d, N):
    """Upper-triangular (N+1)x(N+1) array: beta_tree[j, k] = beta after j ups at step k."""
    beta_tree = np.zeros((N + 1, N + 1))
    for k in range(N + 1):
        j = np.arange(k + 1)
        beta_tree[j, k] = beta_0 * (u ** j) * (d ** (k - j))
    return beta_tree


# ---------------------------------------------------------------------------
# Cost and Snell envelope
# ---------------------------------------------------------------------------

def compute_cost(beta_tree, N):
    k_array = np.arange(N + 1)[np.newaxis, :]
    return beta_tree * np.exp(0.1 * k_array)


def snell_envelope(g_tree, N, p=0.5):
    U = np.zeros_like(g_tree)
    U[:, N] = g_tree[:, N]

    for k in range(N - 1, -1, -1):
        for j in range(k + 1):
            expected = p * U[j + 1, k + 1] + (1 - p) * U[j, k + 1]
            U[j, k] = max(g_tree[j, k], expected)

    stop_mask = np.isclose(U, g_tree) & (beta_tree_nonzero_mask(g_tree, N))
    return U, stop_mask


def beta_tree_nonzero_mask(g_tree, N):
    mask = np.zeros_like(g_tree, dtype=bool)
    for k in range(N + 1):
        mask[:k + 1, k] = True
    return mask


# ---------------------------------------------------------------------------
# Visualisation (small tree only)
# ---------------------------------------------------------------------------

def draw_tree(beta_tree, U, stop_mask, N):
    G = nx.DiGraph()
    pos = {}
    colors = []
    labels = {}

    for k in range(N + 1):
        for j in range(k + 1):
            node = (j, k)
            G.add_node(node)
            pos[node] = (k, j - k / 2)
            colors.append("#b2182b" if stop_mask[j, k] else "#2166ac")
            labels[node] = f"β={beta_tree[j,k]:.2f}\nU={U[j,k]:.2f}"

            if k < N:
                G.add_edge(node, (j + 1, k + 1))  # up
                G.add_edge(node, (j, k + 1))       # down

    fig, ax = plt.subplots(figsize=(max(12, N * 2.2), max(6, N * 1.1)))
    nx.draw(G, pos, ax=ax, node_color=colors, node_size=900,
            font_size=6, arrows=True, arrowsize=10, width=0.6,
            edge_color="#999999")
    nx.draw_networkx_labels(G, pos, labels, font_size=5, ax=ax)
    ax.set_title(f"Snell Envelope (N={N})  —  red = stop, blue = continue")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

BETA_0 = 0.4
U_FACTOR = 2.0
D_FACTOR = 0.5
P = 0.5

if __name__ == "__main__":
    # --- Large tree for performance ---
    N_large = 50
    bt_large = build_beta_tree(BETA_0, U_FACTOR, D_FACTOR, N_large)
    g_large = compute_cost(bt_large, N_large)
    U_large, mask_large = snell_envelope(g_large, N_large, P)
    print(f"N={N_large}: U[0,0] = {U_large[0, 0]:.6f}")

    active = sum(1 for k in range(N_large + 1) for j in range(k + 1))
    stopped = sum(1 for k in range(N_large + 1) for j in range(k + 1) if mask_large[j, k])
    print(f"  Nodes: {active}, Stopping: {stopped} ({stopped/active:.1%})")

    # --- Small tree for visualisation ---
    N_small = 6
    bt_small = build_beta_tree(BETA_0, U_FACTOR, D_FACTOR, N_small)
    g_small = compute_cost(bt_small, N_small)
    U_small, mask_small = snell_envelope(g_small, N_small, P)
    print(f"\nN={N_small}: U[0,0] = {U_small[0, 0]:.6f}")

    fig = draw_tree(bt_small, U_small, mask_small, N_small)
    plt.show()
