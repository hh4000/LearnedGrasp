import argparse
from dataclasses import dataclass

import torch
from sklearn.neighbors import KDTree
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, fps
import time
import tqdm
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection


d_path = Path(".data_cache/data_src_train/processed")

# The cached .pt files store graph connectivity (edge_index, shape [2, E]),
# not triangle faces — the faces are discarded during preprocessing
# (see grasp_mesh_set.py). So each mesh is loaded as a torch_geometric graph
# rather than a trimesh.Trimesh.
@dataclass
class MeshSamplingData:
    indices: torch.Tensor
    edges: torch.Tensor


def build_sampling(vertices, n_points, k_neighbors):
    """Farthest-point downsample to n_points, then build an undirected kNN graph
    over the sampled points (edges in compact sampled-index space)."""
    sampled_indices = fps(vertices, ratio=n_points / len(vertices))
    kdtree = KDTree(vertices[sampled_indices])
    _, neighbors = kdtree.query(vertices[sampled_indices], k=k_neighbors + 1)
    edge_list = []
    seen = set()  # O(1) dedup of directed pairs instead of a linear list scan
    for idx_list in neighbors:
        idx_root = int(idx_list[0])
        for idx in idx_list[1:]:
            a, b = idx_root, int(idx)
            if (a, b) not in seen:
                # Mark both directions seen so the reverse pair (b, a) is also
                # skipped when b's neighbor list is processed later.
                seen.add((a, b))
                seen.add((b, a))
                edge_list.extend([[a, b], [b, a]])
    return MeshSamplingData(
        indices=sampled_indices.detach().clone(),
        edges=torch.tensor(edge_list).T,
    )


meshes          = []
mesh_data_2k    = []
mesh_data_500   = []
k_neighbors = 5
for f in d_path.glob("mesh_*.pt"):
    meshdata = torch.load(f)
    vertices = meshdata["pos"]
    normals = meshdata["normals"]
    edge_index = meshdata["edge_index"]
    mesh = Data(pos=vertices, normals=normals, edge_index=edge_index)
    meshes.append(mesh)
    mesh_data_2k.append(build_sampling(vertices, 2000, k_neighbors))
    mesh_data_500.append(build_sampling(vertices, 500, k_neighbors))


"""SOFT VALIDATION
# CHECK FIRST MESH FOR SOFT VALIDATION
MESH_IDX = 1  # which mesh to inspect — used for both the mesh and its samplings
first_mesh = meshes[MESH_IDX]

print(f"Loaded {len(meshes)} meshes")
print(f"First mesh: {first_mesh.num_nodes} nodes, {first_mesh.num_edges} edges")
print(f"  bounds: {first_mesh.pos.min(0).values.tolist()} -> {first_mesh.pos.max(0).values.tolist()}")
print(f"  extents: {(first_mesh.pos.max(0).values - first_mesh.pos.min(0).values).tolist()}")
print(f"  centroid: {first_mesh.pos.mean(0).tolist()}")

# Soft checks: warn instead of asserting so loading isn't blocked
if first_mesh.num_nodes == 0:
    print("  WARNING: mesh has no vertices")
if first_mesh.num_edges == 0:
    print("  WARNING: mesh has no edges")
if first_mesh.normals.shape != first_mesh.pos.shape:
    print("  WARNING: normals shape does not match vertex shape")
if first_mesh.edge_index.max() >= first_mesh.num_nodes:
    print("  WARNING: edge_index references a node outside the vertex set")


# PLOT FIRST MESH as a 3D wireframe (vertices + graph edges)
def plot_graph(ax, pos, edges, title):

    if len(edges) > 0:
        segments = pos[edges]  # [E, 2, 3]
        ax.add_collection3d(
            Line3DCollection(segments, colors="steelblue", linewidths=0.3, alpha=0.5)
        )
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=1, c="crimson")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    # Equal aspect ratio so the object isn't distorted
    ax.set_box_aspect(np.ptp(pos, axis=0))


def sampling_to_arrays(mesh, sampling):
    idx = sampling.indices.cpu().numpy()
    ds_pos = mesh.pos.cpu().numpy()[idx]
    ds_edges = sampling.edges.cpu().numpy().T  # [E, 2]
    return ds_pos, ds_edges


def view_pyvista(panels):
    import pyvista as pv

    plotter = pv.Plotter(shape=(1, len(panels)), window_size=[1800, 700])
    for i, (pos, edges, title) in enumerate(panels):
        plotter.subplot(0, i)
        pts = pos.astype(np.float32)
        plotter.add_mesh(pv.PolyData(pts), color="crimson", point_size=4,
                         render_points_as_spheres=True)
        if len(edges) > 0:
            # VTK line-cell format: [2, i0, j0, 2, i1, j1, ...]
            cells = np.hstack(
                [np.full((len(edges), 1), 2, dtype=np.int64), edges]
            ).ravel()
            wire = pv.PolyData(pts)
            wire.lines = cells
            plotter.add_mesh(wire, color="steelblue", line_width=1, opacity=0.5)
        plotter.add_text(title, font_size=10)
    plotter.link_views()  # shared camera across the three views
    plotter.show()


parser = argparse.ArgumentParser(description="Visualise a mesh at 3 resolutions.")
parser.add_argument("--pyvista", action="store_true",
                    help="Render interactively with PyVista (GPU/OpenGL) "
                         "instead of saving a Matplotlib PNG.")
args = parser.parse_args()

pos = first_mesh.pos.cpu().numpy()
edges = first_mesh.edge_index.cpu().numpy().T  # [E, 2]
pos_2k, edges_2k = sampling_to_arrays(first_mesh, mesh_data_2k[MESH_IDX])
pos_500, edges_500 = sampling_to_arrays(first_mesh, mesh_data_500[MESH_IDX])

panels = [
    (pos, edges,
     f"First mesh: {first_mesh.num_nodes} nodes, {first_mesh.num_edges} edges"),
    (pos_2k, edges_2k,
     f"Downsampled ~2k: {len(pos_2k)} nodes, {len(edges_2k)} edges"),
    (pos_500, edges_500,
     f"Downsampled ~500: {len(pos_500)} nodes, {len(edges_500)} edges"),
]

if args.pyvista:
    view_pyvista(panels)
else:
    fig = plt.figure(figsize=(24, 8))
    for i, (p, e, title) in enumerate(panels):
        ax = fig.add_subplot(1, len(panels), i + 1, projection="3d")
        plot_graph(ax, p, e, title)
    out_png = Path("first_mesh.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out_png.resolve()}")
    plt.show()
"""


# ====================================================================
# Node-wise contrastive graph encoding
# --------------------------------------------------------------------
# A GCN encoder maps every node (features = position ⊕ normal, 6-D) to a
# low-dim embedding. It is trained contrastively across all three mesh
# resolutions (full / 2k / 500) with, per anchor:
#   positives  P1 = a graph-near node (few hops),  P2 = the anchor under
#               a small feature augmentation (jittered pos + normal)
#   negatives  N1 = a random node on a *different* mesh,
#               N2 = a graph-far node on the *same* mesh (many hops)
# The anchor is duplicated to form every positive×negative triplet
# (4 per anchor), trained with a margin triplet loss.
# ====================================================================
import torch.nn as nn
import torch.nn.functional as F

# ---- hyper-parameters ----------------------------------------------
EMB_DIM    = 32     # output encoding dimensionality (compact but expressive)
HID_DIM    = 64
MARGIN     = 0.5
CLOSE_HOPS = 2      # P1 must be within this many hops of the anchor
FAR_HOPS   = 6      # N2 must be at least this many hops away
MAX_HOPS   = 12     # BFS horizon
POS_JITTER = 0.005  # std of positional augmentation noise
NRM_JITTER = 0.05   # std of normal augmentation noise


class GCNBranch(nn.Module):
    """3-layer GCN feature extractor for a single resolution."""

    def __init__(self, in_dim=6, hid=HID_DIM):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid)
        self.conv2 = GCNConv(hid, hid)
        self.conv3 = GCNConv(hid, hid)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        return x


class MultiScaleNodeEncoder(nn.Module):
    """One embedding per full-resolution node, fusing three scales at once:
      - the full graph        → fine *local* structure,
      - the 2k downsampled    → medium-range *regional* context,
      - the 500 downsampled   → coarse *global* context.
    Each scale has its own GCN branch; the coarse features are gathered back
    onto the full nodes (via nearest-sampled-node maps) and concatenated, then
    an MLP head fuses them into the final L2-normalised embedding."""

    def __init__(self, in_dim=6, hid=HID_DIM, out=EMB_DIM, n_coarse=2):
        super().__init__()
        self.full_branch = GCNBranch(in_dim, hid)
        self.coarse_branches = nn.ModuleList(
            [GCNBranch(in_dim, hid) for _ in range(n_coarse)]
        )
        self.head = nn.Sequential(
            nn.Linear(hid * (1 + n_coarse), hid),
            nn.ReLU(),
            nn.Linear(hid, out),
        )

    def forward(self, ms):
        # Local features on the full-resolution nodes.
        feats = [self.full_branch(ms["x"], ms["ei"])]            # [Nf, hid]
        # Regional/global features, gathered from each coarse graph.
        for branch, cg, cmap in zip(self.coarse_branches, ms["coarse"], ms["maps"]):
            fc = branch(cg["x"], cg["ei"])                       # [Nc, hid]
            feats.append(fc[cmap])                               # → [Nf, hid]
        fused = torch.cat(feats, dim=1)                          # [Nf, hid*(1+n_coarse)]
        return F.normalize(self.head(fused), dim=1)


def augment(x):
    """Slight feature augmentation: jitter positions and normals (re-normalised)."""
    pos = x[:, :3] + torch.randn_like(x[:, :3]) * POS_JITTER
    nrm = F.normalize(x[:, 3:] + torch.randn_like(x[:, 3:]) * NRM_JITTER, dim=1)
    return torch.cat([pos, nrm], dim=1)


def augment_ms(ms):
    """Augment input features at every scale; reuse graph structure and maps."""
    out = dict(ms)
    out["x"] = augment(ms["x"])
    out["coarse"] = [{"x": augment(c["x"]), "ei": c["ei"]} for c in ms["coarse"]]
    return out


def build_multiscale_graph(mesh, samplings, device):
    """Build a fused multi-resolution graph for one mesh:
      - the full graph (features, edges, sparse adjacency for hop sampling),
      - one coarse graph per sampling (2k, 500),
      - a map from each full node to its nearest coarse (sampled) node, so the
        coarse branches' features can be gathered back onto the full nodes.
    """
    x_full = torch.cat([mesh.pos, mesh.normals], dim=1).float().to(device)
    ei_full = mesh.edge_index.long().to(device)
    nf = x_full.size(0)
    adj = torch.sparse_coo_tensor(
        ei_full, torch.ones(ei_full.size(1), device=device), (nf, nf)
    ).coalesce()
    pos_full = mesh.pos.cpu().numpy()

    coarse, maps = [], []
    for samp in samplings:
        idx = samp.indices.long()
        xc = torch.cat([mesh.pos[idx], mesh.normals[idx]], dim=1).float().to(device)
        eic = samp.edges.long().to(device)
        coarse.append({"x": xc, "ei": eic})
        # Nearest sampled node for every full node (Euclidean in position).
        nn_idx = KDTree(mesh.pos[idx].cpu().numpy()).query(pos_full, k=1)[1][:, 0]
        maps.append(torch.as_tensor(nn_idx, dtype=torch.long, device=device))

    return {"x": x_full, "ei": ei_full, "adj": adj, "n": nf,
            "coarse": coarse, "maps": maps}


def hop_distances(adj, sources, n, max_h):
    """Multi-source BFS via sparse frontier propagation. Returns [n, B] hop
    distances from each source (−1 = unreached within max_h)."""
    dev = adj.device
    b = sources.numel()
    cols = torch.arange(b, device=dev)
    dist = torch.full((n, b), -1.0, device=dev)
    dist[sources, cols] = 0.0
    frontier = torch.zeros((n, b), device=dev)
    frontier[sources, cols] = 1.0
    visited = frontier.bool().clone()
    for h in range(1, max_h + 1):
        nxt = torch.sparse.mm(adj, frontier) > 0          # reachable in +1 hop
        new = nxt & ~visited
        if not new.any():
            break
        dist[new] = float(h)
        visited |= new
        frontier = new.float()
    return dist


def sample_triplet_indices(graph, batch, device):
    """Pick anchors and, per anchor, a near node (P1) and a far node (N2)."""
    n = graph["n"]
    b = min(batch, n)
    anchors = torch.randperm(n, device=device)[:b]
    dist = hop_distances(graph["adj"], anchors, n, MAX_HOPS)  # [n, b]

    keep, near, far = [], [], []
    for j in range(b):
        d = dist[:, j]
        near_pool = ((d >= 1) & (d <= CLOSE_HOPS)).nonzero(as_tuple=True)[0]
        if near_pool.numel() == 0:
            continue  # isolated node — skip
        far_pool = (d >= FAR_HOPS).nonzero(as_tuple=True)[0]
        if far_pool.numel() == 0:                       # graph smaller than FAR_HOPS
            unreached = (d < 0).nonzero(as_tuple=True)[0]
            far_pool = unreached if unreached.numel() else d.argmax().view(1)
        keep.append(anchors[j])
        near.append(near_pool[torch.randint(near_pool.numel(), (1,), device=device)])
        far.append(far_pool[torch.randint(far_pool.numel(), (1,), device=device)])

    if not keep:
        return None
    return (torch.stack(keep),
            torch.cat(near),
            torch.cat(far))


def fit_pca_rgb(all_emb):
    """Fit a single PCA→RGB mapping on the pooled embeddings of all meshes, so
    the colours mean the same thing across every subplot. Returns a function
    that maps an [N, D] embedding to [N, 3] RGB using the shared transform."""
    pooled = np.concatenate(all_emb, axis=0)
    mean = pooled.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(pooled - mean, full_matrices=False)
    components = vt[:3]                                   # shared PCA axes
    proj = (pooled - mean) @ components.T
    lo = proj.min(0)
    span = np.ptp(proj, axis=0) + 1e-9                    # shared colour range

    def to_rgb(emb):
        p = (emb - mean) @ components.T
        return (p - lo) / span

    return to_rgb


def visualize_embeddings(encoder, device, n_meshes=5, off_screen=False,
                         screenshot="embeddings.png"):
    """Render node embeddings of the first n_meshes full meshes, colouring each
    node by the PCA→RGB of its encoding, in a linked PyVista grid. The PCA is
    fit on all meshes' embeddings at once so colours are comparable across them."""
    import pyvista as pv

    encoder.eval()
    n_meshes = min(n_meshes, len(meshes))

    # Pass 1: encode every mesh and pool embeddings to fit one shared PCA.
    embeddings = []
    with torch.no_grad():
        for i in range(n_meshes):
            ms = build_multiscale_graph(
                meshes[i], (mesh_data_2k[i], mesh_data_500[i]), device)
            embeddings.append(encoder(ms).cpu().numpy())
    to_rgb = fit_pca_rgb(embeddings)

    # Pass 2: colour each mesh with the shared transform and render.
    plotter = pv.Plotter(shape=(1, n_meshes), window_size=[360 * n_meshes, 720],
                         off_screen=off_screen)
    for i in range(n_meshes):
        rgb = (to_rgb(embeddings[i]) * 255).clip(0, 255).astype(np.uint8)
        plotter.subplot(0, i)
        cloud = pv.PolyData(meshes[i].pos.cpu().numpy().astype(np.float32))
        plotter.add_mesh(cloud, scalars=rgb, rgb=True, point_size=6,
                         render_points_as_spheres=True)
        plotter.add_text(f"mesh {i}", font_size=9)
    plotter.link_views()
    if off_screen:
        plotter.screenshot(screenshot)
        print(f"Saved embedding visualization to {Path(screenshot).resolve()}")
    else:
        plotter.show()


def train(cfg):
    device = torch.device(cfg.device)
    print(f"Training multi-scale node encoder on {device} | emb_dim={cfg.emb}")

    # One encoder, fusing full + 2k + 500 for every node.
    encoder = MultiScaleNodeEncoder(in_dim=6, out=cfg.emb).to(device)
    optim = torch.optim.Adam(encoder.parameters(), lr=cfg.lr)

    # One fused multi-scale graph per mesh. Anchors/positives/negatives are
    # sampled on the full graph; the encoder pulls in the coarse context.
    graphs = [build_multiscale_graph(mesh, (mesh_data_2k[i], mesh_data_500[i]), device)
              for i, mesh in enumerate(meshes)]

    best, bad, best_state = float("inf"), 0, None
    history = []          # mean loss per epoch, for the loss curve
    best_epoch = -1
    for epoch in range(cfg.epochs):
        encoder.train()
        losses = []
        for _ in tqdm.trange(cfg.iters, desc=f"epoch {epoch}", leave=False):
            gi = torch.randint(len(graphs), (1,)).item()
            g = graphs[gi]
            # N1 source: a different mesh.
            gj = gi
            while gj == gi:
                gj = torch.randint(len(graphs), (1,)).item()
            g_neg = graphs[gj]

            tri = sample_triplet_indices(g, cfg.batch, device)
            if tri is None:
                continue
            anchors, near, far = tri

            emb = encoder(g)
            emb_aug = encoder(augment_ms(g))
            emb_neg = encoder(g_neg)
            neg_nodes = torch.randint(g_neg["n"], (anchors.numel(),), device=device)

            a = emb[anchors]            # anchor
            p1 = emb[near]              # positive: graph-near node
            p2 = emb_aug[anchors]       # positive: augmented same node
            n1 = emb_neg[neg_nodes]     # negative: other mesh
            n2 = emb[far]               # negative: graph-far node

            # Anchor duplication: every (positive × negative) combination.
            a_rep = torch.cat([a, a, a, a])
            pos = torch.cat([p1, p2, p1, p2])
            neg = torch.cat([n1, n1, n2, n2])
            loss = F.triplet_margin_loss(a_rep, pos, neg, margin=MARGIN)

            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(loss.item())

        epoch_loss = float(np.mean(losses)) if losses else float("nan")
        history.append(epoch_loss)
        print(f"epoch {epoch:3d} | loss {epoch_loss:.4f} | best {best:.4f}")

        # Early stopping: stop when the loss stops improving (stagnates/rises).
        if epoch_loss < best - cfg.min_delta:
            best, bad, best_epoch = epoch_loss, 0, epoch
            best_state = {k: v.detach().clone() for k, v in encoder.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                print(f"Early stopping at epoch {epoch} "
                      f"(no improvement for {cfg.patience} epochs).")
                break

    if best_state is not None:
        encoder.load_state_dict(best_state)
    return encoder, device, history, best_epoch


def plot_loss_curve(history, best_epoch, out_path):
    """Plot the per-epoch training loss and mark the best (saved) epoch."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(len(history)), history, marker="o", ms=3, label="train loss")
    if 0 <= best_epoch < len(history):
        ax.axvline(best_epoch, color="crimson", ls="--", alpha=0.7,
                   label=f"best (epoch {best_epoch}, {history[best_epoch]:.4f})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("triplet loss")
    ax.set_title("Node encoder training loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved loss curve to {Path(out_path).resolve()}")


def save_encoder(encoder, cfg, history, best_epoch, out_path):
    """Persist the trained encoder weights plus the config needed to rebuild it."""
    torch.save(
        {
            "state_dict": encoder.state_dict(),
            "arch": "MultiScaleNodeEncoder",
            "emb_dim": cfg.emb,
            "in_dim": 6,
            "hid_dim": HID_DIM,
            "n_coarse": 2,
            "history": history,
            "best_epoch": best_epoch,
        },
        out_path,
    )
    print(f"Saved encoder weights to {Path(out_path).resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train a node-wise graph encoder.")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--iters", type=int, default=40, help="iterations per epoch")
    ap.add_argument("--batch", type=int, default=64, help="anchors per iteration")
    ap.add_argument("--emb", type=int, default=EMB_DIM)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--min-delta", type=float, default=1e-4, dest="min_delta")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--offscreen", action="store_true",
                    help="save the PyVista visualization to a PNG instead of showing it")
    ap.add_argument("--out", default="embeddings.png")
    ap.add_argument("--ckpt", default="node_encoder.pt",
                    help="path to save the trained encoder weights")
    ap.add_argument("--loss-curve", default="loss_curve.png", dest="loss_curve",
                    help="path to save the training loss curve")
    cfg = ap.parse_args()

    encoder, device, history, best_epoch = train(cfg)
    save_encoder(encoder, cfg, history, best_epoch, cfg.ckpt)
    plot_loss_curve(history, best_epoch, cfg.loss_curve)
    visualize_embeddings(encoder, device, n_meshes=5,
                         off_screen=cfg.offscreen, screenshot=cfg.out)
