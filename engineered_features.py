import torch

class FeatureCalculator:
    @staticmethod
    def _distance_to_gaze(
            vertices: torch.Tensor,
            *,
            Ks: tuple[int|None],
            percentiles: torch.Tensor,
            sigma: float
    ):
        device = vertices.device
        percentiles = percentiles.to(device)
        weights = vertices[:, -1].clamp(1e-12)
        dists = (-weights.log() * 2).sqrt() * sigma
        dists = dists.sort().values
        parts = []
        for k in Ks:
            d = dists[:k] if k else dists
            parts.append(torch.quantile(d, percentiles))
        parts.append(dists[:1])  # lowest distance as offset
        return torch.cat(parts)

    @staticmethod
    def _curvature_at_gaze(
            vertices: torch.Tensor,  # (N,7)
            edge_index: torch.Tensor,  # (2,E)
    ):
        device = vertices.device
        src, dst = edge_index.long()
        weights = vertices[:, 6]
        gazed_point = weights.sort().indices[-1]
        pos = vertices[:, :3]
        diff = pos[dst] - pos[src]
        laplacian = torch.zeros((vertices.shape[0], 3), device=device)
        laplacian = laplacian.index_add(0, src, diff)

        valence = torch.zeros(vertices.shape[0], device=device)
        valence = valence.index_add(0, src, torch.ones_like(src, dtype=torch.float))
        valence = valence.clamp(min=1)
        laplacian /= valence.unsqueeze(1)
        curvature_sgn = (laplacian * vertices[:, 3:6]).sum(dim=1)
        curv_mean = (curvature_sgn * weights).sum() / weights.sum()
        curv_at_gaze = curvature_sgn[gazed_point]
        topk = weights.topk(50).indices
        curv_local_var = curvature_sgn[topk].var()
        return torch.stack([curv_mean, curv_at_gaze, curv_local_var])

    @staticmethod
    def _centroid_dist(
            vertices: torch.Tensor,
    ):
        weights = vertices[:, 6]
        gazed_idx = weights.sort().indices[-1]
        gazed_point = vertices[gazed_idx, :3]
        centroid = vertices[:, :3].mean(dim=0)
        return (gazed_point - centroid).norm()

    def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            meta: torch.Tensor,
            batch: torch.Tensor = None,
            **kwargs
    ) -> torch.Tensor:
        device = x.device
        dist_to_gaze_kwargs = dict(
            Ks=kwargs.get('Ks', (50, 400, None)),
            percentiles=kwargs.get('percentiles', torch.tensor([.10, .50, .90])),
            sigma=kwargs.get('sigma', .1),
        )

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=device)

        if meta.dim() == 1:
            meta = meta.unsqueeze(0)

        B = int(batch.max().item()) + 1
        results = []
        for i in range(B):
            node_mask = batch == i
            vi = x[node_mask]
            offset = node_mask.nonzero(as_tuple=True)[0][0]
            edge_mask = batch[edge_index[0]] == i
            ei = edge_index[:, edge_mask] - offset
            feat = torch.cat([
                self._distance_to_gaze(vi, **dist_to_gaze_kwargs),
                self._curvature_at_gaze(vi, ei),
                self._centroid_dist(vi).unsqueeze(0),
                meta[i].ravel(),
            ])
            results.append(feat)

        return torch.stack(results)  # (B, F)


if __name__ == "__main__":
    from grasp_mesh_set import PrecachedMANOGraspDataset

    dataset = torch.load('.data_cache/data_src_test.pt', weights_only=False)
    sample = dataset[0]
    calc = FeatureCalculator()
    vec = calc.forward(sample.x, sample.edge_index, sample.meta)
    print(vec.shape, vec)
