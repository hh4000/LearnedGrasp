import os
from functools import cached_property
import pandas as pd
import torch
from pathlib import Path
from torch_geometric.data import Data, Dataset
import torch_geometric as T
import hashlib
import trimesh
from dataclasses import dataclass


@dataclass
class SampleData:
    obj_uid: str
    local_pos: tuple[torch.Tensor | None]
    meta_quat: torch.Tensor
    labels: torch.Tensor
    @property
    def hash(self):
        # Create a unique hash for the sample based on its attributes
        hasher = hashlib.sha256()
        hasher.update(self.obj_uid.encode('utf-8'))
        for p in self.local_pos:
            if p is None:
                p = torch.zeros(3)
            hasher.update(p.cpu().numpy().tobytes())

        hasher.update(self.meta_quat.cpu().numpy().tobytes())
        hasher.update(self.labels.cpu().numpy().tobytes())
        return hasher.hexdigest()[:16] # we take 16 numbers for simplicity -> 64 bits of uniqueness, which should be sufficient for our dataset size (approx 50% probability of repeat at 4.3 billion samples — 0.00005% chance at 100k)
@dataclass
class MeshData:
    path: Path
    uid: str
    meta_scale: torch.Tensor

class PrecachedMANOGraspDataset(Dataset):
    def __init__(
            self,
            root: str | Path ,
            mesh_data: list[MeshData],
            sample_data: list[SampleData],
            transform=None,
            pre_transform=None,
            force_overwrite: bool = False,
    ):
        """
        root: Directory where processed data will be stored.
        """

        if isinstance(root, str): root = Path(root)
        root = (root.expanduser().absolute())


        if isinstance(root, str): root = Path(root)
        self.meshes = mesh_data
        self.samples = sample_data
        self.sample_hashes = sorted([sample.hash for sample in self.samples])
        self.force_overwrite = force_overwrite
        super().__init__(str(root), transform, pre_transform)
    @cached_property
    def data_dims(self) -> int:
        s = self.get(0)
        return s.x.shape[1]
    @property
    def processed_file_names(self):
        # Returns a list of filenames to check for in self.processed_dir
        return [f'mesh_{mesh.uid}.pt' for mesh in self.meshes] + [f'sample_{sample.hash}.pt' for sample in self.samples]
    @staticmethod
    def gaussian_weight(pos: torch.Tensor, target_pos: tuple[torch.Tensor | None, ...], sigma: float =0.1) -> torch.Tensor:
        """
        Get gaussian weight of nodes throughout the mesh

        :param pos: node positions (N, 3)
        :param target_pos: tuple of target node positions — torch.Tensor (3,) or None
        :param sigma: gaussian standard deviation
        :return: weights (N, P)
        """
        N = pos.shape[0]
        P = len(target_pos)
        weights = torch.zeros(N,P)
        valid_idx = [i for i, v in enumerate(target_pos) if v is not None]
        stacked_target_poses = torch.cat([target_pos[i].unsqueeze(0) for i in valid_idx])
        _sigma = -2*sigma**2
        dist_sq = torch.sum((pos.unsqueeze(1)-stacked_target_poses.unsqueeze(0)).pow(2), dim=-1)
        w_valid = torch.exp(dist_sq /_sigma)
        weights[:, valid_idx] = w_valid
        return weights

    def process(self):
        # This method ONLY runs if the files in processed_file_names don't exist
        temp_mesh_pos = {}
        # PASS 1: Mesh-wise data
        for mesh_data in self.meshes:
            mesh_data: MeshData
            mesh_uid = mesh_data.uid
            out_path = Path(self.processed_dir) / f'mesh_{mesh_uid}.pt'
            if out_path.exists() and not self.force_overwrite:
                # Already cached — load pos so sample pass can still use it
                cached = torch.load(out_path, weights_only=False)
                temp_mesh_pos[mesh_uid] = cached['pos']
                continue
            mesh = trimesh.load(mesh_data.path, force="mesh")
            mesh: trimesh.Trimesh
            # 1. Geometry Extraction
            pos     = torch.tensor(mesh.vertices,       dtype=torch.float)
            normals = torch.tensor(mesh.vertex_normals, dtype=torch.float)
            temp_mesh_pos[mesh_uid] = pos

            # 2. Connectivity
            faces = torch.tensor(mesh.faces, dtype=torch.long).t()
            edges = torch.cat([faces[:2], faces[1:], faces[::2]], dim=1)
            edge_index = torch.cat([edges, edges.flip(0)], dim=1)
            edge_index = torch.unique(edge_index, dim=1)

            # 3. Metadata
            struct_data = dict(
                pos=pos,
                scale=mesh_data.meta_scale,
                normals=normals,
                edge_index=edge_index,
            )
            torch.save(struct_data, out_path)

        # PASS 2: Sample-wise data
        for sample in self.samples:
            out_path = Path(self.processed_dir) / f'sample_{sample.hash}.pt'
            if out_path.exists() and not self.force_overwrite: continue
            pos = temp_mesh_pos[sample.obj_uid]
            weights = self.gaussian_weight(pos, sample.local_pos)
            struct_data = dict(
                uid=sample.obj_uid,
                weights=weights,
                meta_quat=sample.meta_quat,
                labels=sample.labels,
            )
            torch.save(struct_data, out_path)



    def len(self):
        return len(self.samples)
    
    def get(self, idx):
        sample_hash = self.sample_hashes[idx]
        sample_data = torch.load(Path(self.processed_dir) / f'sample_{sample_hash}.pt', weights_only=False)
        mesh_data   = torch.load(Path(self.processed_dir) / f'mesh_{sample_data.pop("uid")}.pt', weights_only=False)
        node_data  = torch.cat([mesh_data.pop('pos'), mesh_data.pop('normals'), sample_data.pop('weights')], dim=-1)
        edge_index = mesh_data.pop('edge_index')
        meta_data  = torch.cat([mesh_data.pop('scale'), sample_data.pop('meta_quat')], dim=-1)
        labels     = sample_data.pop('labels')
        return Data(x=node_data, edge_index=edge_index, meta=meta_data.unsqueeze(0), y=labels.unsqueeze(0))

def get_scale_factor(
        mesh_dir: str | Path,
        output_path: str | Path = Path(".data_cache"),
        force_recalculate: bool = False
):
    """
    Gets the scale factor for
    """