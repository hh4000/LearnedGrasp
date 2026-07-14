import torch
from torch import Tensor, nn, cat
from torch_geometric.nn import GCNConv, EdgeConv, global_max_pool as graph_global_max_pool, global_mean_pool as graph_global_mean_pool, fps

from torch_geometric.data import Data as GeoData
import torch.nn.functional as F

class GraspCNNv1(nn.Module):
    def __init__(self, dropout2d: float = 0.3, dropoutfc: float = 0.5):
        super().__init__()
        if not (0 <= dropout2d < 1):
            raise ValueError("[GraspCNN]:\tdropout2d must be between 0 and 1")
        if not (0 <= dropoutfc < 1):
            raise ValueError("[GraspCNN]:\tdropoutfc must be between 0 and 1")
        # conv stacks
        self.seq1 = nn.Sequential(
            # size 4 x 128² px
            nn.Conv2d(4, 64, kernel_size=7, padding=3),
            nn.BatchNorm2d(64),
            # size 64 x 128² px
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # size 64 x 64² px
            nn.Dropout2d(p=dropout2d),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            # size 128 x 64² px
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # size 128 x 32² px

            nn.Dropout2d(p=dropout2d),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            # size 128 x 32² px
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2, stride=2),
            # size 128 x 32² px
            nn.Dropout2d(p=dropout2d),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            # size 128 x 32² px
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2, stride=2),
            # size 128 x 32² px
            nn.Dropout2d(p=dropout2d),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            # size 128
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p=dropoutfc),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=dropoutfc),
            nn.Linear(64, 3),
        )
    def forward(self, x):
        logits = self.seq1(x)
        return logits
class GraspCNNv2(nn.Module):
    def __init__(self, dropout2d: float = 0.3, dropoutfc: float = 0.5):
        super().__init__()
        if not (0 <= dropout2d < 1):
            raise ValueError("[GraspCNN]:\tdropout2d must be between 0 and 1")
        if not (0 <= dropoutfc < 1):
            raise ValueError("[GraspCNN]:\tdropoutfc must be between 0 and 1")
        # conv stacks
        self.seq1 = nn.Sequential(

            # size 4 x 128² px
            nn.Conv2d(4, 64, kernel_size=7, padding=3),
            # size 64 x 128² px
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            # size 128 x 128² px
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # size 128 x 64² px

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            # size 128 x 64² px
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            # size 128 x 64² px
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # size 128 x 32² px
            nn.Dropout2d(p=dropout2d),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(p=dropout2d),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Dropout(p=dropoutfc),
            # size 2048
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(p=dropoutfc),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=dropoutfc),
            nn.Linear(128, 3),
        )
    def forward(self, x):
        logits = self.seq1(x)
        return logits
class GraspCNNv3(nn.Module):
    def __init__(self, dropout2d: float = 0.3, dropoutfc: float = 0.5):
        super().__init__()
        if not (0 <= dropout2d < 1):
            raise ValueError("[GraspCNN]:\tdropout2d must be between 0 and 1")
        if not (0 <= dropoutfc < 1):
            raise ValueError("[GraspCNN]:\tdropoutfc must be between 0 and 1")

        # RGB stream: first conv block at full resolution
        self.rgb_block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # size 64 x 64²
        )

        # Gaze stream: produces spatial attention mask at same resolution
        self.gaze_block1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # size 16 x 64²
            nn.Conv2d(16, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Sigmoid(),
            # size 64 x 64² — one attention weight per RGB feature map
        )

        # Shared trunk: processes attended RGB features
        self.trunk = nn.Sequential(
            # size 64 x 64²
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # size 128 x 32²
            nn.Dropout2d(p=dropout2d),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(p=dropout2d),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(p=dropout2d),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            # size 128
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p=dropoutfc),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=dropoutfc),
            nn.Linear(64, 3),
        )

    def forward(self, x):
        rgb  = x[:, :3]  # (B, 3, H, W)
        gaze = x[:, 3:]  # (B, 1, H, W)

        rgb_feat  = self.rgb_block1(rgb)    # (B, 64, 64, 64)
        attention = self.gaze_block1(gaze)  # (B, 64, 64, 64)

        attended = rgb_feat * attention     # spatial gating
        return self.trunk(attended)
class MANOGraspCNNv1(nn.Module):
    def __init__(self, dropout2d: float = 0.3, dropoutfc: float = 0.5, n_values: int = 15):
        super().__init__()
        if not (0 <= dropout2d < 1):
            raise ValueError("[GraspCNN]:\tdropout2d must be between 0 and 1")
        if not (0 <= dropoutfc < 1):
            raise ValueError("[GraspCNN]:\tdropoutfc must be between 0 and 1")
        self.n_values = n_values

        # RGB stream: first conv block at full resolution
        self.rgb_block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # size 64 x 64²
        )

        # Gaze stream: produces spatial attention mask at same resolution
        self.gaze_block1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # size 16 x 64²
            nn.Conv2d(16, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Sigmoid(),
            # size 64 x 64² — one attention weight per RGB feature map
        )
        # Shared trunk: processes attended RGB features
        self.trunk = nn.Sequential(
            # size 64 x 64²
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # size 128 x 32²
            nn.Dropout2d(p=dropout2d),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(p=dropout2d),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(p=dropout2d),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            # size 128
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p=dropoutfc),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=dropoutfc),
            nn.Linear(64, n_values*2), # split into mean and logvar
        )
        nn.init.zeros_(self.trunk[-1].bias)
        nn.init.normal_(self.trunk[-1].weight, mean=0, std = 0.01)


    def forward(self, x) -> tuple[Tensor, Tensor]:
        rgb  = x[:, :3]  # (B, 3, H, W)
        gaze = x[:, 3:]  # (B, 1, H, W)

        rgb_feat  = self.rgb_block1(rgb)    # (B, 64, 64, 64)
        attention = self.gaze_block1(gaze)  # (B, 64, 64, 64)

        attended = rgb_feat * attention     # spatial gating
        out = self.trunk(attended)
        pred_mean = out [:, self.n_values:]
        pred_logvar = out [:, :self.n_values]
        return (pred_mean, pred_logvar)
class MANOGraspCNNv2(nn.Module):
    def __init__(self, dropout2d: float = 0.3, dropoutfc: float = 0.5, n_values: int = 15):
        super().__init__()
        if not (0 <= dropout2d < 1):
            raise ValueError("[GraspCNN]:\tdropout2d must be between 0 and 1")
        if not (0 <= dropoutfc < 1):
            raise ValueError("[GraspCNN]:\tdropoutfc must be between 0 and 1")
        n_values = int(n_values)
        self.n_values = n_values

        # RGB stream: first conv block at full resolution
        self.rgb_block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # size 64 x 64²
        )

        # Gaze stream: produces spatial attention mask at same resolution
        self.gaze_block1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # size 16 x 64²
            nn.Conv2d(16, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Sigmoid(),
            # size 64 x 64² — one attention weight per RGB feature map
        )
        # Shared trunk: processes attended RGB features
        self.trunk = nn.Sequential(
            # size 64 x 64²
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # size 128 x 32²
            nn.Dropout2d(p=dropout2d),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(p=dropout2d),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(p=dropout2d),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            # size 128
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p=dropoutfc),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=dropoutfc),
            nn.Linear(64, n_values),
        )
        nn.init.zeros_(self.trunk[-1].bias)
        nn.init.normal_(self.trunk[-1].weight, mean=0, std = 0.01)


    def forward(self, x) -> tuple[Tensor, Tensor]:
        rgb  = x[:, :3]  # (B, 3, H, W)
        gaze = x[:, 3:]  # (B, 1, H, W)

        rgb_feat  = self.rgb_block1(rgb)    # (B, 64, 64, 64)
        attention = self.gaze_block1(gaze)  # (B, 64, 64, 64)

        attended = rgb_feat * attention     # spatial gating
        out = self.trunk(attended)
        return out
class MANOGraspCNNv3(nn.Module):
    def __init__(self, dropout2d: float = 0.3, dropoutfc: float = 0.5, n_values: int = 3):
        super().__init__()
        if not (0 <= dropout2d < 1):
            raise ValueError("[GraspCNN]:\tdropout2d must be between 0 and 1")
        if not (0 <= dropoutfc < 1):
            raise ValueError("[GraspCNN]:\tdropoutfc must be between 0 and 1")
        n_values = int(n_values)
        self.n_values = n_values

        # RGB stream: first conv block at full resolution
        self.rgb_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # size 64 x 64²
        )

        # Gaze stream: produces spatial attention mask at same resolution
        self.gaze_block1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # size 16 x 64²
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Sigmoid(),
            # size 64 x 64² — one attention weight per RGB feature map
        )
        # Shared trunk: processes attended RGB features
        self.trunk = nn.Sequential(
            # size 64 x 64²
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # size 128 x 32²
            nn.Dropout2d(p=dropout2d),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(p=dropout2d),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(p=dropout2d),
            nn.AdaptiveAvgPool2d((2,2)),
            nn.Flatten(),
            # size 128
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=dropoutfc),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=dropoutfc),
            nn.Linear(64, n_values),
        )
        nn.init.zeros_(self.trunk[-1].bias)
        nn.init.normal_(self.trunk[-1].weight, mean=0, std = 0.01)


    def forward(self, x) -> tuple[Tensor, Tensor]:
        rgb  = x[:, :3]  # (B, 3, H, W)
        gaze = x[:, 3:]  # (B, 1, H, W)

        rgb_feat  = self.rgb_block1(rgb)    # (B, 64, 64, 64)
        attention = self.gaze_block1(gaze)  # (B, 64, 64, 64)

        attended = rgb_feat * attention     # spatial gating
        out = self.trunk(attended)
        return out
class MANOGraspGNNv1(nn.Module):
    """
    First iteration of MANO-pose prediction from graph.
    object mesh vertices are encoded into shape (N,7) with vertex 3D pose, normal vector, and node weight (node weight is dependent on gaze point proximity)
    """
    def __init__(self, dropoutgnn=0.0, dropoutfc=0.0, n_in = 7, n_out = 3):
        super().__init__()
        if not (0 <= dropoutgnn < 1):
            raise ValueError("[GraspGNN]:\tdropoutgnn must be between 0 and 1")
        if not (0 <= dropoutfc < 1):
            raise ValueError("[GraspGNN]:\tdropoutfc must be between 0 and 1")
        self.dropoutgnn = dropoutgnn
        self.dropoutfc = dropoutfc
        self.gnns = nn.ModuleList([
            GCNConv(n_in, 64),
            GCNConv(64, 64),
            GCNConv(64, 64),
        ])
        self.regs = nn.ModuleList([
            nn.Sequential(
                nn.ReLU(),
                nn.Dropout(p=dropoutgnn),
            ),
            nn.Sequential(
                nn.ReLU(),
                nn.Dropout(p=dropoutgnn),
            ),
            nn.Identity()
        ])

        self.mlp = nn.Sequential(
            nn.Linear(64*2 + 5, 128), # 64 local + 64 global + 5 meta features (quaternions + normalized scale factor
            nn.ReLU(),
            nn.Dropout(p=dropoutfc),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=dropoutfc),
            nn.Linear(64, n_out)
        )
    def forward(self, data: GeoData):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Per-graph index of the node with highest gaze weight (last input feature)
        gaze = data.x[:, -1]
        B = int(batch.max().item()) + 1
        max_gaze = gaze.new_full((B,), float('-inf'))
        max_gaze.scatter_reduce_(0, batch, gaze, reduce='amax', include_self=True)
        is_max = (gaze == max_gaze[batch])
        node_idx = torch.arange(gaze.shape[0], device=gaze.device, dtype=torch.long)
        gazed_indices = batch.new_zeros(B)
        gazed_indices.scatter_(0, batch[is_max], node_idx[is_max])

        for gnn, reg in zip(self.gnns, self.regs):
            x = gnn(x, edge_index)
            x = reg(x)

        x_local:Tensor  = x[gazed_indices]                    # (B, 64)
        x_global        = graph_global_max_pool(x, batch)     # (B, 64)
        x_cat           = cat([x_local, x_global, data.meta], dim=1)  # (B, 133)
        return self.mlp(x_cat)
class MANOGraspGNNv2(nn.Module):
    """
    Updates:
      - Uses EdgeConv instead of GCN
    """
    def __init__(self, dropoutgnn=0.0, dropoutfc=0.0, n_in = 7, n_out = 3):
        super().__init__()
        if not (0 <= dropoutgnn < 1):
            raise ValueError("[GraspGNN]:\tdropoutgnn must be between 0 and 1")
        if not (0 <= dropoutfc < 1):
            raise ValueError("[GraspGNN]:\tdropoutfc must be between 0 and 1")
        self.dropoutgnn = dropoutgnn
        self.dropoutfc = dropoutfc
        self.gnns = nn.ModuleList([
            EdgeConv(self._edge_mlp(n_in,64), aggr='max'),
            EdgeConv(self._edge_mlp(64,64), aggr='max'),
            EdgeConv(self._edge_mlp(64,64), aggr='max'),
        ])
        self.regs = nn.ModuleList([
            nn.Sequential(
                nn.ReLU(),
                nn.Dropout(p=dropoutgnn),
            ),
            nn.Sequential(
                nn.ReLU(),
                nn.Dropout(p=dropoutgnn),
            ),
            nn.Identity()
        ])

        self.mlp = nn.Sequential(
            nn.Linear(64*2 + 5, 128), # 64 local + 64 global + 5 meta features (quaternions + normalized scale factor
            nn.ReLU(),
            nn.Dropout(p=dropoutfc),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=dropoutfc),
            nn.Linear(64, n_out)
        )
    @staticmethod
    def _edge_mlp(in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(2*in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
    def forward(self, data: GeoData):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Per-graph index of the node with highest gaze weight (last input feature)
        gaze = data.x[:, -1]
        B = int(batch.max().item()) + 1
        max_gaze = gaze.new_full((B,), float('-inf'))
        max_gaze.scatter_reduce_(0, batch, gaze, reduce='amax', include_self=True)
        is_max = (gaze == max_gaze[batch])
        node_idx = torch.arange(gaze.shape[0], device=gaze.device, dtype=torch.long)
        gazed_indices = batch.new_zeros(B)
        gazed_indices.scatter_(0, batch[is_max], node_idx[is_max])

        for gnn, reg in zip(self.gnns, self.regs):
            x = gnn(x, edge_index)
            x = reg(x)

        x_local:Tensor  = x[gazed_indices]                    # (B, 64)
        x_global        = graph_global_max_pool(x, batch)     # (B, 64)
        x_cat           = cat([x_local, x_global, data.meta], dim=1)  # (B, 133)
        return self.mlp(x_cat)
class MANOGraspGNNv3(nn.Module):
    """
    Updates:
      - Includes mean pool into mlp layer
    """
    def __init__(self, dropoutgnn=0.0, dropoutfc=0.0, n_in = 7, n_out = 3):
        super().__init__()
        if not (0 <= dropoutgnn < 1):
            raise ValueError("[GraspGNN]:\tdropoutgnn must be between 0 and 1")
        if not (0 <= dropoutfc < 1):
            raise ValueError("[GraspGNN]:\tdropoutfc must be between 0 and 1")
        self.dropoutgnn = dropoutgnn
        self.dropoutfc = dropoutfc
        self.gnns = nn.ModuleList([
            EdgeConv(self._edge_mlp(n_in,64), aggr='max'),
            EdgeConv(self._edge_mlp(64,64), aggr='max'),
            EdgeConv(self._edge_mlp(64,64), aggr='max'),
        ])
        self.regs = nn.ModuleList([
            nn.Sequential(
                nn.ReLU(),
                nn.Dropout(p=dropoutgnn),
            ),
            nn.Sequential(
                nn.ReLU(),
                nn.Dropout(p=dropoutgnn),
            ),
            nn.Identity()
        ])

        self.mlp = nn.Sequential(
            nn.Linear(64*3 + 5, 128), # 64 local + 64 global max + 64 global mean + 5 meta features (quaternions + normalized scale factor
            nn.ReLU(),
            nn.Dropout(p=dropoutfc),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=dropoutfc),
            nn.Linear(64, n_out)
        )
    @staticmethod
    def _edge_mlp(in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(2*in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
    def forward(self, data: GeoData):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Per-graph index of the node with highest gaze weight (last input feature)
        gaze = data.x[:, -1] # gaze weight of input data
        B = int(batch.max().item()) + 1
        max_gaze = gaze.new_full((B,), float('-inf'))
        max_gaze.scatter_reduce_(0, batch, gaze, reduce='amax', include_self=True)
        is_max = (gaze == max_gaze[batch])
        node_idx = torch.arange(gaze.shape[0], device=gaze.device, dtype=torch.long)
        masked = torch.where(is_max, node_idx, torch.full_like(node_idx, gaze.shape[0]))
        gazed_indices = masked.new_full((B,), gaze.shape[0])
        gazed_indices.scatter_reduce_(0, batch, masked, reduce='amin', include_self=True)

        for gnn, reg in zip(self.gnns, self.regs):
            x = gnn(x, edge_index)
            x = reg(x)

        x_local:Tensor  = x[gazed_indices]                    # (B, 64)
        x_global_max    = graph_global_max_pool(x, batch)     # (B, 64)
        x_global_mean   = graph_global_mean_pool(F.relu(x), batch)
        x_cat           = cat([x_local, x_global_max, x_global_mean, data.meta], dim=1)  # (B, 197)
        return self.mlp(x_cat)
class MANOGraspGNNv4(nn.Module):
    """
    Update:
      - Reverts to GCN rather than EdgeConv
    """
    def __init__(self, dropoutgnn=0.0, dropoutfc=0.0, n_in = 7, n_out = 3):
        super().__init__()
        if not (0 <= dropoutgnn < 1):
            raise ValueError("[GraspGNN]:\tdropoutgnn must be between 0 and 1")
        if not (0 <= dropoutfc < 1):
            raise ValueError("[GraspGNN]:\tdropoutfc must be between 0 and 1")
        self.dropoutgnn = dropoutgnn
        self.dropoutfc = dropoutfc
        self.gnns = nn.ModuleList([
            GCNConv(n_in, 64),
            GCNConv(64, 64),
            GCNConv(64, 64),
        ])
        self.regs = nn.ModuleList([
            nn.Sequential(
                nn.ReLU(),
                nn.Dropout(p=dropoutgnn),
            ),
            nn.Sequential(
                nn.ReLU(),
                nn.Dropout(p=dropoutgnn),
            ),
            nn.Identity(),
        ])

        self.mlp = nn.Sequential(
            nn.Linear(64*3 + 5, 128), # 64 local + 64 global max + 64 global mean + 5 meta features (quaternions + normalized scale factor
            nn.ReLU(),
            nn.Dropout(p=dropoutfc),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=dropoutfc),
            nn.Linear(64, n_out)
        )
    def forward(self, data: GeoData):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Per-graph index of the node with highest gaze weight (last input feature)
        gaze = data.x[:, -1] # gaze weight of input data
        B = int(batch.max().item()) + 1
        max_gaze = gaze.new_full((B,), float('-inf'))
        max_gaze.scatter_reduce_(0, batch, gaze, reduce='amax', include_self=True)
        is_max = (gaze == max_gaze[batch])
        node_idx = torch.arange(gaze.shape[0], device=gaze.device, dtype=torch.long)
        masked = torch.where(is_max, node_idx, torch.full_like(node_idx, gaze.shape[0]))
        gazed_indices = masked.new_full((B,), gaze.shape[0])
        gazed_indices.scatter_reduce_(0, batch, masked, reduce='amin', include_self=True)

        for gnn, reg in zip(self.gnns, self.regs):
            x = gnn(x, edge_index)
            x = reg(x)

        x_local:Tensor  = x[gazed_indices]                    # (B, 64)
        x_global_max    = graph_global_max_pool(x, batch)     # (B, 64)
        x_global_mean   = graph_global_mean_pool(F.relu(x), batch)
        x_cat           = cat([x_local, x_global_max, x_global_mean, data.meta], dim=1)  # (B, 197)
        return self.mlp(x_cat)
class MANOGraspGNNv5(nn.Module):
    """
    Applies skip-connections
    """
    def __init__(self, dropoutgnn=0.0, dropoutfc=0.0, n_in = 7, n_out = 3):
        super().__init__()
        if not (0 <= dropoutgnn < 1):
            raise ValueError("[GraspGNN]:\tdropoutgnn must be between 0 and 1")
        if not (0 <= dropoutfc < 1):
            raise ValueError("[GraspGNN]:\tdropoutfc must be between 0 and 1")
        self.dropoutgnn = dropoutgnn
        self.dropoutfc = dropoutfc
        self.gnns = nn.ModuleList([
            GCNConv(n_in, 64),
            GCNConv(64 + n_in, 64), # skip-connection appends previous input (64+7=71)
            GCNConv(2*64 + n_in, 64), # skip-connection appends previous input (64+71 = 135)
        ])
        self.regs = nn.ModuleList([
            nn.Sequential(
                nn.ReLU(),
                nn.Dropout(p=dropoutgnn),
            ),
            nn.Sequential(
                nn.ReLU(),
                nn.Dropout(p=dropoutgnn),
            ),
            nn.Identity(),
        ])

        self.mlp = nn.Sequential(
            nn.Linear((64*3+n_in)*3 + 5, 128), # 64 local + 64 global max + 64 global mean + 5 meta features (quaternions + normalized scale factor
            nn.ReLU(),
            nn.Dropout(p=dropoutfc),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=dropoutfc),
            nn.Linear(64, n_out)
        )
    def forward(self, data: GeoData):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # Per-graph index of the node with highest gaze weight (last input feature)
        gaze = data.x[:, -1] # gaze weight of input data
        B = int(batch.max().item()) + 1
        max_gaze = gaze.new_full((B,), float('-inf'))
        max_gaze.scatter_reduce_(0, batch, gaze, reduce='amax', include_self=True)
        is_max = (gaze == max_gaze[batch])
        node_idx = torch.arange(gaze.shape[0], device=gaze.device, dtype=torch.long)
        masked = torch.where(is_max, node_idx, torch.full_like(node_idx, gaze.shape[0]))
        gazed_indices = masked.new_full((B,), gaze.shape[0])
        gazed_indices.scatter_reduce_(0, batch, masked, reduce='amin', include_self=True)

        for gnn, reg in zip(self.gnns, self.regs):
            x_new = gnn(x, edge_index)
            x_new = reg(x_new)
            x = torch.cat([x, x_new], dim = 1)

        x_local:Tensor  = x[gazed_indices]                    # (B, 199)
        x_global_max    = graph_global_max_pool(x, batch)     # (B, 199)
        x_global_mean   = graph_global_mean_pool(F.relu(x), batch) # (B, 199)
        x_cat           = cat([x_local, x_global_max, x_global_mean, data.meta], dim=1)  # (B, 197)
        return self.mlp(x_cat)
class MANOGraspGNNv6(nn.Module):
    """
    Removes skip-connections and instead uses multilayer readout
    """
    def __init__(self, dropoutgnn=0.0, dropoutfc=0.0, n_in = 7, n_out = 3):
        super().__init__()
        if not (0 <= dropoutgnn < 1):
            raise ValueError("[GraspGNN]:\tdropoutgnn must be between 0 and 1")
        if not (0 <= dropoutfc < 1):
            raise ValueError("[GraspGNN]:\tdropoutfc must be between 0 and 1")
        self.dropoutgnn = dropoutgnn
        self.dropoutfc = dropoutfc
        self.gnns = nn.ModuleList([
            GCNConv(n_in, 64),
            GCNConv(64, 64),
            GCNConv(64, 64),
        ])
        self.regs = nn.ModuleList([
            nn.Sequential(
                nn.ReLU(),
                nn.Dropout(p=dropoutgnn),
            ),
            nn.Sequential(
                nn.ReLU(),
                nn.Dropout(p=dropoutgnn),
            ),
            nn.Identity(),
        ])

        self.mlp = nn.Sequential(
            nn.Linear(581, 128), # 3 (64,) vectors per layer + (5,) meta vector = 3*3*64 + 5  = 581
            nn.ReLU(),
            nn.Dropout(p=dropoutfc),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=dropoutfc),
            nn.Linear(64, n_out)
        )
    def forward(self, data: GeoData):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Per-graph index of the node with highest gaze weight (last input feature)
        gaze = data.x[:, -1] # gaze weight of input data
        B = int(batch.max().item()) + 1
        max_gaze = gaze.new_full((B,), float('-inf'))
        max_gaze.scatter_reduce_(0, batch, gaze, reduce='amax', include_self=True)
        is_max = (gaze == max_gaze[batch])
        node_idx = torch.arange(gaze.shape[0], device=gaze.device, dtype=torch.long)
        masked = torch.where(is_max, node_idx, torch.full_like(node_idx, gaze.shape[0]))
        gazed_indices = masked.new_full((B,), gaze.shape[0])
        gazed_indices.scatter_reduce_(0, batch, masked, reduce='amin', include_self=True)

        locals = []
        global_max = []
        global_mean = []
        for gnn, reg in zip(self.gnns, self.regs):
            x = gnn(x, edge_index)
            x = reg(x)
            locals.append(x[gazed_indices])
            global_max.append(graph_global_max_pool(x, batch))
            global_mean.append(graph_global_mean_pool(x, batch))
        x_local         = torch.cat(locals,     dim = 1) # (192,)
        x_global_max    = torch.cat(global_max, dim = 1) # (192,)
        x_global_mean   = torch.cat(global_mean,dim = 1) # (192,)
        x_cat           = cat([x_local, x_global_max, x_global_mean, data.meta], dim=1)  # (B, 3*192 + 5 = 581)
        return self.mlp(x_cat)
class MANOGraspGNNv7(nn.Module):
    """
    From v5 -> more layers
    """
    def __init__(self, dropoutgnn=0.0, dropoutfc=0.0, n_in = 7, n_out = 3):
        super().__init__()
        if not (0 <= dropoutgnn < 1):
            raise ValueError("[GraspGNN]:\tdropoutgnn must be between 0 and 1")
        if not (0 <= dropoutfc < 1):
            raise ValueError("[GraspGNN]:\tdropoutfc must be between 0 and 1")
        self.dropoutgnn = dropoutgnn
        self.dropoutfc = dropoutfc
        self.gnns = nn.ModuleList([
            GCNConv(n_in, 64),
            GCNConv(64 + n_in, 64), # skip-connection appends previous input  (64 + 7   = 71)
            GCNConv(64*2 + n_in, 64), # skip-connection appends previous input (64 + 71  = 135)
            GCNConv(64*3 + n_in, 64), # skip-connection appends previous input (64 + 135 = 199)
            GCNConv(64*4 + n_in, 64), # skip-connection appends previous input (64 + 199 = 263)
        ])
        self.regs = nn.ModuleList([
            nn.Sequential(
                nn.ReLU(),
                nn.Dropout(p=dropoutgnn),
            ),
            nn.Sequential(
                nn.ReLU(),
                nn.Dropout(p=dropoutgnn),
            ),
            nn.Sequential(
                nn.ReLU(),
                nn.Dropout(p=dropoutgnn),
            ),
            nn.Sequential(
                nn.ReLU(),
                nn.Dropout(p=dropoutgnn),
            ),
            nn.Identity(),
        ])

        self.mlp = nn.Sequential(
            nn.Linear((64*5+n_in)*3 + 5, 256), # 64 local + 64 global max + 64 global mean + 5 meta features (quaternions + normalized scale factor
            nn.ReLU(),
            nn.Dropout(p=dropoutfc),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=dropoutfc),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=dropoutfc),
            nn.Linear(64, n_out)
        )
    def forward(self, data: GeoData):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # Per-graph index of the node with highest gaze weight (last input feature)
        gaze = data.x[:, -1] # gaze weight of input data
        B = int(batch.max().item()) + 1
        max_gaze = gaze.new_full((B,), float('-inf'))
        max_gaze.scatter_reduce_(0, batch, gaze, reduce='amax', include_self=True)
        is_max = (gaze == max_gaze[batch])
        node_idx = torch.arange(gaze.shape[0], device=gaze.device, dtype=torch.long)
        masked = torch.where(is_max, node_idx, torch.full_like(node_idx, gaze.shape[0]))
        gazed_indices = masked.new_full((B,), gaze.shape[0])
        gazed_indices.scatter_reduce_(0, batch, masked, reduce='amin', include_self=True)

        for gnn, reg in zip(self.gnns, self.regs):
            x_new = gnn(x, edge_index)
            x_new = reg(x_new)
            x = torch.cat([x, x_new], dim = 1)

        x_local:Tensor  = x[gazed_indices]                    # (B, 3*64 + n_in)
        x_global_max    = graph_global_max_pool(x, batch)     # (B, 3*64 + n_in)
        x_global_mean   = graph_global_mean_pool(F.relu(x), batch) # (B, 3*64 + n_in)
        x_cat           = cat([x_local, x_global_max, x_global_mean, data.meta], dim=1)  # (B, 3*(3*64 + n_in)
        return self.mlp(x_cat)
class MANOGraspGNNv8(nn.Module):
    """
    Update:
      - Reverts to GCN rather than EdgeConv
    """
    def __init__(self, dropoutgnn=0.0, dropoutfc=0.0, n_in = 7, n_out = 3):
        super().__init__()
        if not (0 <= dropoutgnn < 1):
            raise ValueError("[GraspGNN]:\tdropoutgnn must be between 0 and 1")
        if not (0 <= dropoutfc < 1):
            raise ValueError("[GraspGNN]:\tdropoutfc must be between 0 and 1")
        self.dropoutgnn = dropoutgnn
        self.dropoutfc = dropoutfc
        self.gnns = nn.ModuleList([
            GCNConv(n_in, 64),
            GCNConv(64, 64),
            GCNConv(64, 64),
            GCNConv(64, 64),
            GCNConv(64, 64),
            GCNConv(64, 64),
            GCNConv(64, 64),
        ])
        self.regs = nn.ModuleList([
            nn.Sequential(
                nn.ReLU(),
                nn.Dropout(p=dropoutgnn),
            ),
            nn.Sequential(
                nn.ReLU(),
                nn.Dropout(p=dropoutgnn),
            ),
            nn.Sequential(
                nn.ReLU(),
                nn.Dropout(p=dropoutgnn),
            ),
            nn.Sequential(
                nn.ReLU(),
                nn.Dropout(p=dropoutgnn),
            ),
            nn.Sequential(
                nn.ReLU(),
                nn.Dropout(p=dropoutgnn),
            ),
            nn.Sequential(
                nn.ReLU(),
                nn.Dropout(p=dropoutgnn),
            ),
            nn.Identity(),
        ])

        self.mlp = nn.Sequential(
            nn.Linear(64*3 + 5, 128), # 64 local + 64 global max + 64 global mean + 5 meta features (quaternions + normalized scale factor
            nn.ReLU(),
            nn.Dropout(p=dropoutfc),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=dropoutfc),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(p=dropoutfc),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(p=dropoutfc),
            nn.Linear(64, n_out)
        )
    def forward(self, data: GeoData):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Per-graph index of the node with highest gaze weight (last input feature)
        gaze = data.x[:, -1] # gaze weight of input data
        B = int(batch.max().item()) + 1
        max_gaze = gaze.new_full((B,), float('-inf'))
        max_gaze.scatter_reduce_(0, batch, gaze, reduce='amax', include_self=True)
        is_max = (gaze == max_gaze[batch])
        node_idx = torch.arange(gaze.shape[0], device=gaze.device, dtype=torch.long)
        masked = torch.where(is_max, node_idx, torch.full_like(node_idx, gaze.shape[0]))
        gazed_indices = masked.new_full((B,), gaze.shape[0])
        gazed_indices.scatter_reduce_(0, batch, masked, reduce='amin', include_self=True)

        for gnn, reg in zip(self.gnns, self.regs):
            x = gnn(x, edge_index)
            x = reg(x)

        x_local:Tensor  = x[gazed_indices]                    # (B, 64)
        x_global_max    = graph_global_max_pool(x, batch)     # (B, 64)
        x_global_mean   = graph_global_mean_pool(F.relu(x), batch)
        x_cat           = cat([x_local, x_global_max, x_global_mean, data.meta], dim=1)  # (B, 197)
        return self.mlp(x_cat)
class MANOGraspGNNv9(nn.Module):
    """
    Update:
      - Reverts to GCN rather than EdgeConv
    """
    def __init__(self, dropoutgnn=0.0, dropoutfc=0.0, n_in = 7, n_out = 3):
        super().__init__()
        if not (0 <= dropoutgnn < 1):
            raise ValueError("[GraspGNN]:\tdropoutgnn must be between 0 and 1")
        if not (0 <= dropoutfc < 1):
            raise ValueError("[GraspGNN]:\tdropoutfc must be between 0 and 1")
        self.dropoutgnn = dropoutgnn
        self.dropoutfc = dropoutfc
        self.gnns = nn.ModuleList([
            GCNConv(n_in, 64),
            GCNConv(64, 64),
            GCNConv(64, 64),
            GCNConv(64, 64),
            GCNConv(64, 64),
            GCNConv(64, 64),
            GCNConv(64, 64),
            GCNConv(64, 64),
            GCNConv(64, 64),
        ])
        self.regs = nn.ModuleList([
            nn.Sequential(
                nn.ReLU(),
                nn.Dropout(p=dropoutgnn),
            ),
            nn.Sequential(
                nn.ReLU(),
                nn.Dropout(p=dropoutgnn),
            ),
            nn.Sequential(
                nn.ReLU(),
                nn.Dropout(p=dropoutgnn),
            ),
            nn.Sequential(
                nn.ReLU(),
                nn.Dropout(p=dropoutgnn),
            ),
            nn.Sequential(
                nn.ReLU(),
                nn.Dropout(p=dropoutgnn),
            ),
            nn.Sequential(
                nn.ReLU(),
                nn.Dropout(p=dropoutgnn),
            ),
            nn.Sequential(
                nn.ReLU(),
                nn.Dropout(p=dropoutgnn),
            ),
            nn.Sequential(
                nn.ReLU(),
                nn.Dropout(p=dropoutgnn),
            ),
            nn.Identity(),
        ])

        self.mlp = nn.Sequential(
            nn.Linear(64*3 + 5, 128), # 64 local + 64 global max + 64 global mean + 5 meta features (quaternions + normalized scale factor
            nn.ReLU(),
            nn.Dropout(p=dropoutfc),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=dropoutfc),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(p=dropoutfc),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(p=dropoutfc),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(p=dropoutfc),
            nn.Linear(64, n_out)
        )
    def forward(self, data: GeoData):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Per-graph index of the node with highest gaze weight (last input feature)
        gaze = data.x[:, -1] # gaze weight of input data
        B = int(batch.max().item()) + 1
        max_gaze = gaze.new_full((B,), float('-inf'))
        max_gaze.scatter_reduce_(0, batch, gaze, reduce='amax', include_self=True)
        is_max = (gaze == max_gaze[batch])
        node_idx = torch.arange(gaze.shape[0], device=gaze.device, dtype=torch.long)
        masked = torch.where(is_max, node_idx, torch.full_like(node_idx, gaze.shape[0]))
        gazed_indices = masked.new_full((B,), gaze.shape[0])
        gazed_indices.scatter_reduce_(0, batch, masked, reduce='amin', include_self=True)

        for gnn, reg in zip(self.gnns, self.regs):
            x = gnn(x, edge_index)
            x = reg(x)

        x_local:Tensor  = x[gazed_indices]                    # (B, 64)
        x_global_max    = graph_global_max_pool(x, batch)     # (B, 64)
        x_global_mean   = graph_global_mean_pool(F.relu(x), batch)
        x_cat           = cat([x_local, x_global_max, x_global_mean, data.meta], dim=1)  # (B, 197)
        return self.mlp(x_cat)


if __name__ == '__main__':
    import torch

    try:
        from thop import profile, clever_format
        has_thop = True
    except ImportError:
        has_thop = False
        print("Note: install 'thop' for FLOP counts  (pip install thop)\n")

    def make_dummy_image(n_in=4, size=128):
        """Dummy input for the CNN models: a single (1, C, H, W) image."""
        return torch.randn(1, n_in, size, size)

    N_IN = 8  # node feature width fed to the GNN models

    def make_dummy_graph(n_nodes=128, n_in=N_IN, n_meta=5, avg_degree=6):
        """Dummy input for the GNN models: a single-graph PyG ``Data`` batch.

        Mirrors what the training ``DataLoader`` produces: node features ``x``
        of shape (N, n_in) where the last column is the gaze weight, an
        ``edge_index`` of shape (2, E), a ``batch`` vector (all zeros for one
        graph) and the per-graph ``meta`` features of shape (B, n_meta).
        """
        x = torch.randn(n_nodes, n_in)
        n_edges = n_nodes * avg_degree
        edge_index = torch.randint(0, n_nodes, (2, n_edges), dtype=torch.long)
        batch = torch.zeros(n_nodes, dtype=torch.long)
        data = GeoData(x=x, edge_index=edge_index, batch=batch)
        data.meta = torch.randn(1, n_meta)
        return data

    # (name, class, input-factory) — CNNs eat an image, GNNs eat a graph.
    cnn_models = [('GraspCNNv1', GraspCNNv1), ('GraspCNNv2', GraspCNNv2), ('GraspCNNv3', GraspCNNv3),
                  ('MANOGraspCNNv1', MANOGraspCNNv1), ('MANOGraspCNNv2', MANOGraspCNNv2), ('MANOGraspCNNv3', MANOGraspCNNv3)]
    gnn_models = [("MANOGraspGNNv1", MANOGraspGNNv1), ("MANOGraspGNNv2", MANOGraspGNNv2), ("MANOGraspGNNv3", MANOGraspGNNv3),
                  ("MANOGraspGNNv4", MANOGraspGNNv4), ("MANOGraspGNNv5", MANOGraspGNNv5), ("MANOGraspGNNv6", MANOGraspGNNv6),
                  ("MANOGraspGNNv7", MANOGraspGNNv7), ("MANOGraspGNNv8", MANOGraspGNNv8), ("MANOGraspGNNv9", MANOGraspGNNv9)]

    # CNNs take no n_in (fixed 4-channel image); GNNs are built with n_in=N_IN.
    all_models = ([(name, cls, {}, make_dummy_image) for name, cls in cnn_models] +
                  [(name, cls, {'n_in': N_IN}, make_dummy_graph) for name, cls in gnn_models])

    for name, cls, kwargs, make_input in all_models:
        model = cls(**kwargs)
        model.eval()
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{name}")
        print(f"  Trainable parameters : {n_params:,}")
        if has_thop:
            dummy = make_input()
            try:
                macs, _ = profile(model, inputs=(dummy,), verbose=False)
                macs_fmt, _ = clever_format([macs, n_params], '%.3f')
                print(f"  MACs (≈ FLOPs/2)     : {macs_fmt}")
            except Exception as e:
                # thop has no hooks for the PyG message-passing layers, so it
                # may raise rather than count them; report instead of crashing.
                print(f"  MACs (≈ FLOPs/2)     : n/a ({type(e).__name__}: {e})")
        print()