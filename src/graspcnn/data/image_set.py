import torch
from torch.utils.data import Dataset
from torch import Tensor, nn
import os
import matplotlib.pyplot as plt
from torchvision import transforms as T
import numpy as np

from abc import ABC
class NormalizeBaseClass(torch.nn.Module, ABC):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = Tensor(mean).view(3,1,1)
        self.std = Tensor(std).view(3,1,1)
    def forward(self, x:Tensor) -> Tensor:
        pass
class UnNormalize(NormalizeBaseClass):
    def forward(self, x:Tensor) -> Tensor:
        return x * self.std + self.mean
class ReNormalize(NormalizeBaseClass):
    def forward(self, x:Tensor) -> Tensor:
        return (x - self.mean) / self.std
class GraspImageDatasetBase(Dataset, ABC):
    def __init__(
            self,
            images: Tensor,
            labels: Tensor,
            mean: Tensor,
            std: Tensor,
            augment: bool = False,
            **kwargs,
    ):
        self.images                 = images.contiguous()  # (N, 4, H, W)
        self.labels                 = labels  # (N,)
        self.mean                   = mean  # (4,)
        self.std                    = std  # (4,)
        self.class_names            = ["lateral", "pinch", "power"]
        self._augment               = augment
        self._flip_kwargs           = {}
        self._color_jitter_kwargs   = {}
        self._device = None
        self._init_flip(kwargs.get("p_flip", 0.5))
        self._init_color_jitter(
            p           = kwargs.get("p_jitter", 0.5),
            brightness  = kwargs.get("brightness_jitter", 0.2),
            contrast    = kwargs.get("contrast_jitter", 0.2),
        )

    def _init_flip(self, p: float) -> nn.Module:
        self._flip_kwargs = {'p': p}
        self._flip = T.RandomHorizontalFlip(p=p)
        return self._flip
    def _init_color_jitter(self, p: float, brightness: float, contrast: float):
        self._color_jitter_kwargs = {'p':p, 'brightness':brightness, 'contrast':contrast}
        self._color_jitter = T.RandomApply([
            UnNormalize(self.mean[:3], self.std[:3]),
            T.ColorJitter(brightness=brightness, contrast=contrast),
            ReNormalize(self.mean[:3], self.std[:3])
        ], p = p)
        return self._color_jitter
    def half(self):
        self.images = self.images.half()

    @property
    def augment(self) -> bool:
        if hasattr(self, '_augment'):
            return self._augment
        else:
            return False

    @augment.setter
    def augment(self, val: bool) -> None:
        self._augment = val
        if val:
            if not hasattr(self, '_flip'): self._init_flip(0.5)
            if not hasattr(self, '_color_jitter'): self._init_color_jitter(0.5,0.2,0.2)
    @property
    def device(self):
        if hasattr(self, '_device') and self._device: return self._device
        else: return "cpu"
    def to(self, device: str) -> None:
        if device not in ('cpu', 'cuda', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'):
            raise ValueError(f'device {device} is not supported')
        self._device = device
        for prop in ('images', 'labels', 'mean', 'std'):
            if hasattr(self, prop):
                setattr(self, prop, getattr(self, prop).to(device))
        # reinitialize transforms so UnNormalize/ReNormalize use updated mean/std
        if not self._flip_kwargs: self._flip_kwargs = {"p": 0.5}
        if not self._color_jitter_kwargs: self._color_jitter_kwargs = {"p": 0.5, "brightness": 0.2, "contrast": 0.2}
        self._init_flip(**self._flip_kwargs)
        self._init_color_jitter(**self._color_jitter_kwargs)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        if self.augment:
            return self._augment_image(self.images[idx]), self.labels[idx]
        else:
            return self.images[idx], self.labels[idx]


    def _augment_image(self, image: Tensor) -> Tensor:

        image = self._flip(image)
        rgb, gaze = image[:3], image[3:]
        # in order to jitter we must unnormalize the set (in range [0,1]

        rgb = self._color_jitter(rgb)
        return torch.cat([rgb, gaze], dim=0)
class GraspImageDataset(GraspImageDatasetBase):
    """
    Precached anticipatory-gaze image dataset.

    Each item
    ---------
    image : float32 Tensor (IMAGE_SIZE, IMAGE_SIZE, 4)
            Normalised channels: R, G, B, Gaze
    label : int64   Tensor ()
            Index into class_names

    Attributes
    ----------
    class_names   : list[str]  — index → class label
    channel_names : list[str]  — channel index → name
    mean, std     : Tensor(4,) — per-channel normalisation parameters
    """

    channel_names = ["R", "G", "B", "Gaze"]

MANO_COLS   = [f"mano_{i}" for i in range(15)]
class GraspImagePosesDataset(Dataset):
    """
    RGB+Gaze image → MANO pose dataset.

    Each item: (image, pose)
      image : float32 tensor (4, H, W) — normalised R, G, B, Gaze channels
      pose  : float32 tensor (15,)     — normalised MANO joint angles

    Attributes
    ----------
    pose_names          : list[str] — MANO column names
    img_mean / img_std  : Tensor (4,) — image normalisation stats
    pose_mean / pose_std: Tensor (15,) — pose normalisation stats
    """

    def __init__(
        self,
        images:    Tensor,   # (N, 4, H, W) float32, already normalised
        poses:     Tensor,   # (N, 15)      float32, already normalised
        img_mean:  Tensor,
        img_std:   Tensor,
        pose_mean: Tensor,
        pose_std:  Tensor,
    ):
        self.images    = images
        self.poses     = poses
        self.img_mean  = img_mean
        self.img_std   = img_std
        self.pose_mean = pose_mean
        self.pose_std  = pose_std
        self.pose_names = MANO_COLS

    def __len__(self) -> int:
        return len(self.poses)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.images[idx], self.poses[idx]
class GraspImagePosesDatasetV2(GraspImageDataset):
    @classmethod
    def from_v1(cls, old: GraspImagePosesDataset, **kwargs):
        return cls(
            images  = old.images,
            labels  = old.poses,
            mean    = old.img_mean,
            std     = old.img_std,
            **kwargs

        )
if __name__ == "__main__":
    ds: GraspImageDataset = torch.load(os.path.expanduser("~/Documents/P10/hot3d/data/test_images.pt"), weights_only=False)
    ds.augment = True
    ds.to('cuda')
    print(ds.device)
    print(ds.labels.device)
    print(ds.images.device)
    for _ in range(10):
        idx = np.random.randint(len(ds))

        img, label = ds[idx]
        img*=ds.std.view(4,1,1)
        img+=ds.mean.view(4,1,1)
        print(img.min())
        print(img.max())
        print()
        img, gaze = img[:3].permute(1,2, 0).contiguous(), img[3]

        fig, ax = plt.subplots(1,2,figsize=(10,5))
        ax[0].imshow(img)
        ax[1].imshow(gaze)
        plt.show()