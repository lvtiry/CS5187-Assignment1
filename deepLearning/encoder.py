# ResNet encoder
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import Dataset

from config import DEEP, DEEP_WEIGHTS, GALLERY_DIR

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MULTIPLE = 32  # ResNet feature map = input / 32

# Building the ResNet Backbone
def build_trunk(backbone: str = DEEP.backbone, weights: str | None = DEEP.weights):
    import torchvision

    torch.hub.set_dir(str(DEEP_WEIGHTS))
    if not backbone.startswith("resnet"):
        raise ValueError(f"unsupported backbone {backbone!r}; expected a resnet")

    model = getattr(torchvision.models, backbone)(weights=weights)
    out_dim = model.fc.in_features
    trunk = nn.Sequential(*list(model.children())[:-2]).eval()
    for param in trunk.parameters():
        param.requires_grad_(False)
    return trunk, out_dim

# Generalised mean pooling over the spatial dimensions
def gem_pool(fmap: torch.Tensor, p: float = DEEP.gem_p, eps: float = 1e-6) -> torch.Tensor:
    return F.avg_pool2d(
        fmap.clamp(min=eps).pow(p), (fmap.shape[-2], fmap.shape[-1])
    ).pow(1.0 / p)

# Wraps the convolutional trunk
class Encoder:
    # Load the backbone network
    def __init__(self, backbone: str = DEEP.backbone, weights: str | None = DEEP.weights,
                 device: str = "cuda"):
        self.trunk, self.dim = build_trunk(backbone, weights)
        self.device = device
        self.trunk.to(device)

    # GGet feature map
    @torch.inference_mode()
    def feature_maps(self, batch: torch.Tensor) -> torch.Tensor:
        return self.trunk(batch.to(self.device, non_blocking=True))

    # GeM descriptors
    @torch.inference_mode()
    def global_descriptors(self, batch: torch.Tensor) -> torch.Tensor:
        return gem_pool(self.feature_maps(batch)).flatten(1)

    # Descriptor of a single map
    @torch.inference_mode()
    def descriptor(self, fmap: torch.Tensor) -> torch.Tensor:
        return gem_pool(fmap.unsqueeze(0)).flatten(1)[0]

    # Encodes a single PIL image
    @torch.inference_mode()
    def encode_pil(self, img: Image.Image, short_side: int, long_side: int | None = None):
        prepared = prepare(img, short_side, long_side)
        fmap = self.feature_maps(prepared.tensor.unsqueeze(0))[0]
        return fmap, prepared.scale_x, prepared.scale_y, prepared.size

# Preprocessing Results
@dataclass
class Prepared:
    tensor: torch.Tensor
    scale_x: float
    scale_y: float
    size: tuple[int, int]

# Calculates the input dimensions for a grid
def target_size(width: int, height: int, short_side: int, long_side: int | None = None):
    scale = short_side / float(min(width, height))
    if long_side is not None:
        scale = min(scale, long_side / float(max(width, height)))
    nw = max(MULTIPLE, int(round(width * scale / MULTIPLE)) * MULTIPLE)
    nh = max(MULTIPLE, int(round(height * scale / MULTIPLE)) * MULTIPLE)
    return nw, nh

# Resize and apply ImageNet mean-variance normalization, then output a CHW tensor
def to_tensor(img: Image.Image, size: tuple[int, int]) -> torch.Tensor:
    resized = img.resize(size, Image.BICUBIC)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32)[:, None, None]
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32)[:, None, None]
    return (tensor - mean) / std

# Preprocess PIL images into tensors and record the scale
def prepare(img: Image.Image, short_side: int, long_side: int | None = None) -> Prepared:
    width, height = img.size
    nw, nh = target_size(width, height, short_side, long_side)
    return Prepared(to_tensor(img, (nw, nh)), nw / width, nh / height, (nw, nh))

# Image dataset
class GalleryDataset(Dataset):
    # Constructs a list of record numbers, target input dimensions, and the root directory of the images
    def __init__(self, ids, size, root: str | Path | None = None):
        self.ids = [int(i) for i in ids]
        self.size = (int(size[0]), int(size[1]))
        self.root = Path(root) if root is not None else GALLERY_DIR

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        gid = self.ids[index]
        with Image.open(self.root / f"{gid}.jpg") as img:
            img.draft("RGB", self.size)
            tensor = to_tensor(img.convert("RGB"), self.size)
        return gid, tensor

# PCA Whitening
# fitting, transformation, and retrieval; the transformed vectors are L2-normalized, so the dot product is the cosine.
class Whitening:
    # Save the mean, principal components, and scaling factors used for albinism
    def __init__(self, mean: np.ndarray, components: np.ndarray, scale: np.ndarray):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.components = np.asarray(components, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)

    # Dimensions after whitening
    @property
    def dim(self) -> int:
        return int(self.components.shape[0])

    # Project the original descriptors into the albino space and normalize them using L2
    def transform(self, x: np.ndarray) -> np.ndarray:
        single = np.ndim(x) == 1
        x = np.atleast_2d(np.asarray(x, dtype=np.float32))
        z = (x - self.mean) @ self.components.T
        z /= self.scale
        z = l2_normalize(z)
        return z[0] if single else z

    # Fits PCA to the image gallery descriptors, whitens them, and returns the transformed objects
    @classmethod
    def fit(cls, features: np.ndarray, dim: int = DEEP.pca_dim,
            sample: int = DEEP.pca_sample, seed: int = DEEP.seed) -> "Whitening":
        from sklearn.decomposition import PCA

        features = np.asarray(features, dtype=np.float32)
        rng = np.random.default_rng(seed)
        if len(features) > sample:
            features = features[rng.choice(len(features), size=sample, replace=False)]
        # a smoke run may have fewer images than the requested dimension
        dim = int(min(dim, features.shape[0] - 1, features.shape[1]))
        pca = PCA(n_components=dim, whiten=True, svd_solver="randomized", random_state=seed)
        pca.fit(features)
        scale = np.sqrt(pca.explained_variance_).astype(np.float32) + 1e-8
        return cls(pca.mean_, pca.components_, scale)

    # Save the mean, principal components, and scaling factors as npz
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.mean, components=self.components, scale=self.scale)
        return path

    # Reads albinism parameters from npz
    @classmethod
    def load(cls, path: str | Path) -> "Whitening":
        data = np.load(path)
        return cls(data["mean"], data["components"], data["scale"])

# Perform L2 normalization along the specified axis
def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, eps)
