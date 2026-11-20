import numpy as np
import torch
from torch.utils.data import Dataset

class PatchDataset(Dataset):
    """Yields (1, bands, p, p) patches centred on a pixel, with that pixel's label."""
    def __init__(self, cube, labels, patch=9, n_samples=4000, seed=0):
        self.labels, self.p = labels, patch
        pad = patch // 2
        self.padded = np.pad(cube, ((0, 0), (pad, pad), (pad, pad)), mode="reflect")
        rng = np.random.default_rng(seed)
        pos = np.argwhere(labels == 1); neg = np.argwhere(labels == 0)
        k = n_samples // 2
        sel = np.vstack([pos[rng.choice(len(pos), k)], neg[rng.choice(len(neg), k)]])
        rng.shuffle(sel)
        self.idx = sel

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        y, x = self.idx[i]
        patch = self.padded[:, y:y + self.p, x:x + self.p]
        return torch.from_numpy(patch[None].copy()), int(self.labels[y, x])
