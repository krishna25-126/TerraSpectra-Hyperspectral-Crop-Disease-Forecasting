import numpy as np
from sklearn.decomposition import PCA

def normalize(cube, eps=1e-8):
    """Per-band standardisation. cube: (B,H,W)."""
    m = cube.mean(axis=(1, 2), keepdims=True)
    s = cube.std(axis=(1, 2), keepdims=True)
    return (cube - m) / (s + eps)

def fit_pca(cube, n_components=30):
    B, H, W = cube.shape
    return PCA(n_components=n_components).fit(cube.reshape(B, -1).T)

def apply_pca(cube, pca):
    B, H, W = cube.shape
    out = pca.transform(cube.reshape(B, -1).T)
    return out.T.reshape(-1, H, W).astype("float32")
