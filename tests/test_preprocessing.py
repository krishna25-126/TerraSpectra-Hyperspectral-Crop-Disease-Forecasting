import numpy as np
from terraspectra.data.preprocessing import normalize, fit_pca, apply_pca

def test_pca_shape():
    c = normalize(np.random.rand(50, 16, 16).astype("float32"))
    assert apply_pca(c, fit_pca(c, 10)).shape == (10, 16, 16)
