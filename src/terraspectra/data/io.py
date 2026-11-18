import numpy as np
import rasterio

def load_cube(path):
    """Return (cube[bands,H,W] float32, profile) from a GeoTIFF."""
    with rasterio.open(path) as src:
        return src.read().astype("float32"), src.profile

def load_cube_windowed(path, window):
    """Read a rasterio Window only -- avoids loading multi-GB cubes."""
    with rasterio.open(path) as src:
        return src.read(window=window).astype("float32")
