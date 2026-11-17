"""Generate a mock hyperspectral cube (bands, H, W) + stress labels as GeoTIFF/npy."""
import numpy as np
import rasterio
from rasterio.transform import from_origin

BANDS, H, W = 200, 256, 256
rng = np.random.default_rng(42)

def make_cube():
    wl = np.linspace(400, 2500, BANDS)
    healthy = 0.1 + 0.5 / (1 + np.exp(-(wl - 720) / 20))      # red-edge shape
    stressed = healthy.copy()
    stressed[(wl > 680) & (wl < 760)] *= 0.8                  # subtle chlorophyll shift
    labels = np.zeros((H, W), np.uint8)
    yy, xx = np.ogrid[:H, :W]
    for cy, cx, r in [(70, 80, 25), (180, 170, 30)]:
        labels[(yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2] = 1
    cube = np.where(labels[None] == 1, stressed[:, None, None], healthy[:, None, None])
    cube = cube + rng.normal(0, 0.01, cube.shape)
    return cube.astype("float32"), labels

if __name__ == "__main__":
    cube, labels = make_cube()
    transform = from_origin(75.80, 26.95, 0.0001, 0.0001)  # near Jaipur, ~10m px
    with rasterio.open("data/mock_cube.tif", "w", driver="GTiff", height=H, width=W,
                       count=BANDS, dtype="float32", crs="EPSG:4326", transform=transform) as dst:
        dst.write(cube)
    np.save("data/mock_labels.npy", labels)
    print("Saved cube", cube.shape, "stressed px:", int(labels.sum()))
