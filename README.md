# TerraSpectra: Hyperspectral Crop Disease Forecasting

Early crop-disease detection from hyperspectral satellite cubes (200+ bands) using a
**3D-CNN + Vision Transformer** hybrid, served via **FastAPI** and visualised on a
**React + Deck.gl** GIS dashboard.

## Architecture
```
Hyperspectral cube (GeoTIFF/HDF5)
   -> Rasterio loader -> normalise -> PCA (200+ -> 30 bands)
   -> 3D-CNN (spatial-spectral features) -> ViT (spectral self-attention)
   -> healthy / stressed pixel map
   -> FastAPI (tiling + stitching) -> React/Deck.gl heatmap + analytics
```

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python data/generate_mock_cube.py          # mock cube + labels
python -m terraspectra.train               # train hybrid model
uvicorn terraspectra.api.main:app --reload # API on :8000
cd frontend && npm install && npm run dev  # dashboard on :5173
```
(Set `PYTHONPATH=src` or `pip install -e .` equivalent when running modules.)

## Layout
- `src/terraspectra/data` - IO, preprocessing, datasets
- `src/terraspectra/models` - 3D-CNN, ViT, hybrid
- `src/terraspectra/api` - FastAPI + tiling
- `frontend/` - React + Deck.gl dashboard
- `docs/` - report, memory audit, presentation outline

## Status
See `docs/PROGRESS.md` for the 20-day log.
