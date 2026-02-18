# Snowflake Chione 2026 — Source Code

This directory contains the simulation engine and analysis tools for the Snowflake Chione 2026 project — a computational reconstruction of the Nakaya Diagram using the Reiter cellular automaton model.

## Quick Start

### 1. Install Dependencies

```bash
# From the repository root:
make setup

# Or manually:
pip install -r requirements.txt

# FFmpeg is required for video encoding:
# Ubuntu/Debian: sudo apt install ffmpeg
# macOS: brew install ffmpeg
# Windows: download from https://ffmpeg.org
```

### 2. Run a Quick Test

```bash
make test
```

This runs 4 simulations with a small grid to verify everything works. Results appear in `results/`.

### 3. Run the Full 12×12 Parameter Sweep

```bash
# Serial (safe, low memory):
make scan

# Parallel (faster, uses multiple CPU cores):
make scan-parallel
```

This generates **144 simulations** across 12 α × 12 γ values, producing:
- `.mp4` videos (H.265 lossless — mass view + rainbow time view)
- `.png` snapshots (final crystal images)
- `scan_results.csv` (metrics: area, perimeter, compactness, branching factor, class, etc.)

### 4. Run MAP-Elites Shape Exploration

```bash
make research GIVEN_DIR=results/Run_v39_... BUDGET=50
```

Uses evolutionary quality-diversity search to discover novel crystal morphologies beyond the regular grid.

### 5. Generate 3D Interactive Visualization

```bash
make viz-3d DIR=results/Run_v39_...
```

Opens an interactive 3D HTML surface plot in your browser, where freeze time is mapped to height.

### 6. Docker (No Local Setup)

```bash
make docker-build
make docker-run
```

Results are mounted to `./results/` on the host.

---

## Source Files

| File | Size | Purpose |
|:--|:--:|:--|
| `engine.py` | 24K | Core Reiter CA simulation engine. Runs the hexagonal cellular automaton with diffusion (α), freezing (β), and vapour supply (γ). Streams frames to `VideoWriter` for MP4 generation. |
| `cli.py` | 48K | Command-line interface — orchestrates parameter sweeps, collects metrics into `scan_results.csv`, generates HTML dashboards. |
| `video_writer.py` | 20K | H.265 lossless video encoder. Streams frames via FFmpeg subprocess pipe. Supports 4K upscaling and rainbow time-view rendering. |
| `viz.py` | 8K | Visualization module — renders final PNG snapshots (mass view + rainbow time view) using hexagonal grid plotting. |
| `utils.py` | 12K | Utility functions — metric calculation (compactness, branching factor, growth rate), custom colormaps, memory monitoring. |
| `plotting.py` | 8K | Phase diagram and Nakaya diagram generation from scan results CSV. |
| `research.py` | 24K | MAP-Elites quality-diversity search — evolutionary exploration of the (α, γ, β) parameter space to discover novel crystal morphologies. |
| `viz_3d_builder.py` | 20K | Generates interactive 3D HTML visualization using Plotly.js. Freeze time maps to surface height, revealing growth chronology. |
| `main.py` | 0.5K | Entry point — imports and calls `cli.main()`. |
| `__init__.py` | 0 | Python package marker. |

---

## Parameters

The simulation is controlled by three parameters:

| Parameter | Symbol | Range (12×12 grid) | Physical Meaning |
|:--|:--:|:--:|:--|
| Diffusivity | α | 0.01 – 2.5 | Rate of vapour diffusion → maps to temperature |
| Vapour supply | γ | 0.0001 – 0.01 | Background supersaturation → maps to supersaturation |
| Initial vapour | β | 0.4 (fixed) | Initial vapour mass per lattice site |

---

## Output Structure

After running a scan, results appear in a timestamped directory:

```
results/Run_v39_res12x12_ALL_YYYY-MM-DD/
├── Intermediate/                    # PNG snapshots
│   ├── snowflake_Alpha0.01_Gamma0.0001_Beta0.4.png      # Mass view
│   ├── snowflake_Alpha0.01_Gamma0.0001_Beta0.4_Time.png  # Rainbow time view
│   └── ...                          # (144 × 2 = 288 images)
├── Videos/                          # MP4 animations
│   ├── snowflake_Alpha0.01_Gamma0.0001.mp4
│   └── ...                          # (144 videos)
├── scan_results.csv                 # Aggregated metrics (144 rows)
├── phase_diagram.png                # Generated phase diagram
└── phase_diagram_rainbow.png        # Rainbow version
```

### CSV Columns

| Column | Description |
|:--|:--|
| `Alpha`, `Gamma` | Simulation parameters |
| `Area` | Total frozen cells |
| `Perimeter` | Boundary cell count |
| `Compactness` | Area / (Perimeter²) ratio |
| `Radius` | Maximum distance from centre |
| `Steps` | Simulation steps to completion |
| `GrowthRate` | Cells frozen per time step |
| `Class` | Morphological classification (Dendrite, Hybrid, Faceted, Sparse, No Growth) |
| `Temperature` | Mapped physical temperature (°C) |
| `Supersaturation` | Mapped physical supersaturation (g/m³) |

---

## Examples

### Single simulation

```bash
python3 -m src.cli --fast_test --force_cpu
```

### Custom parameter sweep

```bash
python3 -m src.cli \
  --alpha_min 0.3 --alpha_max 1.5 \
  --gamma_min 0.001 --gamma_max 0.005 \
  --gif
```

### Generate 3D visualization from existing results

```bash
python3 src/viz_3d_builder.py --results_dir results/Run_v39_...
# → Opens viz_3d.html in browser
```
