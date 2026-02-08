# Snowflake Growth Simulation (Reiter Model)
**Author: Lorenzo Granado**

> **Abstract**: This project implements a high-performance computational physics simulation of snowflake growth using the Reiter Cellular Automata model. By simulating the competing physical processes of vapor diffusion and deposition on a 2D hexagonal grid, we successfully reproduce the complex, branching morphologies observed in nature with striking beauty. Our results confirm that the transition from simple plate-like crystals to complex dendritic structures can be modelled purely through local interaction rules, aligning with theoretical predictions of phase transitions in crystal growth. The simulation generates high-resolution visualisations, including phase diagrams and 3D time-evolution structures, offering new insights into the non-equilibrium dynamics of ice formation.

---

## 🔬 Scientific Overview

Snowflake formation is a classic example of pattern formation in non-equilibrium thermodynamics. While the six-fold symmetry of ice is due to the molecular structure of the water crystal lattice, the complex branching patterns (dendrites) arise from a macroscopic instability known as the **Mullins-Sekerka instability**.

This project builds upon the foundational work of **Reiter (2005)**, who demonstrated that a simplified Cellular Automata (CA) model could capture the essential physics of this instability without solving the full partial differential equations of heat and mass transfer.

Key connections to literature:
*   **Nakaya Correspondence**: Our simulation sweep successfully recreates the "Nakaya Diagram", mapping our model parameters ($\alpha, \gamma$) to physical temperature and supersaturation.
*   **Morphological Selection**: We observe the distinct regimes predicted by crystal growth theory: Faceted growth (Plates) at low diffusion rates and Dendritic growth (Stars) at high diffusion rates.
*   **Phase Transitions**: The sudden onset of branching in our results supports the hypothesis of a first-order phase transition in morphological complexity.

---

## ⚙️ Physics Engine & Methodology

The core of the simulation is a **Real-Valued Cellular Automata** operating on a hexagonal grid. The state of each cell is defined by a continuous variable $s(x,y)$ representing the water mass/vapour density.

### 1. The Numerical Method
The system evolves in discrete time steps ($t$) according to three rules:
1.  **Diffusion**: Vapour diffuses to neighbouring cells (simulating Brownian motion).
    $$ s_{next}(x) = s(x) + \frac{\alpha}{2} (s_{avg} - s(x)) $$
2.  **Boundary Reception**: Cells adjacent to existing ice ($s \ge 1.0$) become "receptive".
3.  **Freezing (Addition)**: Receptive cells absorb vapour ($\gamma$) from the supersaturated background.
    $$ s_{next}(x) = s(x) + \gamma $$

### 2. Visualisation Features
*   **Standard Mass View**: Renders the ice crystal in "Ice Blue", with opacity determined by mass density.
*   **Rainbow Time-Evolution**: A novel visualisation where colour represents the **Time of Freezing**.
    *   **Blue/Green**: Old ice (core).
    *   **Red/Orange**: Newly frozen tips.
    *   *Scientific Value*: This reveals the growth history and tip velocity dynamics in a single static image.
*   **3D Space-Time Structures**: By stacking 2D frames over time, we generate 3D voxel structures that visualise the entire growth history as a solid object in $(x,y,t)$ space.

### Example Results

| Standard View (Mass) | Rainbow View (Time) |
| :---: | :---: |
| ![Blue Ice](assets/snowflake_Alpha2.50_Gamma0.0010_Beta0.4.png) | ![Rainbow](assets/snowflake_Alpha2.50_Gamma0.0010_Beta0.4_Time.png) |
| [Video Animation](assets/snowflake_Alpha1.50_Gamma0.0050.mp4) | [Rainbow Animation](assets/rainbow_Alpha1.50_Gamma0.0050.mp4) |

### 3. Phase Diagrams
We conduct massive parameter sweeps to map the ($\alpha, \gamma$) space, generating:
*   **Morphology Maps**: Grids of final snowflake shapes.
*   **Metric Heatmaps**: Visualising physical properties like *Compactness* ($P^2/A$) and *Tip Velocity* to quantitatively identify phase boundaries.

---

## 🏛 Architecture

The project is organized as follows:

```ascii
snowflake2026/
├── src/
│   ├── main.py           # Unified entry point (CLI & Orchestrator)
│   ├── engine.py         # Physics Kernels (CPU & GPU/Numba)
│   ├── video_writer.py   # H.265 Video Encoding Logic
│   ├── plotting.py       # Nakaya Diagram & Data Visualization
│   ├── utils.py          # Metric calculations & Helpers
│   └── viz.py            # Matplotlib Rendering utilities
├── docs/
│   ├── ExperimentDesign.md # Detailed Physics Theory
│   └── METRICS.md          # Definitions of CSV columns
├── results/              # Simulation Outputs
│   └── Run_.../          # Timestamped Run Directory
│       ├── Intermediate/ # Static PNG Snapshots
│       ├── Videos/       # MP4 Animations (Standard & Rainbow)
│       └── SpaceTime/    # 3D Visualization Stacks
├── Makefile              # Automation Commands
└── requirements.txt      # Python Dependencies
```

---

## 🚀 Quick Start

### Prerequisites
*   Python 3.8+
*   FFmpeg (for video generation)
*   NVIDIA GPU (Optional, for CUDA acceleration)

### Setup
```bash
# 1. Clone only the release files
git clone ...
cd snowflake2026

# 2. auto-setup (creates venv)
make setup
```

### Running Simulations

**1. Fast Test (Verify Setup)**
Runs a small 2x2 scan to ensure everything works.
```bash
make test
```

**2. Full Simulation (CPU)**
Runs a comprehensive scan using multi-core processing.
```bash
make run-parallel
```

**3. High-Performance Scan (GPU)**
If you have an NVIDIA GPU, this mode is significantly faster for physics calculations.
```bash
make run-gpu
```

**Viewing Results**:
Check the `results/` folder. Open `dashboard_video.html` (generated after a full run) to see a grid of all simulated snowflakes.

---

## 📘 Physics Theory & Parameters

The simulation is controlled by three dimensionless parameters. See `docs/ExperimentDesign.md` for deep details.

| Parameter | Symbol | Physical Meaning | Effect |
| :--- | :--- | :--- | :--- |
| **Alpha** | $\alpha$ | **Diffusivity** | Controls branching. Low $\alpha$ = Plates, High $\alpha$ = Dendrites. |
| **Gamma** | $\gamma$ | **Vapour Background** | Controls growth speed. Higher $\gamma$ = Faster growth. |
| **Beta** | $\beta$ | **Initial Condition** | Background vapour level (usually 0.4). |

### CSV Metrics (`metrics_log.csv`)
For every simulation, we log physical properties to CSV.
*   **Temperature / Supersaturation**: Mapped from $\alpha, \gamma$ using our Nakaya transform.
*   **Compactness**: $C = P^2/A$. A dimensionless measure of shape complexity.
*   **Tip Velocity**: $\Delta R_{max} / \Delta t$. Critical for testing the "Velocity-Growth Hypothesis".

---

## 🔭 Scientific Applications

This simulation framework opens several avenues for research:
1.  **Material Science**: Understanding dendritic growth in alloy solidification (metallurgy).
2.  **Atmospheric Physics**: Improving microphysics parameterisations in weather models by accurately predicting ice crystal terminal velocities based on shape.
3.  **Pattern Formation**: Studying the universality of branching instabilities in diffusion-limited aggregation systems.

---

## 📚 References

### Technical References
1.  **Reiter, C. A. (2005)**. "A local cellular model for snow crystal growth". *Chaos, Solitons & Fractals*, 23(4), 1111-1119.
2.  **Libbrecht, K. G. (2005)**. "The physics of snow crystals". *Reports on Progress in Physics*, 68(4), 855.
3.  **Gravner, J., & Griffeath, D. (2009)**. "Modeling snow crystal growth: A three-dimensional mesoscopic approach". *Physical Review E*, 79(1), 011601.

### See Also
*   [SnowCrystals.com](http://www.snowcrystals.com/) - Kenneth Libbrecht's definitive guide to snowflake physics.
*   [Wolfram MathWorld: Cellular Automata](https://mathworld.wolfram.com/CellularAutomaton.html)

---

## 🤝 Contributing & Credits

This project was developed with the assistance of advanced AI coding agents.
*   **Core Architecture**: Google DeepMind's **Antigravity** (Agentic AI).
*   **Code Generation & Optimization**: **Gemini 3** (Google) and **Claude Opus 4.5** (Anthropic).
*   **Refactoring Assistance**: **ChatGPT 5.2** (OpenAI).

All code is open-source. Contributions to improve the physical accuracy (e.g., adding hexagonal anisotropy to the diffusion kernel) are welcome.

---

## 📄 License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.
