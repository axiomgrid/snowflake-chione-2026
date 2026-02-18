# Snowflake Chione 2026 — Makefile
# Generates snowflake crystal simulations (MP4, PNG, CSV)

PYTHON := python3
PIP := pip3

.PHONY: help setup test scan scan-parallel research clean docker-build docker-run

help:
	@echo "Snowflake Chione 2026 — Simulation Commands"
	@echo ""
	@echo "  make setup          — Install Python dependencies"
	@echo "  make test           — Run a fast 4-simulation test"
	@echo "  make scan           — Run the full 12×12 parameter sweep (serial)"
	@echo "  make scan-parallel  — Run the full 12×12 sweep (CPU multiprocessing)"
	@echo "  make research       — Run MAP-Elites shape exploration"
	@echo "  make clean          — Remove __pycache__ directories"
	@echo "  make docker-build   — Build the Docker image"
	@echo "  make docker-run     — Run the scan inside Docker"

setup:
	@echo ">> Installing dependencies..."
	$(PIP) install -r requirements.txt
	@echo ">> Done. Ensure FFmpeg is installed: apt install ffmpeg (Linux) or brew install ffmpeg (macOS)"

test:
	@echo ">> Running fast test (4 simulations)..."
	$(PYTHON) -m src.cli --fast_test --force_cpu

# Full 12×12 parameter sweep (144 simulations → MP4, PNG, CSV)
scan:
	@echo ">> Running 12×12 parameter sweep..."
	$(PYTHON) -m src.cli \
		--alpha_min 0.01 --alpha_max 2.5 \
		--gamma_min 0.0001 --gamma_max 0.01 \
		--gif

scan-parallel:
	@echo ">> Running 12×12 parameter sweep (parallel)..."
	$(PYTHON) -m src.cli \
		--parallel --force_cpu \
		--alpha_min 0.01 --alpha_max 2.5 \
		--gamma_min 0.0001 --gamma_max 0.01 \
		--gif

# MAP-Elites quality-diversity exploration (300 iterations)
# Usage: make research GIVEN_DIR=results/Run_... BUDGET=50
research:
	@if [ -z "$(GIVEN_DIR)" ]; then echo "Error: specify seed directory with GIVEN_DIR=results/..."; exit 1; fi
	@echo ">> Running MAP-Elites shape exploration..."
	$(PYTHON) -m src.cli \
		--research_shapes \
		--given_dirs $(GIVEN_DIR) \
		--research_output "$(or $(OUTPUT_DIR),research_output)" \
		--research_budget $(or $(BUDGET),50) \
		--map_resolution $(or $(MAP_RES),10) \
		--force_cpu

clean:
	@echo ">> Cleaning __pycache__..."
	@find . -type d -name "__pycache__" -exec rm -rf {} +

# 3D interactive visualization (generates viz_3d.html)
# Usage: make viz-3d DIR=results/Run_...
viz-3d:
	@if [ -z "$(DIR)" ]; then echo "Error: specify result directory with DIR=results/..."; exit 1; fi
	@echo ">> Generating 3D visualization for $(DIR)..."
	$(PYTHON) src/viz_3d_builder.py --results_dir "$(DIR)"
	@echo ">> Done! Open $(DIR)/research_shapes/viz_3d.html in your browser."

# Docker
docker-build:
	docker build -t snowflake-chione-2026 .

docker-run:
	docker run --rm -it \
		-v "$$(pwd)/results:/app/results" \
		snowflake-chione-2026
