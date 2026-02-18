"""
ENGINE.PY — Snowflake Physics Kernels
=====================================
Contains the core Reiter Cellular Automata logic (CPU & GPU).
"""
import numpy as np
import os

# --- JIT COMPILATION SETUP ---
try:
    from numba import jit
    JIT_AVAILABLE = True
except ImportError:
    JIT_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func): return func
        return decorator

# --- CUDA SETUP ---
try:
    from numba import cuda, float32, int32, bool_
    CUDA_AVAILABLE = cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False

# --- METRICS COLLECTOR ---
class MetricsCollector:
    def __init__(self, filepath):
        self.filepath = filepath
        self.buffer = []
        self.header_written = False
        
        # Ensure directory exists
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
        except:
            pass
        
    def log_frame(self, frame_idx, alpha, gamma, grid, freeze_times):
        """
        Calculates and buffers metrics for a single frame.
        """
        # 1. Area (Mass >= 1.0)
        frozen_mask = grid >= 1.0
        area = np.sum(frozen_mask)
        
        # 2. Radius (Max distance from center)
        rows, cols = grid.shape
        cy, cx = rows // 2, cols // 2
        
        if area > 0:
            py, px = np.where(frozen_mask)
            # Max radius is max dist
            # Optimization: We can approximate or do reduced calculation if slow
            # But numpy is fast enough for 600x600 usually
            dists_sq = (px - cx)**2 + (py - cy)**2
            max_r = np.sqrt(np.max(dists_sq))
        else:
            max_r = 0.0
            
        # 3. Perimeter (Boundary cells)
        # Simple Perimeter: Count of ice cells with at least one non-ice neighbor.
        padded = np.pad(frozen_mask, 1, mode='constant', constant_values=0)
        # Sum of neighbors
        neighbors = (padded[:-2, 1:-1].astype(int) + 
                     padded[2:, 1:-1].astype(int) + 
                     padded[1:-1, :-2].astype(int) + 
                     padded[1:-1, 2:].astype(int))
        
        exposed = (frozen_mask) & (neighbors < 4)
        perimeter = np.sum(exposed)
        
        # 4. Instantaneous Rates (Finite Difference)
        # Initialize prev values if not present
        if not hasattr(self, 'prev_area'):
            self.prev_area = 0
            self.prev_radius = 0.0
            self.prev_perim = 0
            
        dA_dt = area - self.prev_area
        dR_dt = max_r - self.prev_radius
        dP_dt = perimeter - self.prev_perim
        
        # Update prev
        self.prev_area = area
        self.prev_radius = max_r
        self.prev_perim = perimeter
        
        # 5. Physical Coordinates (Nakaya Mapping)
        # T = -10 - (alpha * 4.0)
        # Sigma = gamma * 50.0
        T = -10.0 - (float(alpha) * 4.0)
        sigma = float(gamma) * 50.0

        # Record
        self.buffer.append({
            'Frame': frame_idx,
            'Alpha': float(alpha),
            'Gamma': float(gamma),
            'Temperature': float(T),
            'Supersaturation': float(sigma),
            'Area': int(area),
            'Radius': float(max_r),
            'Perimeter': int(perimeter),
            'dA_dt': int(dA_dt),
            'dR_dt': float(dR_dt),
            'dP_dt': int(dP_dt)
        })
        
        # Flush every 100 frames to save memory
        if len(self.buffer) >= 100:
            self.flush()

    def flush(self):
        if not self.buffer: return
        
        import csv
        mode = 'a' if self.header_written else 'w'
        
        with open(self.filepath, mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.buffer[0].keys())
            if not self.header_written:
                writer.writeheader()
                self.header_written = True
            writer.writerows(self.buffer)
        
        self.buffer = []

    def close(self):
        self.flush()

# ==========================================
# PART 1: PHYSICS ENGINE (CustomReiter Model v40)
# ==========================================

# --- HEX GRID OFFSETS (Odd-r) ---
OFFSETS_ODD = np.array([[-1, 0], [-1, 1], [0, -1], [0, 1], [1, 0], [1, 1]], dtype=np.int32)
OFFSETS_EVEN = np.array([[-1, -1], [-1, 0], [0, -1], [0, 1], [1, -1], [1, 0]], dtype=np.int32)

@jit(nopython=True, fastmath=True)
def get_receptive_mask(total, rows, cols):
    receptive = np.zeros((rows, cols), dtype=np.bool_)
    for r in range(rows):
        offsets = OFFSETS_ODD if r % 2 == 1 else OFFSETS_EVEN
        for c in range(cols):
            # If cell is frozen (Total mass >= 1.0)
            if total[r, c] >= 1.0:
                receptive[r, c] = True
                continue
            
            # Check neighbors
            for i in range(6):
                nr, nc = r + offsets[i, 0], c + offsets[i, 1]
                if 0 <= nr < rows and 0 <= nc < cols:
                    if total[nr, nc] >= 1.0:
                        receptive[r, c] = True
                        break
    return receptive

@jit(nopython=True, fastmath=True)
def simul_step_reiter(total, alpha, gamma):
    rows, cols = total.shape
    
    # 1. Identify Receptive Cells
    receptive = get_receptive_mask(total, rows, cols)
    
    # 2. Split State: "v" (Diffusive Vapor), "c" (Crystal)
    v_vapor = np.zeros_like(total)
    c_crystal = np.zeros_like(total)
    
    for r in range(rows):
        for c in range(cols):
            if receptive[r, c]:
                v_vapor[r, c] = 0.0
                c_crystal[r, c] = total[r, c]
            else:
                v_vapor[r, c] = total[r, c]
                c_crystal[r, c] = 0.0
                
    # 3. Growth: Receptive cells gain Gamma (Crystal grows)
    for r in range(rows):
        for c in range(cols):
            if receptive[r, c]:
                c_crystal[r, c] += gamma
                
    # 4. Diffusion of Vapor ('v')
    v_new = np.zeros_like(v_vapor)
    
    for r in range(rows):
        offsets = OFFSETS_ODD if r % 2 == 1 else OFFSETS_EVEN
        for c in range(cols):
            local_sum = 0.0
            divisor = 0
            for i in range(6):
                nr, nc = r + offsets[i, 0], c + offsets[i, 1]
                if 0 <= nr < rows and 0 <= nc < cols:
                    local_sum += v_vapor[nr, nc]
                    divisor += 1
            
            if divisor > 0:
                avg = local_sum / divisor
                v_new[r, c] = v_vapor[r, c] + (alpha * 0.5) * (avg - v_vapor[r, c])
            else:
                v_new[r, c] = v_vapor[r, c]
                
    # 5. Recombine
    t_next = v_new + c_crystal
    return t_next

# ==========================================
# PART 1b: GPU PHYSICS ENGINE (CUDA)
# ==========================================

if CUDA_AVAILABLE:
    TPB = 16 # Threads Per Block

    @cuda.jit
    def kernel_identify_receptive(total, receptive, rows, cols):
        r, c = cuda.grid(2)
        if r >= rows or c >= cols: return
        
        if total[r, c] >= 1.0:
            receptive[r, c] = True
            return
            
        is_odd = (r % 2 == 1)
        found_ice = False
        
        # Unrolled neighbor check (same offsets as CPU)
        # N1 TL
        nr, nc = r - 1, (c if is_odd else c - 1)
        if nr >= 0 and nc >= 0 and nc < cols:
            if total[nr, nc] >= 1.0: found_ice = True
        # N2 TR
        if not found_ice:
            nr, nc = r - 1, (c + 1 if is_odd else c)
            if nr >= 0 and nc >= 0 and nc < cols:
                if total[nr, nc] >= 1.0: found_ice = True
        # N3 L
        if not found_ice:
            nr, nc = r, c - 1
            if nc >= 0:
                if total[nr, nc] >= 1.0: found_ice = True
        # N4 R
        if not found_ice:
            nr, nc = r, c + 1
            if nc < cols:
                if total[nr, nc] >= 1.0: found_ice = True
        # N5 BL
        if not found_ice:
            nr, nc = r + 1, (c if is_odd else c - 1)
            if nr < rows and nc >= 0 and nc < cols:
                if total[nr, nc] >= 1.0: found_ice = True
        # N6 BR
        if not found_ice:
            nr, nc = r + 1, (c + 1 if is_odd else c)
            if nr < rows and nc >= 0 and nc < cols:
                if total[nr, nc] >= 1.0: found_ice = True
                
        receptive[r, c] = found_ice

    @cuda.jit
    def kernel_growth_prepare(total, receptive, v_vapor, c_crystal, gamma, rows, cols):
        r, c = cuda.grid(2)
        if r >= rows or c >= cols: return
        is_rec = receptive[r, c]
        mass = total[r, c]
        if is_rec:
            v_vapor[r, c] = 0.0
            c_crystal[r, c] = mass + gamma
        else:
            v_vapor[r, c] = mass
            c_crystal[r, c] = 0.0

    @cuda.jit
    def kernel_diffusion(v_vapor, v_new, alpha, rows, cols):
        r, c = cuda.grid(2)
        if r >= rows or c >= cols: return
        center_v = v_vapor[r, c]
        is_odd = (r % 2 == 1)
        local_sum = 0.0
        divisor = 0
        
        # N1 TL
        nr, nc = r - 1, (c if is_odd else c - 1)
        if nr >= 0 and nc >= 0 and nc < cols:
            local_sum += v_vapor[nr, nc]; divisor += 1
        # N2 TR
        nr, nc = r - 1, (c + 1 if is_odd else c)
        if nr >= 0 and nc >= 0 and nc < cols:
            local_sum += v_vapor[nr, nc]; divisor += 1
        # N3 L
        nr, nc = r, c - 1
        if nc >= 0:
            local_sum += v_vapor[nr, nc]; divisor += 1
        # N4 R
        nr, nc = r, c + 1
        if nc < cols:
            local_sum += v_vapor[nr, nc]; divisor += 1
        # N5 BL
        nr, nc = r + 1, (c if is_odd else c - 1)
        if nr < rows and nc >= 0 and nc < cols:
            local_sum += v_vapor[nr, nc]; divisor += 1
        # N6 BR
        nr, nc = r + 1, (c + 1 if is_odd else c)
        if nr < rows and nc >= 0 and nc < cols:
            local_sum += v_vapor[nr, nc]; divisor += 1

        if divisor > 0:
            avg = local_sum / divisor
            v_new[r, c] = center_v + (alpha * 0.5) * (avg - center_v)
        else:
            v_new[r, c] = center_v

    @cuda.jit
    def kernel_combine(v_new, c_crystal, total, rows, cols):
        r, c = cuda.grid(2)
        if r >= rows or c >= cols: return
        total[r, c] = v_new[r, c] + c_crystal[r, c]

    @cuda.jit
    def kernel_check_edge(total, flag_array, rows, cols):
        r, c = cuda.grid(2)
        if r >= rows or c >= cols: return
        is_edge = (r < 2) or (r >= rows - 2) or (c < 2) or (c >= cols - 2)
        if is_edge:
            if total[r, c] >= 1.0:
                flag_array[0] = 1

    class SnowflakeGPU:
        def __init__(self, rows, cols, beta):
            import math
            self.rows = rows
            self.cols = cols
            # Host
            self.h_total = np.ones((rows, cols), dtype=np.float32) * beta
            self.h_total[rows//2, cols//2] = 1.0
            # Device
            self.d_total = cuda.to_device(self.h_total)
            self.d_receptive = cuda.device_array((rows, cols), dtype=np.bool_)
            self.d_v_vapor = cuda.device_array((rows, cols), dtype=np.float32)
            self.d_v_new = cuda.device_array((rows, cols), dtype=np.float32)
            self.d_c_crystal = cuda.device_array((rows, cols), dtype=np.float32)
            self.d_flag = cuda.device_array(1, dtype=np.int32)
            # Blocks
            blocks_x = math.ceil(rows / TPB)
            blocks_y = math.ceil(cols / TPB)
            self.blocks = (blocks_x, blocks_y)
            self.threads = (TPB, TPB)

        def step(self, alpha, gamma):
            kernel_identify_receptive[self.blocks, self.threads](self.d_total, self.d_receptive, self.rows, self.cols)
            kernel_growth_prepare[self.blocks, self.threads](self.d_total, self.d_receptive, self.d_v_vapor, self.d_c_crystal, gamma, self.rows, self.cols)
            kernel_diffusion[self.blocks, self.threads](self.d_v_vapor, self.d_v_new, alpha, self.rows, self.cols)
            kernel_combine[self.blocks, self.threads](self.d_v_new, self.d_c_crystal, self.d_total, self.rows, self.cols)

        def check_edge(self):
            cuda.to_device(np.array([0], dtype=np.int32), to=self.d_flag)
            kernel_check_edge[self.blocks, self.threads](self.d_total, self.d_flag, self.rows, self.cols)
            flag = self.d_flag.copy_to_host()[0]
            return (flag == 1)

        def get_result(self):
            return self.d_total.copy_to_host()

def run_single_sim_gpu(alpha, gamma, beta, steps, grid_size=600, stop_at_edge=True,
                       video_path=None, rainbow_path=None, metrics_path=None, 
                       video_4k_path=None, rainbow_4k_path=None, video_backend='vispy',
                       return_history=True):
                       
    engine = SnowflakeGPU(grid_size, grid_size, beta)
    sparse_history = []
    
    # --- OUTPUT HANDLERS ---
    writer = None
    writer_4k = None
    metrics_collector = None
    
    # Initialize Metrics
    if metrics_path:
        metrics_collector = MetricsCollector(metrics_path)
    
    # Initialize Video Writers
    from .video_writer import VideoWriter, VispyHexVideoWriter, MatplotlibHexVideoWriter
    from .utils import get_ice_cmap, log_debug
    
    # Select 4K video writer based on backend
    if video_backend.startswith('vispy'):
        HexWriter4K = VispyHexVideoWriter
    else:
        HexWriter4K = MatplotlibHexVideoWriter
    
    ice_cmap = get_ice_cmap()
    
    log_debug(f"Starting GPU Sim: Gamma={gamma} Alpha={alpha} Backend={video_backend}")
    
    if video_path:
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        writer = VideoWriter(video_path, grid_size, grid_size, fps=30)
        
    if video_4k_path:
        os.makedirs(os.path.dirname(video_4k_path), exist_ok=True)
        if video_backend.startswith('vispy'):
             writer_4k = HexWriter4K(video_4k_path, 4000, 4000, fps=30, grid_shape=(grid_size, grid_size), backend=video_backend)
        else:
             writer_4k = HexWriter4K(video_4k_path, 4000, 4000, fps=30, grid_shape=(grid_size, grid_size))

    max_steps = 100000000 if stop_at_edge else steps
    
    frame_idx = 0
    
    for i in range(1, max_steps + 1):
        engine.step(alpha, gamma)
        
        # Check every 50 steps
        if i % 50 == 0:
            frame_idx += 1
            
            # Fetch Data (Device -> Host)
            # Needed for Edge Check, Video, History
            current_grid = engine.get_result()
            
            if stop_at_edge:
                # Optimized: GPU-side check? 
                # We have check_edge method but let's using host copy for now since we needed it for video anyway?
                # Actually check_edge is faster on GPU because it returns boolean.
                # But we need grid for Video...
                pass # Logic handled below
            
            # Video Output
            if writer:
                writer.write_frame(current_grid, alpha, gamma, i, ice_cmap, normalization_val=1.5)
            
            if writer_4k:
                 writer_4k.write_frame(current_grid, alpha, gamma, i, ice_cmap, normalization_val=1.5, mode='mass')
                 
            # History
            if return_history:
                 sparse_history.append(current_grid)
                 if len(sparse_history) > 200:
                      sparse_history = sparse_history[::2]
                 
            # Metrics
            if metrics_collector:
                 # Freeze times not tracked on GPU easily. Passing zeros.
                 metrics_collector.log_frame(i, alpha, gamma, current_grid, np.zeros_like(current_grid))
            
            # Stop Conditions
            if stop_at_edge and engine.check_edge():
                log_debug(f"GPU Sim Finished (Edge): Step {i}")
                break
                
            # Log
            if i % 5000 == 0:
                print(f"      [GPU Engine] Step {i} ...")
                log_debug(f"GPU Step {i} - History Len: {len(sparse_history)}")
                
            # Active Safety Check (Every 50 steps is frequent enough)
            from .utils import check_memory_status
            check_memory_status()

    # Close Writers
    if writer: writer.close()
    if writer_4k: writer_4k.close()
    if metrics_collector: metrics_collector.close()
    
    final_grid = engine.get_result()
    if return_history:
        sparse_history.append(final_grid)
    
    # Fake Freeze Times (Not implemented on GPU)
    freeze_times = np.zeros((grid_size, grid_size), dtype=np.int32)
    # Post-process from history if needed, or leave blank for now.
    
    full_history_arr = np.array(sparse_history)
    
    return final_grid, freeze_times, full_history_arr

def run_single_sim(alpha, gamma, beta, steps, grid_size=600, equilibrium_mode=False, stop_at_edge=True, video_path=None, rainbow_path=None, metrics_path=None, video_4k_path=None, rainbow_4k_path=None, video_backend='vispy', return_history=True):
    # SIMULATION GRID
    sim_size = 600
    rows, cols = sim_size, sim_size
    
    total = np.ones((rows, cols)) * beta
    total[rows//2, cols//2] = 1.0
    
    # Track freeze time (step when cell turns >= 1.0)
    freeze_times = np.zeros((rows, cols), dtype=np.int32)
    freeze_times[rows//2, cols//2] = 1
    
    # SAFETY: Sparse history for 3D plot (Max 500 frames)
    # We will NOT store every frame if video_path is present.
    sparse_history = []
    
    # --- OUTPUT HANDLERS ---
    writer = None
    rainbow_writer = None
    writer_4k = None
    rainbow_writer_4k = None
    metrics_collector = None
    
    # Initialize Metrics
    if metrics_path:
        metrics_collector = MetricsCollector(metrics_path)
    
    # Initialize Video Writers
    from .video_writer import VideoWriter, HexVideoWriter, VispyHexVideoWriter, MatplotlibHexVideoWriter
    from .utils import get_ice_cmap, get_time_cmap
    
    # Select 4K video writer based on backend
    if video_backend.startswith('vispy'):
        HexWriter4K = VispyHexVideoWriter
        # We need to pass the backend name to the constructor
        # Use partial or just handle in instantiation
    else:  # 'matplotlib'
        HexWriter4K = MatplotlibHexVideoWriter
    
    if video_path:
        # Standard Mass Video
        ice_cmap = get_ice_cmap()
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        writer = VideoWriter(video_path, grid_size, grid_size, fps=30)
        
    if video_4k_path:
        # 4K Standard Mass Video - Use selected backend for hexagonal rendering
        ice_cmap = get_ice_cmap() # Ensure loaded
        os.makedirs(os.path.dirname(video_4k_path), exist_ok=True)
        
        if video_backend.startswith('vispy'):
            writer_4k = HexWriter4K(video_4k_path, 4000, 4000, fps=30, grid_shape=(sim_size, sim_size), backend=video_backend)
        else:
            writer_4k = HexWriter4K(video_4k_path, 4000, 4000, fps=30, grid_shape=(sim_size, sim_size))
        
    if rainbow_path:
        # Rainbow Time Video
        time_cmap = get_time_cmap()
        os.makedirs(os.path.dirname(rainbow_path), exist_ok=True)
        rainbow_writer = VideoWriter(rainbow_path, grid_size, grid_size, fps=30)

    if rainbow_4k_path:
        # 4K Rainbow Time Video - Use selected backend for hexagonal rendering
        time_cmap = get_time_cmap() # Ensure loaded
        os.makedirs(os.path.dirname(rainbow_4k_path), exist_ok=True)
        
        if video_backend.startswith('vispy'):
            rainbow_writer_4k = HexWriter4K(rainbow_4k_path, 4000, 4000, fps=30, grid_shape=(sim_size, sim_size), backend=video_backend)
        else:
            rainbow_writer_4k = HexWriter4K(rainbow_4k_path, 4000, 4000, fps=30, grid_shape=(sim_size, sim_size))
    
    # Determine Loop Limit
    # User requested NO LIMIT by default (infinite growth until edge/equilibrium)
    # Only enforce 'steps' if fixed_steps mode is implicitly active (not stop_at_edge/equil)
    simulation_limit = None
    if not (equilibrium_mode or stop_at_edge):
        simulation_limit = steps

    # Run simulation
    i = 0
    while True:
        i += 1
        
        # Check Step Limit
        if simulation_limit and i > simulation_limit:
            break
            
        prev_total = total
        total = simul_step_reiter(total, alpha, gamma)
        
        # Track Freezing
        newly_frozen = (total >= 1.0) & (freeze_times == 0)
        freeze_times[newly_frozen] = i
        
        # --- VIDEO STREAMING & METRICS ---
        if writer:
            # Standard Mass Video (Clip at 1.5)
            writer.write_frame(total, alpha, gamma, i, ice_cmap, normalization_val=1.5)
            
        if rainbow_writer:
            # Rainbow Time Video
            # Normalize by current frame index 'i' so new growth is always red (max)
            # and old growth shifts towards blue.
            rainbow_writer.write_frame(freeze_times, alpha, gamma, i, time_cmap, normalization_val=i)
            
        # --- 4K VIDEO STREAMING (High-quality hexagonal rendering) ---
        if writer_4k:
            writer_4k.write_frame(total, alpha, gamma, i, ice_cmap, normalization_val=1.5, mode='mass')
            
        if rainbow_writer_4k:
            rainbow_writer_4k.write_frame(freeze_times, alpha, gamma, i, time_cmap, normalization_val=i, mode='time')
            
        if metrics_collector:
            metrics_collector.log_frame(i, alpha, gamma, total, freeze_times)
            
        # --- SPARSE HISTORY (For 3D Plot, limited memory) ---
        # Adaptive: if i is small, take more. If i is large, take less.
        # Target ~400 frames total.
        # Simple heuristic: Record every (MaxSteps // 400) steps? No, we don't know MaxSteps.
        # Just record every 50th frame, and if length > 400, decimate (drop half).
        # Just record every 50th frame, and if length > 400, decimate (drop half).
        if i % 50 == 0:
            if return_history:
                 sparse_history.append(total.copy())
                 if len(sparse_history) > 200:
                     # Decimate: Keep even indices
                     sparse_history = sparse_history[::2]
            
        # Debug Output
        if i % 5000 == 0:
            print(f"      [Engine] Step {i} ...") 
            
        # --- STOPPING CONDITIONS ---
        if stop_at_edge:
             if np.any(total[0:2, :] >= 1.0) or np.any(total[-2:, :] >= 1.0) or \
                np.any(total[:, 0:2] >= 1.0) or np.any(total[:, -2:] >= 1.0):
                 break

        if equilibrium_mode:
            if np.sum(np.abs(total - prev_total)) < 1e-9:
                break
        
        if not (equilibrium_mode or stop_at_edge) and i % 50 == 0:
            if np.sum(np.abs(total - prev_total)) < 0.001:
                break
                
    # Close Writers & Collectors
    if writer: writer.close()
    if rainbow_writer: rainbow_writer.close()
    if writer_4k: writer_4k.close()
    if rainbow_writer_4k: rainbow_writer_4k.close()
    if metrics_collector: metrics_collector.close()
            
    # Always append final state to history
    sparse_history.append(total.copy())
    
    # CROP TO VISUALIZATION SIZE
    center_r, center_c = rows // 2, cols // 2
    half = grid_size // 2
    r_start = center_r - half; r_end = r_start + grid_size
    c_start = center_c - half; c_end = c_start + grid_size
    
    final_crop = total[r_start:r_end, c_start:c_end]
    freeze_crop = freeze_times[r_start:r_end, c_start:c_end]
    
    # Crop History
    history_crop = []
    # If we decimated, sparse_history is already small.
    # Just ensure it's not huge.
    step_safe = max(1, len(sparse_history) // 500)
    for h in sparse_history[::step_safe]:
        history_crop.append(h[r_start:r_end, c_start:c_end])
        
    return final_crop, freeze_crop, np.array(history_crop)
