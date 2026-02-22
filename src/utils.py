"""
UTILS.PY — Helper Functions
===========================
Metrics calculation and Colormap definitions.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --- CUSTOM COLORMAPS ---
def get_ice_cmap():
    """
    White-Blue Gradient Explanation:
    Matches physical density of water/ice.
    0.0 to 0.2 (Black/Dark Blue): Vapor background.
    0.8 (Cyan): Thin ice or deposition front.
    1.0+ (White): Solid crystal structure.
    """
    colors = [
        (0.0, "#000000"), # Background
        (0.2, "#0a1a2b"), # Vapor
        (0.8, "#a0e8ff"), # Ice Edge
        (1.0, "#ffffff")  # Solid Ice
    ]
    return LinearSegmentedColormap.from_list("ice_theme", colors)

def get_time_cmap():
    """Rainbow spectrum for time evolution (Oldest=Blue, Newest=Red)"""
    return plt.get_cmap('turbo')

def calculate_metrics(grid):
    # Simple metrics for shape classification
    frozen = grid >= 1.0
    area = np.sum(frozen)
    if area == 0: return 0, 0, 0, 0, 0, 0, 0
    
    # Perimeter Estimation (Count frozen cells with < 4 frozen neighbors)
    # Simple convolution-like approach
    # Up, Down, Left, Right neighbors
    padded = np.pad(frozen.astype(int), 1, mode='constant', constant_values=0)
    neighbors = (padded[:-2, 1:-1] + padded[2:, 1:-1] + 
                 padded[1:-1, :-2] + padded[1:-1, 2:])
    
    # Boundary cells are frozen cells with < 4 neighbors (exposed to air)
    boundary_mask = (frozen) & (neighbors < 4)
    perimeter = np.sum(boundary_mask)
    
    # Calculate Max Radius (Distance from center)
    rows, cols = grid.shape
    cy, cx = rows // 2, cols // 2
    
    py, px = np.where(boundary_mask)
    if len(py) > 0:
        dists = np.sqrt((px - cx)**2 + (py - cy)**2)
        max_radius = np.max(dists)
    else:
        # Fallback if no boundary (should be covered by area==0 check, but safe)
        max_radius = 1.0
    
    # Aspect Ratio (Circle fit)
    expected_area = np.pi * (max_radius**2)
    ratio = area / (expected_area + 1e-6)
    
    # Complexity (Perimeter / Area) is also useful
    # Using P^2 / A (Compactness) is even better
    compactness = (perimeter**2) / (area + 1e-6)
    
    # --- Correct Hex Perimeter (for normalized metrics) ---
    # The original perimeter uses 4-connected square neighbors, which is wrong for
    # hex grids (odd-r offset). We calculate exposed hex edges properly here.
    # Odd-r hex offsets:
    #   Even row neighbors: (-1,-1),(-1,0),(0,-1),(0,1),(1,-1),(1,0)
    #   Odd  row neighbors: (-1,0),(-1,1),(0,-1),(0,1),(1,0),(1,1)
    padded_hex = np.pad(frozen.astype(np.int32), 1, mode='constant', constant_values=0)
    rows_h, cols_h = frozen.shape
    exposed_edges = np.zeros_like(frozen, dtype=np.int32)
    
    for r in range(rows_h):
        rp = r + 1  # padded index
        if r % 2 == 0:
            # Even row offsets: (-1,-1),(-1,0),(0,-1),(0,1),(1,-1),(1,0)
            hex_neighbors = (padded_hex[rp-1, 1:-1] +         # (-1, 0) -> col stays
                             padded_hex[rp+1, 1:-1] +         # (+1, 0) -> col stays
                             padded_hex[rp, :-2] +             # (0, -1)
                             padded_hex[rp, 2:] +              # (0, +1)
                             padded_hex[rp-1, :-2] +           # (-1, -1)
                             padded_hex[rp+1, :-2])            # (+1, -1)
        else:
            # Odd row offsets: (-1,0),(-1,1),(0,-1),(0,1),(1,0),(1,1)
            hex_neighbors = (padded_hex[rp-1, 1:-1] +         # (-1, 0) -> col stays
                             padded_hex[rp+1, 1:-1] +         # (+1, 0) -> col stays
                             padded_hex[rp, :-2] +             # (0, -1)
                             padded_hex[rp, 2:] +              # (0, +1)
                             padded_hex[rp-1, 2:] +            # (-1, +1)
                             padded_hex[rp+1, 2:])             # (+1, +1)
        exposed_edges[r, :] = (6 - hex_neighbors) * frozen[r, :].astype(np.int32)
    
    perimeter_for_compactness_normalized = int(np.sum(exposed_edges))
    
    # Normalized Compactness: Isoperimetric ratio (4*pi*A / P^2)
    # Perfect circle = 1.0, complex shapes approach 0.0
    # Uses hex-corrected perimeter for accuracy
    compactness_normalized = (4 * np.pi * area) / (perimeter_for_compactness_normalized**2 + 1e-6)
    compactness_normalized = min(1.0, max(0.0, compactness_normalized))
    
    # Branching Factor: ratio of perimeter to convex hull perimeter approximation
    # Higher = more branched/fractal. Approximated via P / (2*pi*R)
    # Uses hex-corrected perimeter
    branching_factor = perimeter_for_compactness_normalized / (2 * np.pi * max_radius + 1e-6)
    
    return area, perimeter, ratio, compactness, max_radius, compactness_normalized, branching_factor

def get_visible_surface(history):
    # Returns mask of cells that are frozen in the history volume
    # history is (frames, rows, cols)
    # We treat any cell >= 1.0 as "frozen"
    return (history >= 1.0)

    return (history >= 1.0)

# --- DEBUG LOGGING ---
import datetime
import os
import resource

LOG_FILE = "debug_crash_log.txt"

def log_debug(msg):
    """
    Append message to log file immediately (flush to disk).
    Used to diagnose OOM crashes where stdout is lost.
    """
    timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")
    
    # Calculate Memory Usage (RSS)
    mem_usage_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    mem_usage_mb = mem_usage_kb / 1024.0
    
    formatted_msg = f"[{timestamp}] [MEM: {mem_usage_mb:.1f} MB] {msg}\n"
    
    try:
        with open(LOG_FILE, "a") as f:
            f.write(formatted_msg)
            f.flush()
            os.fsync(f.fileno()) # Force write to disk
    except Exception as e:
        print(f"Logging Failed: {e}")

def reset_log():
    try:
        with open(LOG_FILE, "w") as f:
            f.write(f"--- STARTED NEW LOG SESSION: {datetime.datetime.now()} ---\n")
    except:
        pass

GLOBAL_MEM_LIMIT_GB = 0.0

def get_system_memory_info():
    """Reads /proc/meminfo to get current memory status in GB."""
    mem_info = {}
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                parts = line.split(':')
                if len(parts) == 2:
                    key = parts[0].strip()
                    val_kb = int(parts[1].strip().split()[0])
                    mem_info[key] = val_kb * 1024 # Bytes
    except:
        pass
    return mem_info

def limit_memory(max_gb=None):
    """
    Sets the active monitoring threshold. Does NOT use setrlimit (incompatible with CUDA).
    """
    global GLOBAL_MEM_LIMIT_GB
    try:
        mem_info = get_system_memory_info()
        available_bytes = mem_info.get('MemAvailable', mem_info.get('MemFree', 0))
        total_bytes = mem_info.get('MemTotal', 0)
        
        if max_gb:
            GLOBAL_MEM_LIMIT_GB = max_gb
            mode = "Static"
        else:
            # Dynamic: 90% of Total (Physical)
            # We use Total instead of Available for the LIMIT because RSS tracks total usage, not delta.
            # But we want to ensure we don't exceed system capabilities.
            # Safe bet: 85% of Total Physical RAM to leave room for kernel/others.
            GLOBAL_MEM_LIMIT_GB = (total_bytes * 0.85) / (1024**3)
            mode = "Dynamic (85% Total)"
            
        avail_gb = available_bytes / (1024**3)
        total_gb = total_bytes / (1024**3)
        swap_total = mem_info.get('SwapTotal', 0) / (1024**3)
        swap_free = mem_info.get('SwapFree', 0) / (1024**3)
        
        msg = f"Active Memory Monitor: Limit={GLOBAL_MEM_LIMIT_GB:.2f} GB (System: {avail_gb:.2f}/{total_gb:.2f} GB Free, Swap: {swap_free:.2f}/{swap_total:.2f} GB Free)"
        print(f"   [System] {msg}")
        log_debug(msg)
        
        if swap_free < 0.5:
             print(f"   [WARNING] Low Swap Space ({swap_free:.2f} GB). Application may be unstable.")
        
        # Disable hard limits if any were set previously
        try:
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            resource.setrlimit(resource.RLIMIT_AS, (resource.RLIM_INFINITY, hard))
        except:
            pass
            
    except Exception as e:
        print(f"   [System] Failed to set memory monitor: {e}")

def check_memory_status():
    """
    Checks if current RSS usage exceeds the limit. Raises MemoryError if so.
    Call this inside tight loops.
    """
    global GLOBAL_MEM_LIMIT_GB
    if GLOBAL_MEM_LIMIT_GB <= 0: return
    
    # RSS is in KB on Linux
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_gb = rss_kb / (1024 * 1024)
    
    
    if rss_gb > GLOBAL_MEM_LIMIT_GB:
        msg = f"SAFETY KILL: Process Rss ({rss_gb:.2f} GB) exceeded limit ({GLOBAL_MEM_LIMIT_GB:.2f} GB)"
        log_debug(msg)
        raise MemoryError(msg)

def memory_checkpoint(label):
    """
    Logs current memory usage to stdout and log file.
    Enforces HARD KILL if limit exceeded to protect system.
    """
    try:
        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_mb = rss_kb / 1024.0
        rss_gb = rss_kb / (1024 * 1024)
        
        msg = f"[MEM_CHECK] {label}: {rss_mb:.1f} MB (Limit: {GLOBAL_MEM_LIMIT_GB:.2f} GB)"
        print(f"   {msg}")
        log_debug(msg)
        
        # Also try to write to /etc/mylastrun.log as requested
        try:
            with open("/etc/mylastrun.log", "a") as f:
                f.write(f"{datetime.datetime.now()} - {msg}\n")
        except:
            pass # Ignore permission errors

        # HARD KILL SWITCH
        if GLOBAL_MEM_LIMIT_GB > 0 and rss_gb > GLOBAL_MEM_LIMIT_GB:
            kill_msg = f"!!! SAFETY KILL !!! Process RSS ({rss_gb:.2f} GB) > Limit ({GLOBAL_MEM_LIMIT_GB:.2f} GB). Terminating immediately."
            print(kill_msg)
            log_debug(kill_msg)
            try:
                with open("/etc/mylastrun.log", "a") as f:
                    f.write(f"{datetime.datetime.now()} - {kill_msg}\n")
            except:
                pass
            
            import signal
            os.kill(os.getpid(), signal.SIGKILL)
            
    except Exception as e:
        # Don't let logging failure stop execution, unless it's the kill switch
        pass
