"""
CLI.PY — Command Line Interface
===============================
Entry point for the Snowflake Scanner.
Orchestrates simulation, multiprocessing, and visualization.
"""
import argparse
import os
import sys
import time
import multiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc

# Import local modules
# (Relative imports work when run as a package: python -m src.cli)
# (Relative imports work when run as a package: python -m src.cli)
from .engine import run_single_sim, run_single_sim_gpu, CUDA_AVAILABLE
# Ensure we don't accidentally import DummyWriter or limit logic here, relying on updated video_writer
from .utils import calculate_metrics, get_visible_surface
from .viz import save_gif_from_history, plot_hex_in_ax, write_html_index
from .plotting import get_physical_coords, plot_nakaya_diagram, generate_html_dashboard

# --- WORKER FUNCTION FOR MULTIPROCESSING ---
def worker_simulation(alpha, gamma, beta, steps, equilibrium, stop_at_edge, use_gpu, video_path, rainbow_path, metrics_path, video_4k_path, rainbow_4k_path, video_backend='vispy', grid_cache_path=None, return_history=True):
    """
    Top-level function for multiprocessing.
    Returns: (final_grid, freeze_grid, history, alpha, gamma)
    """
    # --- 0. CACHE RESUME ---
    if grid_cache_path and os.path.exists(grid_cache_path):
        try:
            # Load from Cache
            data = np.load(grid_cache_path)
            final = data['final']
            freeze = data['freeze']
            
            # History might be large, only load if needed/available
            if 'history' in data and return_history:
                hist = data['history']
            else:
                hist = np.array([]) # Empty if not needed or not found
                
            # Log cache hit (visible in serial mode)
            print(f"   [Cache] Loaded: {os.path.basename(grid_cache_path)}") 
            return (final, freeze, hist, alpha, gamma)
        except Exception as e:
            # If load fails, ignore and regenerate
            pass

    # --- 1. SIMULATION ---
    if use_gpu and CUDA_AVAILABLE:
        final, freeze, hist = run_single_sim_gpu(alpha, gamma, beta, steps=steps, 
                                                 stop_at_edge=stop_at_edge,
                                                 video_path=video_path,
                                                 rainbow_path=rainbow_path,
                                                 metrics_path=metrics_path,
                                                 video_4k_path=video_4k_path,
                                                 rainbow_4k_path=rainbow_4k_path,
                                                 video_backend=video_backend,
                                                 return_history=return_history)
    else:
        # CPU Engine
        final, freeze, hist = run_single_sim(alpha, gamma, beta, steps=steps, 
                                             equilibrium_mode=equilibrium, 
                                             stop_at_edge=stop_at_edge,
                                             video_path=video_path,
                                             rainbow_path=rainbow_path,
                                             metrics_path=metrics_path,
                                             rainbow_4k_path=rainbow_4k_path,
                                             video_backend=video_backend,
                                             return_history=return_history)
    
    # --- 2. CACHE SAVE ---
    if grid_cache_path:
        # Save compressed binary (Fast & Efficient)
        np.savez_compressed(grid_cache_path, final=final, freeze=freeze, history=hist)
        
        # Also save plain CSV for final grid (User Request: "serialized text format")
        # Name: grid_Alpha..._Gamma....csv
        # Derive from grid_cache_path which is likely .../grid_data/data_Alpha...npz
        # We replace .npz with .csv and data_ with grid_
        csv_path = grid_cache_path.replace('.npz', '.csv').replace('data_', 'grid_')
        np.savetxt(csv_path, final, delimiter=",", fmt='%.4f')
    
    # Return results only if requested (last arg in args tuple unpacted in wrapper)
    # But wait, we unpack *args in wrapper. The wrapper receives the tuple.
    # We need to change the signature of worker_simulation to accept return_results
    return (final, freeze, hist, alpha, gamma)

# --- WRAPPER FOR IMAP ---
def worker_wrapper(args):
    # Args: (alpha, gamma, ..., video_backend, return_results)
    alpha, gamma = args[0], args[1]
    return_results = args[-1]
    
    # Call worker with all args except the last one (return_results)
    # Pass return_results as return_history to save massive memory on video-only tasks
    res = worker_simulation(*args[:-1], return_history=return_results)
    
    if return_results:
        return res
    else:
        return None

def run_scan(args):
    print(f"--- SNOWFLAKE SCANNER (Unified v40 Engine) ---")
    
    # Auto-Detect/Enable GPU
    use_gpu_engine = False
    if CUDA_AVAILABLE:
        print(">> NVIDIA GPU Detected: Enabling CUDA Acceleration by default.")
        print("   (Use --force_cpu to disable if desired)")
        use_gpu_engine = True
    
    if args.gpu: use_gpu_engine = True # Explicit flag
    if args.force_cpu: use_gpu_engine = False # Explicit disable

    # EXPONENTIAL ALPHA RANGE (Updated)
    # Added more intermediates as requested: 0.5, 0.7, 1.2, 1.8
    full_alphas = np.array([0.01, 0.03, 0.1, 0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 10.0, 30.0])
    alphas = full_alphas[(full_alphas >= args.alpha_min) & (full_alphas <= args.alpha_max)]
    if len(alphas) == 0:
        alphas = np.array([1.0])
    gammas = np.linspace(args.gamma_min, args.gamma_max, len(alphas)) 
    
    if args.fast_test:
        print("!! FAST TEST MODE !!")
        alphas = np.array([1.0, 1.5]) 
        gammas = np.array([0.001, 0.005]) 
        
    # User Request: Start with biggest for each so we have initial results faster
    alphas = np.flip(alphas) # High -> Low
    gammas = np.flip(gammas) # High -> Low
    
    # CUSTOM SCAN: Override alphas/gammas if custom list requested
    # Supports "missing cells" recovery mode
    custom_tasks = []
    if args.custom_scan:
        print("!! CUSTOM SCAN MODE (Recovering Missing Cells) !!")
        # Hardcoded list of missing cells from User Request (43 cells)
        # 3 columns (0.01, 0.03, 0.1) * 12 gammas = 36 cells
        # 1 column (0.3) * bottom 7 gammas = 7 cells
        # Total = 43 cells
        
        # Determine the full ranges to pick from
        full_gammas = np.linspace(args.gamma_min, args.gamma_max, 12) # 12 gammas
        
        # 1. Alphas 0.01, 0.03, 0.1 (All gammas)
        for a in [0.01, 0.03, 0.1]:
            for g in full_gammas:
                custom_tasks.append((a, g))
                
        # Update Reference Arrays for correct indexing if possible
        # (Optional, but helps stitching if we reconstruct the grid)
        # For now, relying on try-except safe lookup.
                
        # 2. Alpha 0.3 (Bottom 7 gammas)
        # "Bottom" means lower gamma values? Or bottom of the visual chart?
        # Visual chart typically has Gamma increasing upwards or downwards?
        # Based on standard plot: Y-axis is Gamma. Usually sorted min->max (0.0001 bottom, 0.01 top).
        # "Bottom 7 values" likely means the 7 smallest/lowest gammas.
        # Let's target the 7 lowest gammas.
        sorted_gammas_asc = np.sort(full_gammas)
        bottom_7_gammas = sorted_gammas_asc[:7]
        
        for g in bottom_7_gammas:
              custom_tasks.append((0.3, g))
              
        print(f"Custom Scan: Queued {len(custom_tasks)} specific simulations.")
    
    # SCAN VIDEOS MODE (Recover from MP4s)
    if args.scan_videos:
        print(f"!! SCAN VIDEOS MODE: Recovering from {args.scan_videos} !!")
        import re
        args.custom_scan = True # Treat as custom list
        base_output_dir = args.scan_videos
        videos_dir = os.path.join(base_output_dir, "Videos")
        
        if not os.path.exists(videos_dir):
            print(f"Error: Videos directory '{videos_dir}' not found.")
            sys.exit(1)
            
        # Scan for snowflake mp4s
        # Pattern: snowflake_Alpha0.30_Gamma0.0055.mp4
        pattern = re.compile(r"snowflake_Alpha([0-9.]+)_Gamma([0-9.]+)\.mp4")
        
        found_tasks = []
        for f in os.listdir(videos_dir):
            match = pattern.match(f)
            if match:
                a = float(match.group(1))
                g = float(match.group(2))
                found_tasks.append((a,g))
        
        print(f"Found {len(found_tasks)} existing videos.")
        custom_tasks = found_tasks # Override custom tasks
        
        # We need values for alphas/gammas arrays for plotting axis
        # Extract unique sorted values
        if len(custom_tasks) > 0:
             unique_a = sorted(list(set(t[0] for t in custom_tasks)), reverse=True)
             unique_g = sorted(list(set(t[1] for t in custom_tasks)), reverse=True)
             alphas = np.array(unique_a)
             gammas = np.array(unique_g)

    # SCAN GRID DATA MODE (Recover from NPZ cache)
    if args.scan_grid_data:
        print(f"!! SCAN GRID DATA MODE: Recovering from {args.scan_grid_data} !!")
        import re
        args.custom_scan = True
        base_output_dir = args.scan_grid_data
        grid_data_dir = os.path.join(base_output_dir, "grid_data")
        
        if not os.path.exists(grid_data_dir):
            print(f"Error: Grid Data directory '{grid_data_dir}' not found.")
            sys.exit(1)
            
        # Scan for data_Alpha...npz
        # Pattern: data_Alpha2.50_Gamma0.0091.npz
        pattern = re.compile(r"data_Alpha([0-9.]+)_Gamma([0-9.]+)\.npz")
        
        found_tasks = []
        for f in os.listdir(grid_data_dir):
            match = pattern.match(f)
            if match:
                a = float(match.group(1))
                g = float(match.group(2))
                found_tasks.append((a,g))
                
        print(f"Found {len(found_tasks)} existing cached grid files.")
        custom_tasks = found_tasks
        
        if len(custom_tasks) > 0:
             unique_a = sorted(list(set(t[0] for t in custom_tasks)), reverse=True)
             unique_g = sorted(list(set(t[1] for t in custom_tasks)), reverse=True)
             alphas = np.array(unique_a)
             gammas = np.array(unique_g)

    print(f"Scanning Exponential Alphas: {alphas}")
    print(f"Scanning Linear Gammas: [{gammas[0]:.4f} ... {gammas[-1]:.4f}]")
    print(f"Resolution: {len(alphas)}x{len(gammas)} = {len(alphas)*len(gammas)} simulations")
    
    # --- UNIQUE OUTPUT DIRECTORY ---
    first_gamma = gammas[0] if len(gammas) > 0 else 0
    last_gamma = gammas[-1] if len(gammas) > 0 else 0
    
    # --- OUTPUT DIRECTORY ---
    if args.resume:
        base_output_dir = args.resume
        print(f"RESUMING from existing directory: {base_output_dir}")
        if not os.path.exists(base_output_dir):
             print(f"Error: Resume directory '{base_output_dir}' does not exist.")
             sys.exit(1)
    elif args.scan_videos:
        # Already set above
        pass
    elif args.scan_grid_data:
        # Already set above
        pass
    else:    
        epoch_time = int(time.time())
        # Use simple name if custom scan to avoid weird 8x12 names if arrays not updated
        if args.custom_scan:
             run_name = f"Run_v39_CustomScan_Beta{args.beta}_{epoch_time}"
        else:
             run_name = f"Run_v39_Res{len(alphas)}x{len(gammas)}_Beta{args.beta}_{epoch_time}"
        base_output_dir = os.path.join("results", run_name)
    
    
    intermediate_dir = os.path.join(base_output_dir, "Intermediate")
    spacetime_dir = os.path.join(base_output_dir, "SpaceTime")
    videos_dir = os.path.join(base_output_dir, "Videos")
    grid_data_dir = os.path.join(base_output_dir, "grid_data") # New for Cache
    
    print(f"Creating Run Directory: {base_output_dir}")
    os.makedirs(intermediate_dir, exist_ok=True)
    os.makedirs(spacetime_dir, exist_ok=True)
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(grid_data_dir, exist_ok=True)

    # Update Output Paths
    args.output_csv = os.path.join(base_output_dir, "scan_results.csv")
    args.output_plot = os.path.join(base_output_dir, "phase_diagram.png")
    
    results = []
    
    # --- CANVAS SETUP ---
    rows = len(gammas)
    cols = len(alphas)
    # INCREASED RESOLUTION FOR PHASE DIAGRAM (Restored to 300 DPI)
    # With OOM fallback
    thumb_dpi = 300
    fig_size_inch = 4
    
    try:
        thumb_w = int(fig_size_inch * thumb_dpi)
        thumb_h = int(fig_size_inch * thumb_dpi)
        canvas_w = cols * thumb_w
        canvas_h = rows * thumb_h
        print(f"Initializing Canvas: {canvas_w}x{canvas_h} pixels (~{canvas_w*canvas_h*4/1e9:.2f} GB)")
        final_canvas = np.ones((canvas_h, canvas_w, 4), dtype=np.uint8) * 255
    except MemoryError:
        print("WARNING: OOM for High Res Canvas. Falling back to 72 DPI.")
        thumb_dpi = 72
        thumb_w = int(fig_size_inch * thumb_dpi)
        thumb_h = int(fig_size_inch * thumb_dpi)
        canvas_w = cols * thumb_w
        canvas_h = rows * thumb_h
        final_canvas = np.ones((canvas_h, canvas_w, 4), dtype=np.uint8) * 255

    # 3D Canvas (Conditional Allocation)
    spacetime_canvas = None
    if not args.no_3d:
        try:
            print(f"Initializing SpaceTime Canvas: {canvas_w}x{canvas_h} pixels")
            spacetime_canvas = np.ones((canvas_h, canvas_w, 4), dtype=np.uint8) * 255
        except MemoryError:
            print("WARNING: OOM for SpaceTime Canvas. Disabling 3D Grid.")
            spacetime_canvas = None
    
    start_time = time.time()
    sorted_gammas = gammas
    
# --- 1. PREPARE TASKS ---
    tasks = []
    
    if args.custom_scan:
        # Custom Mode: Iterate over specific (alpha, gamma) tuples
        for (a, g) in custom_tasks:
             # Construction Video Path
             video_name = f"snowflake_Alpha{a:.2f}_Gamma{g:.4f}.mp4"
             video_path = os.path.join(videos_dir, video_name)
             
             # ... (rest of logic tailored for custom tasks) ...
             # We can reuse the loop body if we structure it right, 
             # OR just duplicate the inner logic for clarity.
             # Let's duplicate to ensure we strictly follow the custom list.
             
             metrics_name = f"metrics_Alpha{a:.2f}_Gamma{g:.4f}.csv"
             metrics_path = os.path.join(videos_dir, metrics_name)
             
             # 600x600 Video Paths
             if not args.no_rainbow:
                 rainbow_name = f"rainbow_Alpha{a:.2f}_Gamma{g:.4f}.mp4"
                 rainbow_path = os.path.join(videos_dir, rainbow_name)
             else:
                 rainbow_path = None
                 
             # 4K Paths (Only if --video-4000)
             video_4k_path = None
             rainbow_4k_path = None
             if args.video_4000:
                 video_4k_name = f"snowflake_4k_Alpha{a:.2f}_Gamma{g:.4f}.mp4"
                 video_4k_path = os.path.join(videos_dir, video_4k_name)
                 if not args.no_rainbow:
                     rainbow_4k_name = f"rainbow_4k_Alpha{a:.2f}_Gamma{g:.4f}.mp4"
                     rainbow_4k_path = os.path.join(videos_dir, rainbow_4k_name)
             
             # Enforce --video-600 default if not 4k
             generate_600 = args.video_600 and not args.video_4000
             
             # Grid Cache Path
             grid_name = f"data_Alpha{a:.2f}_Gamma{g:.4f}.npz"
             grid_cache_path = os.path.join(grid_data_dir, grid_name)
             
             if generate_600:
                # Combined Task
                # Check scan_videos to skip video gen
                if args.scan_videos or args.scan_grid_data:
                     video_path = None
                     rainbow_path = None
                
                tasks.append((a, g, args.beta, args.steps, args.equilibrium, (not args.fixed_steps), 
                              use_gpu_engine, video_path, rainbow_path, metrics_path, None, None, 
                              args.backend, grid_cache_path, True))
                              
             if args.video_4000:
                # Separate Task (same as standard)
                 if video_4k_path:
                     tasks.append((a, g, args.beta, args.steps, args.equilibrium, (not args.fixed_steps), 
                                  use_gpu_engine, None, None, None, video_4k_path, None, 
                                  args.backend, grid_cache_path, False))
                                      
                    # 4. 4K Rainbow Task
                 if rainbow_4k_path:
                          tasks.append((a, g, args.beta, args.steps, args.equilibrium, (not args.fixed_steps), 
                                       use_gpu_engine, None, None, None, None, rainbow_4k_path, 
                                       args.backend, grid_cache_path, False))
             
    else:
        # Standard Grid Scan
        for i, a in enumerate(alphas):
            for j, g in enumerate(gammas):
                # Construction Video Path
                video_name = f"snowflake_Alpha{a:.2f}_Gamma{g:.4f}.mp4"
                video_path = os.path.join(videos_dir, video_name)
                
                # Grid Cache Path (Optional)
                grid_name = f"data_Alpha{a:.2f}_Gamma{g:.4f}.npz"
                grid_cache_path = os.path.join(grid_data_dir, grid_name)
                
                # Only pass if we want to use it.
                # User asked for it in "resume-from-existing-mp4" mode.
                
                # 4K / Rainbow Versions (Conditional - based on --video-600/--video-4000 flags)
                # Default: --video-600 generates non-4K only.
                # If --video-4000 is set, generate 4K only.
                
                generate_600 = args.video_600 and not args.video_4000
                generate_4000 = args.video_4000
                
                # Standard non-4K videos
                if not generate_600:
                    video_path = None  # Skip standard video if 4K-only mode
                
                if not generate_600:
                    rainbow_path = None
                elif not args.no_rainbow:
                    rainbow_name = f"rainbow_Alpha{a:.2f}_Gamma{g:.4f}.mp4"
                    rainbow_path = os.path.join(videos_dir, rainbow_name)
                
                # 4K videos
                video_4k_path = None
                if generate_4000:
                    video_4k_name = f"snowflake_4k_Alpha{a:.2f}_Gamma{g:.4f}.mp4"
                    video_4k_path = os.path.join(videos_dir, video_4k_name)
                 
                rainbow_4k_path = None
                if generate_4000 and not args.no_rainbow:
                     rainbow_4k_name = f"rainbow_4k_Alpha{a:.2f}_Gamma{g:.4f}.mp4"
                     rainbow_4k_path = os.path.join(videos_dir, rainbow_4k_name)
                
                metrics_name = f"metrics_Alpha{a:.2f}_Gamma{g:.4f}.csv"
                metrics_path = os.path.join(videos_dir, metrics_name)
                
                # Task Tuples - OPTIMIZED FOR --video-600 MODE
                # For non-4K videos (--video-600), we can safely generate snowflake+rainbow in parallel
                # since memory usage is low (~440MB per video).
                # For 4K videos (--video-4000), we still serialize to prevent OOM.
                
                if generate_600:
                    # --video-600 mode: COMBINED task (snowflake + rainbow together)
                    # RESUME LOGIC: If files exist, set paths to None to skip encoding
                    if args.resume:
                        if video_path and os.path.exists(video_path): video_path = None
                        if rainbow_path and os.path.exists(rainbow_path): rainbow_path = None
                    
                if generate_600:
                    # --video-600 mode: COMBINED task (snowflake + rainbow together)
                    # RESUME LOGIC: If files exist, set paths to None to skip encoding
                    if args.resume:
                        if video_path and os.path.exists(video_path): video_path = None
                        if rainbow_path and os.path.exists(rainbow_path): rainbow_path = None
                    
                    # SCAN VIDEOS LOGIC: If detecting from videos, we likely want to SKIP generating the video
                    # and just recover the grid.
                    if args.scan_videos or args.scan_grid_data:
                        video_path = None
                        rainbow_path = None
                        
                    # Both videos share the same simulation run, saving time
                    tasks.append((a, g, args.beta, args.steps, args.equilibrium, (not args.fixed_steps), 
                                  use_gpu_engine, video_path, rainbow_path, metrics_path, None, None, 
                                  args.backend, grid_cache_path, True)) # return_results=True
                
                if generate_4000:
                    # --video-4000 mode: SEPARATE tasks to prevent OOM
                    # 3. 4K Video Task
                    if video_4k_path:
                         tasks.append((a, g, args.beta, args.steps, args.equilibrium, (not args.fixed_steps), 
                                      use_gpu_engine, None, None, None, video_4k_path, None, 
                                      args.backend, grid_cache_path, False))
                                      
                    # 4. 4K Rainbow Task
                    if rainbow_4k_path:
                         tasks.append((a, g, args.beta, args.steps, args.equilibrium, (not args.fixed_steps), 
                                      use_gpu_engine, None, None, None, None, rainbow_4k_path, 
                                      args.backend, grid_cache_path, False))
            
    # Sort tasks by (alpha + gamma) descending - higher values complete faster
    # This prioritizes faster-finishing simulations so you get initial results quickly
    tasks.sort(key=lambda t: (t[0] + t[1]), reverse=True)
    print(f"Tasks sorted by (alpha+gamma) - fastest combinations first.")
    
    # --- 2. EXECUTE & PROCESS SIMULATIONS (STREAMING) ---
    print("Processing Results & Generating Visuals (Streaming Mode)...")
    
    # Parallel Execution Logic:
    # - For --video-600 mode: Allow 2 parallel workers (low memory usage ~440MB per task)
    # - For --video-4000 mode: Force serial to prevent OOM
    # - For CPU mode: Allow parallel with --parallel flag
    # - For SCAN VIDEOS mode: Force serial to prevent Result Queue OOM flood from instant cache hits
    
    generate_600 = args.video_600 and not args.video_4000
    generate_4000 = args.video_4000
    
    generate_600 = args.video_600 and not args.video_4000
    generate_4000 = args.video_4000
    
    if args.scan_videos or args.scan_grid_data:
        # Force Serial for stability during recovery
        print(f"Running {len(tasks)} recovery tasks SERIALLY (prevent cache flood).")
        use_parallel = False
        result_iterator = (worker_wrapper(t) for t in tasks)
    elif generate_600 and use_gpu_engine:
        # --video-600 with GPU: Enable 8 parallel workers (safe memory usage ~440MB x 8 = ~3.5GB)
        use_parallel = True
        workers = 8
        print(f"Running {len(tasks)} simulations in PARALLEL using {workers} workers (--video-600 mode).")
        pool = multiprocessing.Pool(processes=workers)
        result_iterator = pool.imap(worker_wrapper, tasks)
    elif generate_4000 and use_gpu_engine:
        # --video-4000 with GPU: Serial to prevent OOM
        print(f"Running {len(tasks)} simulations SERIALLY (--video-4000 mode, high memory usage).")
        use_parallel = False
        result_iterator = (worker_wrapper(t) for t in tasks)
    elif args.parallel and not use_gpu_engine:
        # CPU mode with --parallel flag
        num_cores = multiprocessing.cpu_count()
        workers = min(8, max(1, num_cores // 2)) 
        print(f"Running {len(tasks)} simulations in PARALLEL using {workers} workers.")
        use_parallel = True
        pool = multiprocessing.Pool(processes=workers)
        result_iterator = pool.imap(worker_wrapper, tasks)
    else:
        # Default: Serial execution
        print(f"Running {len(tasks)} simulations SERIALLY.")
        use_parallel = False
        result_iterator = (worker_wrapper(t) for t in tasks)

    # Stream Processing Loop
    gif_map = {}
    try:
        for idx, result_tuple in enumerate(result_iterator):
            if result_tuple is None:
                # Secondary task finished (Video only)
                continue
                
            (final_grid, freeze_grid, history, a, g) = result_tuple
            # Calculate Grid Coordinates based on a, g
            # alphas is reversed, gammas is reversed.
            # i = index in alphas array
            # j = index in gammas array
            try:
                a_idx = np.where(alphas == a)[0][0]
                g_idx = np.where(gammas == g)[0][0]
                valid_indices = True
            except IndexError:
                # Custom scan might use values not in the grid arrays
                valid_indices = False
                # We cannot stitch into final canvas if we don't know where it goes
                # But we can still generate individual plots
            
            i = a_idx if valid_indices else 0
            j = g_idx if valid_indices else 0
                    
            # 1. VIDEO (Already Handled in Worker)
            t_start = time.time()
            
            print(f"   [Plotting] Generating views for Alpha={a} Gamma={g}...")
            
            # Grid Position (Row = Gamma (Reversed), Col = Alpha (Reversed))
            r_idx = j # Gamma High->Low (Top->Bottom)
            c_idx = i # Alpha High->Low (Matches Label Order Left->Right)
            y_start = r_idx * thumb_h
            x_start = c_idx * thumb_w
            
            # --- OPTIMIZATION: Fast Path for Canvas Stitching ---
            # If high-res image exists, load and resize it instead of re-rendering.
            filename_2d = f"{intermediate_dir}/snowflake_Alpha{a:.2f}_Gamma{g:.4f}_Beta{args.beta}.png"
            loaded_fast = False
            
            if os.path.exists(filename_2d):
                try:
                    # Load existing image (Float 0-1 or Int 0-255)
                    # Matplotlib imread usually returns float 0-1 for PNG
                    img = plt.imread(filename_2d)
                    
                    # Handle RGBA vs RGB
                    if img.shape[2] == 3:
                        # Append Alpha channel (fully opaque)
                        alpha_ch = np.ones((img.shape[0], img.shape[1], 1), dtype=img.dtype)
                        img = np.concatenate([img, alpha_ch], axis=2)
                        
                    # Convert to uint8 0-255 if float
                    if img.dtype == np.float32 or img.dtype == np.float64:
                        img = (img * 255).astype(np.uint8)
                        
                    # Resize to Thumbnail Size
                    # Simple strided slicing might be too aliased for 4K -> 300px
                    # But full interpolation is slow.
                    # Let's use simple nearest neighbor block averaging or striding for speed.
                    # The canvas is just a preview.
                    h_in, w_in, _ = img.shape
                    
                    # Calculate stride
                    stride_h = h_in / thumb_h
                    stride_w = w_in / thumb_w
                    
                    # Integer array indices
                    # We want exactly thumb_h * thumb_w
                    # cv2.resize or PIL.Image.resize is best but we might not have them?
                    # SciPy?
                    # Let's try explicit striding.
                    # indices_y = np.linspace(0, h_in-1, thumb_h).astype(int)
                    # indices_x = np.linspace(0, w_in-1, thumb_w).astype(int)
                    # thumbnail = img[np.ix_(indices_y, indices_x)]
                    
                    # Wait, Matplotlib's 'imread' might read as float. 
                    
                    # Nearest Neighbor Resampling (Fastest & Zero Dependency)
                    # Calculate exact indices to map input to output
                    indices_y = np.linspace(0, h_in-1, thumb_h).astype(int)
                    indices_x = np.linspace(0, w_in-1, thumb_w).astype(int)
                    
                    # Use numpy advanced indexing to resize
                    # ix_ constructs open meshes from multiple sequences
                    thumbnail = img[np.ix_(indices_y, indices_x)]
                    
                    if thumbnail.shape == (thumb_h, thumb_w, 4):
                         final_canvas[y_start:y_start+thumb_h, x_start:x_start+thumb_w, :] = thumbnail
                         loaded_fast = True
                         print(f"      [{time.strftime('%H:%M:%S')}] [Fast Load] {os.path.basename(filename_2d)}")
                         
                except Exception as e:
                    print(f"Error fast loading {filename_2d}: {e}")
                    loaded_fast = False
            
            if not loaded_fast:
                # --- 2. RENDER 2D IMAGE (MASS) SLOW PATH ---
                plt.style.use('dark_background')
                fig_ind, ax_ind = plt.subplots(figsize=(fig_size_inch, fig_size_inch), dpi=thumb_dpi)
                
                # Mass View
                plot_hex_in_ax(ax_ind, final_grid, mode='mass')
                
                # Save High-Res 2D (Disk) - Skip if exists
                if not os.path.exists(filename_2d):
                    fig_ind.savefig(filename_2d, dpi=1000)
                else:
                    print(f"      [Skip Save] {os.path.basename(filename_2d)}")
                
                # Capture for Phase Diagram
                fig_ind.canvas.draw()
                w_fig, h_fig = fig_ind.canvas.get_width_height()
                buf = np.frombuffer(fig_ind.canvas.tostring_argb(), dtype=np.uint8)
                buf = buf.reshape((h_fig, w_fig, 4))
                buf_rgba = np.roll(buf, 3, axis=2) 
                plt.close(fig_ind)
                del fig_ind, ax_ind
                
                thumbnail = buf_rgba[:thumb_h, :thumb_w, :]
                h_t, w_t, _ = thumbnail.shape
                if valid_indices and h_t == thumb_h and w_t == thumb_w:
                    final_canvas[y_start:y_start+h_t, x_start:x_start+w_t, :] = thumbnail
            
            print(f"      [{time.strftime('%H:%M:%S')}] Debug: Before Time Coloring")

            # --- 3. RENDER 2D IMAGE (TIME COLORING) ---
            # Only if we have valid freeze info
            if np.any(freeze_grid > 0):
                fig_time, ax_time = plt.subplots(figsize=(fig_size_inch, fig_size_inch), dpi=thumb_dpi)
                plot_hex_in_ax(ax_time, freeze_grid, mode='time')
                filename_time = f"{intermediate_dir}/snowflake_Alpha{a:.2f}_Gamma{g:.4f}_Beta{args.beta}_Time.png"
                if not os.path.exists(filename_time):
                    fig_time.savefig(filename_time, dpi=1000)
                plt.close(fig_time)
                del fig_time, ax_time
            
            plt.style.use('default') # Reset
            
            # Stitch into Canvas (Mass View)
            # This part is now handled by the fast_load or slow_path logic above.
            # thumbnail = buf_rgba[:thumb_h, :thumb_w, :]
            
            # Grid Position (Row = Gamma (Reversed), Col = Alpha (Reversed))
            # r_idx = j # Gamma High->Low (Top->Bottom)
            # c_idx = i # Alpha High->Low (Matches Label Order Left->Right)
            
            # y_start = r_idx * thumb_h
            # x_start = c_idx * thumb_w

            # x_start = c_idx * thumb_w
            
            # h_t, w_t, _ = thumbnail.shape
            # if valid_indices and h_t == thumb_h and w_t == thumb_w:
            #     final_canvas[y_start:y_start+h_t, x_start:x_start+w_t, :] = thumbnail
            
            print(f"      [{time.strftime('%H:%M:%S')}] Debug: Before 3D Plot")
            
            # --- 4. RENDER 3D SPACE-TIME CUBE ---
            # Using sparse history from worker
            # history is now sparse (max 500 frames)
            if len(history) > 0 and not args.no_3d and spacetime_canvas is not None:
                filename_3d = f"{spacetime_dir}/spacetime_Alpha{a:.2f}_Gamma{g:.4f}_Beta{args.beta}.png"
                skipped_3d_render = False

                # OPTIMIZATION: Fast Load 3D Plot
                if os.path.exists(filename_3d):
                    try:
                        img3d = plt.imread(filename_3d)
                        # Handle RGBA vs RGB
                        if img3d.shape[2] == 3:
                             alpha_ch = np.ones((img3d.shape[0], img3d.shape[1], 1), dtype=img3d.dtype)
                             img3d = np.concatenate([img3d, alpha_ch], axis=2)
                        
                        if img3d.dtype == np.float32 or img3d.dtype == np.float64:
                            img3d = (img3d * 255).astype(np.uint8)
                            
                        h3, w3, _ = img3d.shape
                        
                        # Nearest Neighbor Resampling (Fastest & Zero Dependency)
                        iy = np.linspace(0, h3-1, thumb_h).astype(int)
                        ix = np.linspace(0, w3-1, thumb_w).astype(int)
                        thumb3d = img3d[np.ix_(iy, ix)]
                        
                        if thumb3d.shape == (thumb_h, thumb_w, 4):
                             spacetime_canvas[y_start:y_start+thumb_h, x_start:x_start+thumb_w, :] = thumb3d
                             skipped_3d_render = True
                             print(f"      [{time.strftime('%H:%M:%S')}] [Fast Load 3D] {os.path.basename(filename_3d)}")
                    except Exception as e:
                        print(f"Error fast loading 3D {filename_3d}: {e}")

                if not skipped_3d_render:
                    fig_3d = plt.figure(figsize=(fig_size_inch, fig_size_inch), dpi=thumb_dpi)
                    ax3d = fig_3d.add_subplot(111, projection='3d')
                
                # Pre-calculate time inversion (T=0 at Top)
                    max_time = len(history) # Sparse indices
                    mask = get_visible_surface(history)
                    t_indices, r_indices, c_indices = np.where(mask)
                    
                    x_hex = c_indices + 0.5 * (r_indices % 2)
                    y_hex = r_indices * 0.866
                    z_time = max_time - t_indices 
                    
                    if len(z_time) > 0:
                        ax3d.scatter(x_hex, y_hex, z_time,
                                    c=t_indices, cmap='turbo',
                                    marker='h', s=15, 
                                    alpha=1.0, depthshade=False,
                                    edgecolors='black', linewidths=0.2)
                
                    # Limits
                    mid_x, range_x = np.mean(x_hex), x_hex.max()-x_hex.min() + 1 if len(x_hex)>0 else 10
                    mid_y, range_y = np.mean(y_hex), y_hex.max()-y_hex.min() + 1 if len(y_hex)>0 else 10
                    max_span = max(range_x, range_y, max_time)
                        
                    ax3d.set_xlim(mid_x - max_span/2, mid_x + max_span/2)
                    ax3d.set_ylim(mid_y - max_span/2, mid_y + max_span/2)
                    ax3d.set_zlim(0, max_time)
                    ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Time (Top=0)")
                    ax3d.set_title(f"Alpha={a:.2f} Gamma={g:.4f}")
                    
                    filename_3d = f"{spacetime_dir}/spacetime_Alpha{a:.2f}_Gamma{g:.4f}_Beta{args.beta}.png"
                    if not os.path.exists(filename_3d):
                        fig_3d.savefig(filename_3d, dpi=thumb_dpi)
                    
                    # Capture for SpaceTime Grid
                    fig_3d.canvas.draw()
                    w_3d, h_3d = fig_3d.canvas.get_width_height()
                    buf_3d = np.frombuffer(fig_3d.canvas.tostring_argb(), dtype=np.uint8)
                    buf_3d = buf_3d.reshape((h_3d, w_3d, 4))
                    buf_rgba_3d = np.roll(buf_3d, 3, axis=2)
                    
                    thumbnail_3d = buf_rgba_3d[:thumb_h, :thumb_w, :]
                    if thumbnail_3d.shape == (thumb_h, thumb_w, 4):
                        spacetime_canvas[y_start:y_start+thumb_h, x_start:x_start+thumb_w, :] = thumbnail_3d
                    
                    plt.close(fig_3d)
                    del fig_3d, ax3d, x_hex, y_hex, z_time, mask
                
            # Force cleanup to prevent Matplotlib OOM in long loops
            gc.collect()
                
            # Metrics
            area, perim, ratio, compactness, max_radius, compactness_normalized, branching_factor = calculate_metrics(final_grid)
            
            # New Step Calculation (Max Freeze Time)
            steps_taken = 0
            if np.any(freeze_grid > 0):
                steps_taken = int(np.max(freeze_grid))
            
            # Growth Rate
            growth_rate = area / max(1, steps_taken)
            
            shape_class = "Unknown"
            if area < 50: shape_class = "No Growth"
            elif ratio > 0.8: shape_class = "Plate (Hexagon)"
            elif ratio < 0.4: shape_class = "Dendrite (Star)"
            else: shape_class = "Hybrid / Transition"
            
            results.append({
                "Alpha": a, "Gamma": g, "Area": area, "Perimeter": perim,
                "Ratio": ratio, "Compactness": compactness, "Radius": max_radius,
                "Steps": steps_taken, "GrowthRate": growth_rate,
                "Class": shape_class, "Video": video_name,
                "CompactnessNormalized": compactness_normalized,
                "BranchingFactor": branching_factor
            })
            
            # Map for GIF Index
            if video_name:
                gif_map[(a, g)] = video_name
            
            # Progressive Feedback
            try:
                import resource
                mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                print(f"[{idx+1}/{len(tasks)}] Finished Alpha={a} Gamma={g:.4f} -> {shape_class} (Mem: {mem_usage:.1f} MB)")
            except:
                print(f"[{idx+1}/{len(tasks)}] Finished Alpha={a} Gamma={g:.4f} -> {shape_class}")
                
            # --- MEMORY CLEANUP ---
            # Explicitly delete large arrays to free memory immediately
            del history
            del final_grid
            del freeze_grid
            # Force Garbage Collection (Crucial for serial execution with large objects)
            # Optimize: Only run periodically to avoid slowing down fast scans
            if idx % 10 == 0:
                gc.collect()
            
            # Log Memory (Persistent)
            from .utils import log_debug, check_memory_status
            log_debug(f"Finished Sim {idx+1}/{len(tasks)}: Alpha={a} Gamma={g}")
            
            # Active Safety Check
            # check_memory_status()
            
    finally:
        if use_parallel:
            pool.close()
            pool.join()

    # Save outputs
    # Save outputs
    df = pd.DataFrame(results)
    
    # --- PHYSICAL MAPPING (Nakaya) ---
    print("Calculating Physical Coordinates (Temp/Sigma)...")
    # Apply mapping function row-wise (or vectorized)
    # Vectorized is faster
    # T = -10.0 - (alpha * 4.0)
    # Sigma = gamma * 50.0
    df['Temperature'] = -10.0 - (df['Alpha'] * 4.0)
    df['Supersaturation'] = df['Gamma'] * 50.0
    
    print(f"Scan Complete in {time.time() - start_time:.2f}s")
    df.to_csv(args.output_csv, index=False)
    
    # --- ADVANCED PLOTTING ---
    print("Generating Nakaya Phase Diagram...")
    plot_nakaya_diagram(df, args.output_plot.replace("phase_diagram.png", "nakaya_diagram.png"))
    
    print("Generating HTML Dashboards...")
    videos_dir = os.path.join(os.path.dirname(args.output_csv), "Videos")
    # Dashboard for Standard Videos
    generate_html_dashboard(df, base_output_dir, mode='video')
    # Dashboard for Rainbow Videos
    generate_html_dashboard(df, base_output_dir, mode='rainbow')
    
    # --- PLOT FINAL DIAGRAM (2D) ---
    print("Generating Phase Diagram PDF from Stitched Canvas...")
    fig_final, ax_final = plt.subplots(figsize=(24, 20))
    ax_final.imshow(final_canvas)
    
    x_ticks = np.arange(cols) * thumb_w + thumb_w/2
    y_ticks = np.arange(rows) * thumb_h + thumb_h/2
    ax_final.set_xticks(x_ticks)
    ax_final.set_yticks(y_ticks)
    ax_final.set_xticklabels([f"Alpha={a}" for a in alphas], fontsize=10)
    ax_final.set_yticklabels([f"Gamma={g:.4f}" for g in sorted_gammas], fontsize=10)
    ax_final.set_title(f"Snowflake Morphology Phase Diagram\nBeta={args.beta}", fontsize=20)
    
    plt.savefig(args.output_plot, dpi=100)
    print(f"Saved Plot: {args.output_plot}")
    
    # --- PLOT SPACETIME DIAGRAM (3D Grid) ---
    if not args.no_3d:
        print("Generating SpaceTime GRID...")
        fig_st, ax_st = plt.subplots(figsize=(24, 20))
        ax_st.imshow(spacetime_canvas)
        ax_st.set_xticks(x_ticks)
        ax_st.set_yticks(y_ticks)
        ax_st.set_xticklabels([f"Alpha={a}" for a in alphas], fontsize=10)
        ax_st.set_yticklabels([f"Gamma={g:.4f}" for g in sorted_gammas], fontsize=10)
        ax_st.set_title(f"Space-Time Phase Diagram (3D History)\nBeta={args.beta}", fontsize=20)
        
        st_out = args.output_plot.replace("phase_diagram.png", "spacetime_grid.png")
        plt.savefig(st_out, dpi=100)
        print(f"Saved SpaceTime Grid: {st_out}")
    
    plt.close('all')

    # Write HTML if GIF mode
    if args.gif:
        html_path = os.path.join(os.path.dirname(args.output_plot), "phase_diagram.html")
        write_html_index(html_path, gammas, alphas, gif_map)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution', type=int, default=5)
    parser.add_argument('--alpha_min', type=float, default=0.2)
    parser.add_argument('--alpha_max', type=float, default=1.2)
    parser.add_argument('--gamma_min', type=float, default=0.0001)
    parser.add_argument('--gamma_max', type=float, default=0.01)
    parser.add_argument('--beta', type=float, default=0.4)
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--output_csv', type=str, default="Results/PhaseDiagrams/scan_results.csv")
    parser.add_argument('--output_plot', type=str, default="Results/PhaseDiagrams/phase_diagram.png")
    parser.add_argument('--gif', action='store_true', help="Enable GIF generation (Slower)")
    parser.add_argument('--no_3d', action='store_true', help="Disable 3D SpaceTime plots (Faster)")
    parser.add_argument('--equilibrium', action='store_true', help="Strict Mode: Run until step N is identical to step N-1 (No limit)")
    parser.add_argument('--fixed_steps', action='store_true', help="Disable Stop-At-Edge: Run for exactly --steps count")
    parser.add_argument('--gpu', action='store_true', help="Explicitly enable GPU mode")
    parser.add_argument('--force_cpu', action='store_true', help="Disable GPU auto-detection and force CPU mode")
    parser.add_argument('--parallel', action='store_true', help="Enable CPU Multiprocessing (Faster for Scan)")
    parser.add_argument('--fast_test', action='store_true', help="Run only 4 simulations for testing")
    parser.add_argument('--backend', type=str, default='vispy', 
                        choices=['vispy', 'vispy-cpu', 'vispy-gpu', 'matplotlib'],
                        help="Video backend: 'vispy' (default=cpu), 'vispy-cpu' (fast upscale), 'vispy-gpu' (buffered vram), 'matplotlib' (slow)")
    parser.add_argument('--video-600', action='store_true', dest='video_600', default=True,
                        help="Generate only 600x600 non-4K videos (default)")
    parser.add_argument('--video-4000', action='store_true', dest='video_4000',
                        help="Generate only 4000x4000 4K videos (higher memory usage)")
    parser.add_argument('--no_4k', action='store_true', help="[DEPRECATED] Use --video-600 instead")
    parser.add_argument('--no_rainbow', action='store_true', help="Disable Rainbow video generation (Saves memory)")
    parser.add_argument('--max_memory', type=float, default=0.0, help="Maximum RAM usage in GB (0=Auto/Dynamic based on available RAM)")
    parser.add_argument('--custom_scan', action='store_true', help="Run specific list of missing cells (Recovery Mode)")
    parser.add_argument('--resume', type=str, help="Path to existing run directory to resume from (skips existing videos)")
    parser.add_argument('--scan_videos', type=str, help="Recover from existing Videos directory (Scans MP4s, Regens Grid Data)")
    parser.add_argument('--scan_grid_data', type=str, help="Recover from existing Grid Data (Scans .npz files, Generates Images)")
    # Research Shape Exploration (MAP-Elites)
    parser.add_argument('--research_shapes', action='store_true', help="Enable MAP-Elites shape exploration research mode")
    parser.add_argument('--given_dirs', type=str, help="Comma-separated list of existing result directories to seed from")
    parser.add_argument('--research_output', type=str, default='', help="Output directory name suffix (or auto-generated epoch)")
    parser.add_argument('--research_budget', type=int, default=50, help="Number of new simulations per invocation (default: 50)")
    parser.add_argument('--map_resolution', type=int, default=10, help="MAP-Elites grid resolution (default: 10x10)")
    parser.add_argument('--plan_file', type=str, help="Path to text file with targeted parameter combinations")
    args = parser.parse_args()
    
    # Apply Memory Limit
    from .utils import limit_memory
    limit_val = args.max_memory if args.max_memory > 0 else None
    limit_memory(limit_val)
    
    if args.research_shapes:
        from .research import run_shape_research
        given = args.given_dirs.split(',') if args.given_dirs else []
        run_shape_research(
            given_dirs=given,
            output_dir=args.research_output if args.research_output else None,
            plan_file=args.plan_file,
            budget=args.research_budget,
            map_resolution=args.map_resolution,
            default_steps=args.steps,
            use_gpu=args.gpu,
            force_cpu=args.force_cpu
        )
    else:
        run_scan(args)


if __name__ == "__main__":
    main()

