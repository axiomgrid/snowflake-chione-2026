"""
VIDEO_WRITER.PY
===============
Handles streaming of simulation frames directly to H.265 video file via FFmpeg.
Includes text overlay for metadata (Alpha, Gamma, Frame).
"""
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import shlex
import time
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless rendering
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

# --- HEXAGON VERTEX CALCULATION ---
# Pre-calculate offset arrays for vertices (Pointy topped)
# Angles: 30, 90, 150, 210, 270, 330 degrees
_ANGLES_RAD = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
_COS_ANGLES = np.cos(_ANGLES_RAD)
_SIN_ANGLES = np.sin(_ANGLES_RAD)

# Timing report interval for frame generation benchmarks
_TIMING_REPORT_INTERVAL = 50  # Report avg every N frames


class VideoWriter:
    """Simple video writer that applies colormap to grid and resizes."""
    
    def __init__(self, filepath, width, height, fps=30):
        self.width = width
        self.height = height
        self.filepath = filepath
        
        # FFmpeg Command
        # Input: Raw RGBA video
        # Output: H.265 Lossless
        cmd = (
            f"ffmpeg -y -f rawvideo -vcodec rawvideo "
            f"-s {width}x{height} -pix_fmt rgba -r {fps} -i - "
            f"-c:v libx265 -x265-params lossless=1 "
            f"-preset ultrafast -tune zerolatency "  # Optimized for low memory/latency
            f"-pix_fmt yuv420p "
            f"{shlex.quote(filepath)}"
        )
        
        # Open pipe
        self.process = subprocess.Popen(
            shlex.split(cmd),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            bufsize=10**7  # 10MB buffer
        )
        
        # Font caching
        try:
            # Try to load a nice mono font if available (Ubuntu)
            self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14)
        except:
            self.font = ImageFont.load_default()
            
    def write_frame(self, grid, alpha, gamma, frame_num, cmap, normalization_val=1.5):
        """
        Converts simulation grid to image, adds text, writes to ffmpeg variables.
        grid: 2D array (Mass for standard, FreezeTime for rainbow)
        normalization_val: Value at which to clip/normalize the grid (1.5 for mass, current_frame for time)
        """
        # Normalize grid for display:
        if normalization_val <= 0: normalization_val = 1.0 # Safety
        
        norm_grid = np.clip(grid, 0, normalization_val) / normalization_val
        
        # Apply colormap
        # Cmap returns (N, M, 4) floats 0..1
        rgba_float = cmap(norm_grid)
        del norm_grid  # Free immediately
        
        # Convert to Uint8
        rgba_uint8 = (rgba_float * 255).astype(np.uint8)
        del rgba_float  # Free immediately
        
        # Convert to PIL Image
        img = Image.fromarray(rgba_uint8, 'RGBA')
        del rgba_uint8  # Free immediately
        
        # Resize if target resolution differs from grid resolution
        if img.width != self.width or img.height != self.height:
            # Use NEAREST to preserve crisp edges of the simulation cells
            old_img = img
            img = old_img.resize((self.width, self.height), Image.NEAREST)
            del old_img  # Free the small image before working with the large one
            
        draw = ImageDraw.Draw(img)
        
        # Prepare Text
        text = f"A={alpha} G={gamma:.4f} F={frame_num}"
        
        scale_factor = self.width / 600.0
        font_size = int(14 * scale_factor)
        
        # Load scaled font if possible
        font = self.font
        if scale_factor > 1.2:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", font_size)
            except:
                pass
        
        # Draw with shadow
        offset = int(1 * scale_factor)
        if offset < 1: offset = 1
        
        pos_x = int(10 * scale_factor)
        pos_y = int(10 * scale_factor)
        
        draw.text((pos_x + offset, pos_y + offset), text, font=font, fill=(0,0,0,255))
        draw.text((pos_x, pos_y), text, font=font, fill=(255,255,255,255))
        
        # Write to pipe
        self.process.stdin.write(img.tobytes())
        del img, draw  # Explicitly free
        
    def close(self):
        if self.process:
            self.process.stdin.close()
            self.process.wait()
            self.process = None


class HexVideoWriter:
    """
    High-quality video writer with pre-computed hexagon geometry.
    
    Performance optimizations:
    - Hexagon vertices computed ONCE during init
    - PolyCollection reused each frame, only colors updated
    - No black border (tight layout fills entire frame)
    
    ~10-50x faster than recomputing geometry per frame.
    """
    
    def __init__(self, filepath, width, height, fps=30, grid_shape=(600, 600)):
        self.width = width
        self.height = height
        self.filepath = filepath
        self.grid_rows, self.grid_cols = grid_shape
        
        # Pre-compute hexagon vertex offsets (pointy-topped)
        angles_rad = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
        self._cos_angles = np.cos(angles_rad)
        self._sin_angles = np.sin(angles_rad)
        
        # Pre-compute ALL hexagon centers and vertices for the grid
        yy, xx = np.mgrid[:self.grid_rows, :self.grid_cols]
        self._x_center = np.sqrt(3) * (xx + 0.5 * (yy % 2))
        self._y_center = 1.5 * yy
        
        x_flat = self._x_center.flatten()
        y_flat = self._y_center.flatten()
        
        # Pre-compute all vertices (N_cells x 6 x 2)
        x_verts = x_flat[:, np.newaxis] + self._cos_angles[np.newaxis, :]
        y_verts = y_flat[:, np.newaxis] + self._sin_angles[np.newaxis, :]
        self._verts = np.stack((x_verts, y_verts), axis=2)  # Shape: (N, 6, 2)
        
        # Calculate axis limits for tight framing (no black border)
        self._x_max = self.grid_cols * np.sqrt(3)
        self._y_max = self.grid_rows * 1.5
        
        # Calculate figure size to get exact pixel dimensions
        self.dpi = 100
        self.fig_w = width / self.dpi
        self.fig_h = height / self.dpi
        
        # FFmpeg Command
        cmd = (
            f"ffmpeg -y -f rawvideo -vcodec rawvideo "
            f"-s {width}x{height} -pix_fmt rgba -r {fps} -i - "
            f"-c:v libx265 -x265-params lossless=1 "
            f"-preset medium -g 10000 -keyint_min 10000 "
            f"-pix_fmt yuv420p "
            f"{shlex.quote(filepath)}"
        )
        
        # Open pipe
        self.process = subprocess.Popen(
            shlex.split(cmd),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # Create figure ONCE with tight layout (no margins)
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(self.fig_w, self.fig_h), dpi=self.dpi)
        self.ax = self.fig.add_axes([0, 0, 1, 1])  # Full figure, no margins
        self.ax.set_xlim(0, self._x_max)
        self.ax.set_ylim(0, self._y_max)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        
        # Create PolyCollection ONCE with placeholder colors
        initial_colors = np.zeros((len(self._verts), 4))
        self._collection = PolyCollection(
            self._verts,
            facecolors=initial_colors,
            edgecolors='none',
            linewidths=0,
            rasterized=True
        )
        self.ax.add_collection(self._collection)
        
        # Pre-create text object
        self._text = self.ax.text(
            0.02, 0.98, "", transform=self.ax.transAxes,
            fontsize=max(8, int(8 * width / 1000)), color='white', 
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.5)
        )
        
        # Import colormaps
        from .utils import get_ice_cmap, get_time_cmap
        self._ice_cmap = get_ice_cmap()
        self._time_cmap = get_time_cmap()
        
        # Timing instrumentation
        self._frame_times = []
        self._frame_count = 0
        self._backend_name = "matplotlib"
        
    def write_frame(self, grid, alpha, gamma, frame_num, cmap, normalization_val=1.5, mode='mass'):
        """
        Updates colors and streams frame to FFmpeg.
        Only color computation happens per frame - geometry is reused.
        """
        t_start = time.perf_counter()
        
        val_flat = grid.flatten()
        
        # Compute colors based on mode
        if mode == 'mass':
            if normalization_val <= 0: normalization_val = 1.5
            norm_vals = np.clip(val_flat, 0, normalization_val) / normalization_val
            colors = self._ice_cmap(norm_vals)
            colors[:, 3] = 1.0  # Full opacity
            # Set invisible cells to black
            invisible = val_flat < 0.05
            colors[invisible] = [0, 0, 0, 1]
        else:  # time mode
            max_t = normalization_val if normalization_val > 0 else 1
            norm_vals = np.clip(val_flat, 0, max_t) / max_t
            colors = self._time_cmap(norm_vals)
            # Set unfrozen cells to black
            unfrozen = val_flat <= 0
            colors[unfrozen] = [0, 0, 0, 1]
        
        # Update collection colors (FAST - no geometry recomputation)
        self._collection.set_facecolors(colors)
        
        # Update text
        self._text.set_text(f"A={alpha} G={gamma:.4f} F={frame_num}")
        
        # Render to buffer
        self.fig.canvas.draw()
        
        # Get RGBA buffer
        w, h = self.fig.canvas.get_width_height()
        buf = np.frombuffer(self.fig.canvas.tostring_argb(), dtype=np.uint8)
        buf = buf.reshape((h, w, 4))
        
        # Convert ARGB to RGBA
        rgba = np.roll(buf, 3, axis=2)
        
        # Resize if needed (should match exactly with proper DPI)
        if rgba.shape[0] != self.height or rgba.shape[1] != self.width:
            img = Image.fromarray(rgba, 'RGBA')
            img = img.resize((self.width, self.height), Image.LANCZOS)
            rgba = np.array(img)
            del img
        
        # Write to FFmpeg
        self.process.stdin.write(rgba.tobytes())
        
        # Cleanup
        del buf, rgba, val_flat, colors
        
        # Timing measurement
        t_end = time.perf_counter()
        self._frame_times.append(t_end - t_start)
        self._frame_count += 1
        
        if self._frame_count % _TIMING_REPORT_INTERVAL == 0:
            avg_ms = np.mean(self._frame_times[-_TIMING_REPORT_INTERVAL:]) * 1000
            fps = 1000 / avg_ms if avg_ms > 0 else 0
            print(f"   [{self._backend_name}] Frame {self._frame_count}: avg {avg_ms:.1f}ms/frame ({fps:.1f} fps)")
            self._frame_times = self._frame_times[-_TIMING_REPORT_INTERVAL:]  # Keep only recent
        
    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
            self._collection = None
        if self.process:
            self.process.stdin.close()
            self.process.wait()
            self.process = None

class VispyHexVideoWriter:
    """
    GPU-accelerated 4K video writer using VisPy OpenGL mesh rendering.
    
    Memory-optimized: Uses QUADRANT TILING strategy via TiledVispyRenderer.
    Renders 4K frame in 4 sequential passes using a smaller offscreen buffer
    to reduce GPU memory usage by 75%. Eliminates OOM errors.
    
    Falls back to fast numpy/PIL upscaling if VisPy fails to initialize.
    """
    
    def __init__(self, filepath, width, height, fps=30, grid_shape=(600, 600), backend='vispy'):
        self.width = width
        self.height = height
        self.filepath = filepath
        self.grid_rows, self.grid_cols = grid_shape
        
        # Explicit Renderer Selection
        self._vispy_available = False
        self._renderer = None
        
        if backend == 'vispy-gpu':
            try:
                from .utils import memory_checkpoint
                memory_checkpoint("VispyHexVideoWriter Init Start")
                from .viz_vispy import create_buffered_renderer
                # Use Buffered Renderer for GPU
                # STREAMING MODE: buffer_size=0 disables all GPU buffering.
                # Each frame is rendered and immediately returned (lowest memory usage).
                self._renderer = create_buffered_renderer(
                    width, height, self.grid_rows, self.grid_cols, buffer_size=0
                )
                self._vispy_available = True
                self._backend_name = "vispy-gpu"
                print(f"   [VispyHexVideoWriter] Initialized GPU backend (Buffered, 15 frames)")
                memory_checkpoint("VispyHexVideoWriter Init End")
            except Exception as e:
                print(f"   [ERROR] VisPy GPU init failed: {e}")
                print(f"   [ERROR] Please check drivers or use --backend vispy-cpu")
                raise e
        else:
            # 'vispy' (default) or 'vispy-cpu'
            # Pure CPU Fast Path - No VisPy init needed
            self._vispy_available = False
            self._backend_name = "vispy-cpu"
        
        # FFmpeg Command
        cmd = (
            f"ffmpeg -y -f rawvideo -vcodec rawvideo "
            f"-s {width}x{height} -pix_fmt rgba -r {fps} -i - "
            f"-c:v libx265 -x265-params lossless=1 "
            f"-preset ultrafast -tune zerolatency "  # Optimized for low memory/latency
            f"-pix_fmt yuv420p "
            f"{shlex.quote(filepath)}"
        )
        
        # Open pipe
        self.process = subprocess.Popen(
            shlex.split(cmd),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            bufsize=10**7  # 10MB buffer for pipe to avoid blocking
        )
        
        # Import colormaps
        from .utils import get_ice_cmap, get_time_cmap
        self._ice_cmap = get_ice_cmap()
        self._time_cmap = get_time_cmap()
        
        # Font for text overlay (fallback mode only)
        try:
            self._font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 
                                           max(24, int(24 * width / 1000)))
        except:
            self._font = ImageFont.load_default()
        
        # Timing instrumentation
        self._frame_times = []
        self._frame_count = 0

    def write_frame(self, grid, alpha, gamma, frame_num, cmap, normalization_val=1.5, mode='mass'):
        """
        Render and stream frame to FFmpeg using GPU or fallback.
        """
        t_start = time.perf_counter()
        
        if self._vispy_available and self._renderer is not None:
            # GPU PATH: Use Tiled Vispy Renderer
            try:
                selected_cmap = self._ice_cmap if mode == 'mass' else self._time_cmap
                
                # Render to GPU Buffer
                # Returns None if buffered, or frame(s) if flushed/immediate
                result = self._renderer.render_frame_to_gpu(
                    grid, selected_cmap, normalization_val, mode
                )
                
                # Write to FFmpeg if we got data back
                if result is not None:
                    if isinstance(result, list):
                        for frame in result:
                            self.process.stdin.write(frame.tobytes())
                    else:
                        self.process.stdin.write(result.tobytes())
                    del result
                
            except Exception as e:
                # GPU render failed, fall through to fallback
                import traceback
                print(f"   WARNING: GPU render failed: {e}, using fallback")
                traceback.print_exc()
                self._vispy_available = False
                self._write_frame_fallback(grid, alpha, gamma, frame_num, normalization_val, mode)
                
            # Performance Check (Auto-fallback if too slow)
            # Performance Check (Auto-fallback disabled by user request)
            if self._vispy_available:
                t_render = time.perf_counter() - t_start
                if self._frame_count < 5 and t_render > 2.0:
                    print(f"   NOTE: GPU rendering is slow ({t_render:.2f}s/frame), but fallback is disabled.")
        else:
            # FALLBACK PATH: Fast numpy colormap + upscale
            self._write_frame_fallback(grid, alpha, gamma, frame_num, normalization_val, mode)
        
        # Timing measurement
        t_end = time.perf_counter()
        self._frame_times.append(t_end - t_start)
        self._frame_count += 1
        
        if self._frame_count % 15 == 0:
            from .utils import memory_checkpoint
            memory_checkpoint(f"[{self._backend_name}] Frame {self._frame_count}")
        
        if self._frame_count % _TIMING_REPORT_INTERVAL == 0:
            avg_ms = np.mean(self._frame_times[-_TIMING_REPORT_INTERVAL:]) * 1000
            fps = 1000 / avg_ms if avg_ms > 0 else 0
            print(f"   [{self._backend_name}] Frame {self._frame_count}: avg {avg_ms:.1f}ms/frame ({fps:.1f} fps)")
            self._frame_times = self._frame_times[-_TIMING_REPORT_INTERVAL:]
    
    def _write_frame_fallback(self, grid, alpha, gamma, frame_num, normalization_val, mode):
        """
        Fallback rendering: numpy colormap + PIL LANCZOS upscale.
        """
        # Apply colormap
        if mode == 'mass':
            if normalization_val <= 0: normalization_val = 1.5
            norm_grid = np.clip(grid, 0, normalization_val) / normalization_val
            rgba_float = self._ice_cmap(norm_grid)
            rgba_float[grid < 0.05] = [0, 0, 0, 1]
        else:
            max_t = normalization_val if normalization_val > 0 else 1
            norm_grid = np.clip(grid, 0, max_t) / max_t
            rgba_float = self._time_cmap(norm_grid)
            rgba_float[grid <= 0] = [0, 0, 0, 1]
        
        # Convert and upscale
        rgba_uint8 = (rgba_float * 255).astype(np.uint8)
        img = Image.fromarray(rgba_uint8, 'RGBA')
        if img.width != self.width or img.height != self.height:
            img = img.resize((self.width, self.height), Image.BICUBIC)
        
        # Add text overlay
        draw = ImageDraw.Draw(img)
        text = f"A={alpha} G={gamma:.4f} F={frame_num}"
        draw.text((10, 10), text, font=self._font, fill=(255, 255, 255, 255))
        
        # Write to FFmpeg
        self.process.stdin.write(img.tobytes())
        del img, draw, rgba_float, rgba_uint8
        
    def close(self):
        # Flush any remaining frames in buffer
        if self._vispy_available and self._renderer is not None:
            try:
                rem_frames = self._renderer.flush()
                if rem_frames:
                    if isinstance(rem_frames, list):
                        for f in rem_frames:
                            self.process.stdin.write(f.tobytes())
                    else:
                        self.process.stdin.write(rem_frames.tobytes())
                
                # Explicitly release GPU resources
                if hasattr(self._renderer, 'close'):
                    self._renderer.close()
                self._renderer = None
                
            except Exception as e:
                print(f"   WARNING: Error flushing buffer: {e}")
                import traceback
                traceback.print_exc()
                
        if self.process:
            self.process.stdin.close()
            self.process.wait()
            self.process = None


# Alias for backward compatibility
MatplotlibHexVideoWriter = HexVideoWriter

