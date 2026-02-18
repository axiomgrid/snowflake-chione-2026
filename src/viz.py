"""
VIZ.PY — Visualization Tools
============================
Plotting, GIF generation, and HTML dashboards.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import PolyCollection
from .utils import get_ice_cmap

# --- HEXAGON VERTEX CALCULATION ---
# Pre-calculate offset arrays for vertices (Pointy topped)
# Angles: 30, 90, 150, 210, 270, 330 degrees
_ANGLES_RAD = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
_COS_ANGLES = np.cos(_ANGLES_RAD)
_SIN_ANGLES = np.sin(_ANGLES_RAD)

def plot_hex_in_ax(ax, data, mode='mass', title=""):
    rows, cols = data.shape
    yy, xx = np.mgrid[:rows, :cols]
    
    x_center = np.sqrt(3) * (xx + 0.5 * (yy % 2))
    y_center = 1.5 * yy
    
    x_flat = x_center.flatten()
    y_flat = y_center.flatten()
    val_flat = data.flatten()
    
    colors = None
    mask_visible = None
    
    if mode == 'mass':
        cmap = get_ice_cmap()
        norm_vals = np.clip(val_flat, 0, 1.5) / 1.5
        colors = cmap(norm_vals)
        # User Request: Don't use transparency. Keep colors visible.
        colors[:, 3] = 1.0
        mask_visible = val_flat > 0.05
        
    elif mode == 'time':
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap('turbo')
        max_t = val_flat.max()
        if max_t == 0: max_t = 1
        norm_vals = val_flat / max_t
        colors = cmap(norm_vals)
        
        # Only visible if frozen (time > 0)
        mask_visible = val_flat > 0
        alphas = np.zeros(len(val_flat))
        alphas[mask_visible] = 1.0
        colors[:, 3] = alphas

    if not np.any(mask_visible): return

    xv = x_flat[mask_visible]
    yv = y_flat[mask_visible]
    cv = colors[mask_visible]
    
    x_verts = xv[:, np.newaxis] + _COS_ANGLES[np.newaxis, :]
    y_verts = yv[:, np.newaxis] + _SIN_ANGLES[np.newaxis, :]
    
    verts = np.stack((x_verts, y_verts), axis=2)

    collection = PolyCollection(
        verts,
        facecolors=cv,
        edgecolors='none', 
        linewidths=0,
        rasterized=True
    )
    ax.add_collection(collection)
    
    rows_grid, cols_grid = data.shape
    x_max = cols_grid * np.sqrt(3)
    y_max = rows_grid * 1.5
    ax.set_xlim(-5, x_max + 5)
    ax.set_ylim(-5, y_max + 5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, color='white', fontsize=10)

def save_gif_from_history(history, filename):
    # History is already cropped to visualization size (e.g. 401x401)
    frames, rows, cols = history.shape
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis('off')
    
    # Calculate Hex Centers for the cropped grid
    yy, xx = np.mgrid[:rows, :cols]
    x_center = np.sqrt(3) * (xx + 0.5 * (yy % 2))
    y_center = 1.5 * yy
    
    x_flat = x_center.flatten()
    y_flat = y_center.flatten()
    
    # Pre-calculate Vertices
    x_verts = x_flat[:, np.newaxis] + _COS_ANGLES[np.newaxis, :]
    y_verts = y_flat[:, np.newaxis] + _SIN_ANGLES[np.newaxis, :]
    verts = np.stack((x_verts, y_verts), axis=2)

    # Initialize Collection with zero colors
    colors = np.zeros((rows*cols, 4))
    
    collection = PolyCollection(
        verts,
        facecolors=colors, 
        edgecolors='none', 
        linewidths=0
    )
    ax.add_collection(collection)
    
    # Set Limits
    x_max = cols * np.sqrt(3)
    y_max = rows * 1.5
    ax.set_xlim(-10, x_max + 10)
    ax.set_ylim(-10, y_max + 10)
    ax.set_aspect('equal')
    
    cmap = get_ice_cmap()
    
    def update(frame_idx):
        grid = history[frame_idx]
        flat = grid.flatten()
        norm_vals = np.clip(flat, 0, 1.5) / 1.5
        new_colors = cmap(norm_vals)
        new_colors[:, 3] = np.clip(flat, 0.0, 1.0)
        collection.set_facecolors(new_colors)
        return [collection]

    # Create Animation
    # loop=1 means play once and stop.
    anim = animation.FuncAnimation(fig, update, frames=frames, blit=True)
    try:
        anim.save(filename, writer='pillow', fps=30, loop=1)
    except TypeError:
        # Fallback if matplotlib/pillow version is old
        anim.save(filename, writer='pillow', fps=30)
    plt.close(fig)
    plt.style.use('default')

def write_html_index(filepath, gammas, alphas, map_data):
    """
    Generates an HTML grid displaying the GIFs.
    map_data: Dict mapping keys (gamma_idx, alpha_idx) -> relative_gif_path
    """
    html = []
    html.append("<html><head><style>")
    html.append("body { background-color: #111; color: #eee; font-family: sans-serif; }")
    html.append("table { border-collapse: collapse; margin: 20px auto; }")
    html.append("th, td { border: 1px solid #444; padding: 5px; text-align: center; }")
    html.append("img { width: 150px; height: 150px; object-fit: contain; }")
    html.append(".controls { text-align: center; margin: 20px; }")
    html.append("button { padding: 10px 20px; background: #222; color: #fff; border: 1px solid #555; cursor: pointer; }")
    html.append("button:hover { background: #333; }")
    html.append("</style>")
    html.append("<script>")
    html.append("function syncGifs() {")
    html.append("  var images = document.getElementsByTagName('img');")
    html.append("  for (var i = 0; i < images.length; i++) {")
    html.append("    var src = images[i].src;")
    html.append("    images[i].src = '';")
    html.append("    images[i].src = src;")
    html.append("  }")
    html.append("}")
    html.append("window.onload = function() { setTimeout(syncGifs, 500); };")
    html.append("</script>")
    html.append("</head><body>")
    html.append("<h1 style='text-align:center'>Snowflake Phase Diagram (GIF Grid)</h1>")
    html.append("<div class='controls'><button onclick='syncGifs()'>Synchronize Animations</button></div>")
    html.append("<table>")
    
    # Header Row (Alphas)
    html.append("<tr><th>Gamma \\ Alpha</th>")
    for a in alphas:
        html.append(f"<th>{a}</th>")
    html.append("</tr>")
    
    # Rows (Gammas) - Reversed to match plot usually (High gamma at top)
    for g_idx, g in enumerate(reversed(gammas)):
        real_g_idx = len(gammas) - 1 - g_idx
        html.append(f"<tr><th>{g:.4f}</th>")
        for a_idx, a in enumerate(alphas):
            key = (real_g_idx, a_idx)
            if key in map_data:
                gif_path = map_data[key]
                html.append(f"<td><img src='{gif_path}' loading='lazy'><br><small>Alpha={a}<br>Gamma={g:.4f}</small></td>")
            else:
                html.append("<td>N/A</td>")
        html.append("</tr>")
        
    html.append("</table></body></html>")
    
    with open(filepath, "w") as f:
        f.write("\n".join(html))
    print(f"Generated HTML Dashboard: {filepath}")
