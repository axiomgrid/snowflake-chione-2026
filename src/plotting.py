import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# --- PHYSICS MAPPING ---
def get_physical_coords(alpha, gamma):
    """
    Maps simulation parameters to physical approximation.
    Alpha -> Temperature (T)
    Gamma -> Supersaturation (Sigma)
    
    Mapping Logic (Heuristic):
    - Alpha [0.1, 2.5] maps to Temp [-10C, -20C] (Plate-Dendrite transition zone)
      T = -10 - (alpha * 4.0)  => Alpha=2.5 -> -20C
    
    - Gamma [0.0001, 0.01] maps to Sigma [0.0, 0.5]
      Sigma = Gamma * 50
    """
    T = -10.0 - (alpha * 4.0)
    sigma = gamma * 50.0
    return T, sigma

def plot_nakaya_diagram(df, output_path):
    """
    Plots the results on a T vs Sigma axis (Nakaya Diagram).
    df: pandas DataFrame with results
    """
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate coords if not present
    if 'Temperature' not in df.columns:
        df['Temperature'] = -10.0 - (df['Alpha'] * 4.0)
    if 'Supersaturation' not in df.columns:
        df['Supersaturation'] = df['Gamma'] * 50.0
        
    # Scatter Plot
    # Shape of marker by Class?
    # Color by Compactness?
    
    # Classes: No Growth, Plate, Dendrite, Hybrid
    markers = {'No Growth': 'x', 'Plate (Hexagon)': 'o', 'Dendrite (Star)': '*', 'Hybrid / Transition': 's', 'Unknown': '.'}
    
    for cls, marker in markers.items():
        subset = df[df['Class'] == cls]
        if len(subset) == 0: continue
        
        sc = ax.scatter(subset['Temperature'], subset['Supersaturation'], 
                   c=subset['Compactness'], cmap='plasma', 
                   s=100, alpha=0.9, edgecolors='white', linewidth=0.5,
                   label=cls, marker=marker)
        
    # Formatting
    ax.set_xlabel("Temperature (°C) [Proxy from Alpha]")
    ax.set_ylabel("Supersaturation (σ) [Proxy from Gamma]")
    ax.set_title("Reconstructed Nakaya Phase Diagram")
    
    # Invert X axis (Nakaya usually has 0 on right, -30 on left)
    ax.set_xlim(-5, -25) 
    ax.invert_xaxis()
    
    # Add Water Saturation Line (Reference)
    # sigma_water(T) ~ 0.12 + 0.002(T+15)^2
    t_range = np.linspace(-25, -5, 100)
    s_water = 0.12 + 0.002 * (t_range + 15)**2
    ax.plot(t_range, s_water, 'c--', alpha=0.5, label='Water Saturation (Ref)')
    
    plt.colorbar(plt.cm.ScalarMappable(cmap='plasma'), label='Compactness (P^2/A)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Generated Nakaya Diagram: {output_path}")

def generate_html_dashboard(df, output_dir, mode='video'):
    """
    Generates an HTML grid of the results.
    mode: 'video' (MP4) or 'gif' (GIF) or 'rainbow' (Rainbow MP4)
    """
    html_path = os.path.join(output_dir, f"dashboard_{mode}.html")
    
    # Get unique Alphas and Gammas
    alphas = sorted(df['Alpha'].unique(), reverse=True) # High -> Low
    gammas = sorted(df['Gamma'].unique(), reverse=True) # High -> Low
    
    # Build Map (Alpha, Gamma) -> FilePath
    file_map = {}
    for _, row in df.iterrows():
        a = row['Alpha']
        g = row['Gamma']
        
        if mode == 'video':
            # filename stored in 'Video' column usually relative or name
            # We assume 'Video' col exists and contains filename 'snowflake_A_G.mp4'
            fname = row.get('Video', '')
            if not fname: continue
            path = f"Videos/{fname}"
        elif mode == 'rainbow':
            fname = row.get('Video', '') # Base name
            if not fname: continue
            # Assume Rainbow video has prefix 'rainbow_'
            path = f"Videos/rainbow_{fname}"
        elif mode == 'gif':
            # Not currently generating GIFs?
            # If we did, presumably in GIFs/ folder
            fname = row.get('Video', '').replace('.mp4', '.gif')
            path = f"GIFs/{fname}"
            
        file_map[(a, g)] = path

    # HTML Generation
    html = []
    html.append("<html><head><style>")
    html.append("body { background-color: #111; color: #eee; font-family: sans-serif; }")
    html.append("table { border-collapse: collapse; margin: 20px auto; }")
    html.append("th, td { border: 1px solid #444; padding: 5px; text-align: center; }")
    html.append(".media { width: 150px; height: 150px; object-fit: contain; }")
    html.append("</style></head><body>")
    html.append(f"<h1 style='text-align:center'>Snowflake Dashboard ({mode})</h1>")
    html.append("<table>")
    
    # Header (Alphas)
    html.append("<tr><th>Gamma \\ Alpha</th>")
    for a in alphas:
        html.append(f"<th>{a}</th>")
    html.append("</tr>")
    
    # Rows (Gammas)
    for g in gammas:
        html.append(f"<tr><th>{g:.4f}</th>")
        for a in alphas:
            path = file_map.get((a, g))
            if path:
                if mode in ['video', 'rainbow']:
                    html.append(f"<td><video class='media' src='{path}' loop autoplay muted playsinline></video><br><small>A={a}<br>G={g:.4f}</small></td>")
                else:
                    html.append(f"<td><img class='media' src='{path}'><br><small>A={a}<br>G={g:.4f}</small></td>")
            else:
                html.append("<td>N/A</td>")
        html.append("</tr>")
    
    html.append("</table></body></html>")
    
    with open(html_path, "w") as f:
        f.write("\n".join(html))
    print(f"Generated Dashboard: {html_path}")
