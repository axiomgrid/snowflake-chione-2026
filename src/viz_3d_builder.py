import os
import sys
import csv
import json
import math
import argparse
import base64
import numpy as np

# Lazy Import Engine for generation
try:
    sys.path.append(os.getcwd())
    from src.engine import run_single_sim
    from src.viz import plot_hex_in_ax
    import matplotlib.pyplot as plt
    HAS_ENGINE = True
except ImportError as e:
    print(f"Warning: Could not import engine ({e}). Auto-generation disabled.")
    HAS_ENGINE = False

def ensure_images(row, path_mass, path_time, steps_taken):
    """Generates both Mass and Time/Rainbow images if either is missing."""
    if not HAS_ENGINE:
        return False
        
    # Check if we need to generate anything
    need_mass = not os.path.exists(path_mass)
    need_time = not os.path.exists(path_time)
    
    if not (need_mass or need_time):
        return True # Both exist

    try:
        alpha = float(row['Alpha'])
        gamma = float(row['Gamma'])
        beta = float(row['Beta'])
        
        print(f"  Generating missing images (Mass={need_mass}, Time={need_time}): Steps={steps_taken}")
        
        # Run Simulation (Exact deterministic replay)
        final_grid, freeze_grid, history = run_single_sim(
            alpha, gamma, beta, 
            steps=steps_taken, 
            stop_at_edge=False, 
            return_history=False
        )
        
        # Generate Mass Image
        if need_mass:
            fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
            plot_hex_in_ax(ax, final_grid, mode='mass')
            plt.axis('off') # Ensure clean look? plot_hex_in_ax handles axes?
            # Actually plot_hex_in_ax doesn't hide axes by default?
            # cli.py used dark_background style which hides axes?
            # No, allow default axes or hide them. cli.py doesn't hide axes explicitly.
            # But let's keep consistent with existing style.
            # Usually we want NO axes for sprites.
            ax.axis('off')
            fig.savefig(path_mass, dpi=300, transparent=True)
            plt.close(fig)

        # Generate Time Image
        if need_time:
            fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
            plot_hex_in_ax(ax, freeze_grid, mode='time')
            ax.axis('off')
            fig.savefig(path_time, dpi=300, transparent=True)
            plt.close(fig)
            
        return True
    except Exception as e:
        print(f"  Error generating images: {e}")
        return False

def create_viz_3d(scan_results_path, output_html_path):
    print(f"Reading scan results from: {scan_results_path}")
    
    data_points = []
    
    if not os.path.exists(scan_results_path):
        print(f"Error: scan_results.csv not found at {scan_results_path}")
        return

    with open(scan_results_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # Parse metrics
                alpha = float(row['Alpha'])
                gamma = float(row['Gamma'])
                beta = float(row['Beta'])
                
                # HTML is in {DIR}/research_shapes/viz_3d.html
                # Images are in {DIR}/research_shapes/Intermediate/
                
                # Construct expected filename
                target_steps = str(row.get('TargetSteps', '')).strip()
                steps_suffix = f"_Steps{target_steps}" if target_steps and target_steps != 'None' else ""
                
                # Check formatting carefully
                try:
                    f_alpha = f"{float(alpha):.4f}"
                    f_gamma = f"{float(gamma):.6f}"
                    f_beta = f"{float(beta):.4f}"
                except:
                    print(f"Warning: Could not format params for row: {row}")
                    continue

                # Define Paths
                filename_mass = f"snowflake_Alpha{f_alpha}_Gamma{f_gamma}_Beta{f_beta}{steps_suffix}.png"
                path_mass = os.path.join(os.path.dirname(scan_results_path), "research_shapes", "Intermediate", filename_mass)
                
                filename_time = f"snowflake_Alpha{f_alpha}_Gamma{f_gamma}_Beta{f_beta}_Time.png"
                path_time = os.path.join(os.path.dirname(scan_results_path), "research_shapes", "Intermediate", filename_time)
                
                # Check/Generate
                try:
                    steps_taken = int(float(row.get('Steps', 0)))
                    if steps_taken > 0:
                        ensure_images(row, path_mass, path_time, steps_taken)
                except: pass

                # EMBED AS BASE64 (Normal)
                img_src = None
                img_abs_path = path_mass
                if not os.path.exists(img_abs_path):
                    # Try fallback (no steps suffix)
                    filename_no_steps = f"snowflake_Alpha{f_alpha}_Gamma{f_gamma}_Beta{f_beta}.png"
                    path_no_steps = os.path.join(os.path.dirname(scan_results_path), "research_shapes", "Intermediate", filename_no_steps)
                    if os.path.exists(path_no_steps):
                        img_abs_path = path_no_steps
                    else:
                        print(f"Warning: Image not found: {img_abs_path}")
                        continue 

                try:
                    with open(img_abs_path, "rb") as image_file:
                        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                        img_src = f"data:image/png;base64,{encoded_string}"
                except Exception as e:
                    print(f"Error reading image {img_abs_path}: {e}")
                    continue

                # EMBED AS BASE64 (Time/Rainbow) - Optional
                img_src_time = None
                # path_time already defined above

                if os.path.exists(path_time):
                    try:
                        with open(path_time, "rb") as image_file:
                            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                            img_src_time = f"data:image/png;base64,{encoded_string}"
                    except Exception as e:
                        print(f"Error reading time image {path_time}: {e}")
                else:
                    # Try with Steps suffix just in case
                    filename_time_steps = f"snowflake_Alpha{f_alpha}_Gamma{f_gamma}_Beta{f_beta}{steps_suffix}_Time.png"
                    path_time_steps = os.path.join(os.path.dirname(scan_results_path), "research_shapes", "Intermediate", filename_time_steps)
                    if os.path.exists(path_time_steps):
                         try:
                            with open(path_time_steps, "rb") as image_file:
                                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                                img_src_time = f"data:image/png;base64,{encoded_string}"
                         except: pass

                # Prepare full metadata for tooltip
                metadata = {}
                for k, v in row.items():
                    metadata[k] = v
                metadata['Filename'] = filename_mass

                data_points.append({
                    'x': alpha,          # Alpha
                    'y': gamma,          # Gamma
                    'z': beta,           # Beta
                    'img': img_src,      # Data URI (Normal)
                    'imgT': img_src_time,# Data URI (Time/Rainbow)
                    'metadata': metadata # Full CSV data
                })
            except ValueError:
                continue

    print(f"Found {len(data_points)} valid data points with images.")

    # 2. Generate HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Snowflake Shape Explorer 3D</title>
    <style>
        body {{ margin: 0; overflow: hidden; background-color: #111; color: #fff; font-family: sans-serif; }}
        #info {{ position: absolute; top: 10px; width: 100%; text-align: center; z-index: 100; pointer-events: none; text-shadow: 1px 1px 2px black; }}
        #controls {{ position: absolute; top: 40px; right: 10px; z-index: 100; background: rgba(0,0,0,0.5); padding: 10px; border-radius: 5px; }}
        #tooltip {{ 
            position: absolute; 
            background: rgba(0, 0, 0, 0.9); 
            color: #eee; 
            padding: 8px 12px; 
            border-radius: 4px; 
            pointer-events: none; 
            display: none; 
            z-index: 200;
            font-family: monospace;
            font-size: 11px;
            white-space: pre;
            border: 1px solid #555;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.5);
            max-width: 300px;
        }}
        label {{ display: block; margin-bottom: 5px; cursor: pointer; }}
        input[type=range] {{ width: 200px; }}
        input[type=checkbox] {{ margin-right: 5px; }}
    </style>
    <!-- Three.js from CDN -->
    <script type="importmap">
      {{
        "imports": {{
          "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
          "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
        }}
      }}
    </script>
</head>
<body>
    <div id="info">
        <h2>Snowflake Shape Morphospace</h2>
        <p>X: Alpha (Diffusion) | Y: Gamma (Growth) | Z: Beta (Background)</p>
    </div>
    <div id="controls">
        <label for="sizeSlider">Image Size</label>
        <input type="range" id="sizeSlider" min="0.1" max="5.0" step="0.1" value="1.0">
        <hr style="border: 0; border-top: 1px solid #555; margin: 10px 0;">
        <label for="colorToggle">
            <input type="checkbox" id="colorToggle"> Show Rainbow Time History
        </label>
    </div>
    <div id="tooltip"></div>
    
    <script type="module">
        import * as THREE from 'three';
        import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

        // DATA INJECTION
        const data = {json.dumps(data_points)};

        // SCENE SETUP
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x111111);
        scene.fog = new THREE.Fog(0x111111, 2, 20);

        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        camera.position.set(2, 2, 3); // Better initial view

        const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.body.appendChild(renderer.domElement);

        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;

        // AXES SCALING
        const SCALE_X = 1.0;   // Alpha (0-3 -> 0-3)
        const SCALE_Y = 100.0; // Gamma (0-0.03 -> 0-3)
        const SCALE_Z = 3.0;   // Beta (0-1 -> 0-3)
        
        // Axis Helpers
        const axesHelper = new THREE.AxesHelper(3.5);
        scene.add(axesHelper);

        const gridHelper = new THREE.GridHelper(6, 10, 0x444444, 0x222222);
        // Grid is on XZ plane by default. We want it on bottom?
        // Let's keep it centered for now.
        scene.add(gridHelper);

        // LABELS
        function createLabel(text, x, y, z, color='white', size=0.5) {{
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            canvas.width = 256; 
            canvas.height = 128;
            ctx.fillStyle = color;
            ctx.font = 'bold 40px sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(text, 128, 64);
            
            const texture = new THREE.CanvasTexture(canvas);
            const material = new THREE.SpriteMaterial({{ map: texture, transparent: true }});
            const sprite = new THREE.Sprite(material);
            sprite.position.set(x, y, z);
            sprite.scale.set(2 * size, 1 * size, 1);
            return sprite;
        }}

        // Add Axis Labels
        scene.add(createLabel("Alpha (X)", 3.8, 0, 0, '#ff5555'));
        scene.add(createLabel("Gamma (Y)", 0, 3.8, 0, '#55ff55'));
        scene.add(createLabel("Beta (Z)", 0, 0, 3.8, '#5555ff'));

        // CREATE SPRITES (No lines)
        const textureLoader = new THREE.TextureLoader();
        const sprites = [];
        console.log("Loading " + data.length + " sprites...");
        
        data.forEach((p, index) => {{
            // LOAD TEXTURES (Normal & Time)
            const texNormal = textureLoader.load(p.img);
            let texTime = null;
            if (p.imgT) {{
                texTime = textureLoader.load(p.imgT);
            }}

            // Start with Normal
            // const material = new THREE.SpriteMaterial({{ map: texture, transparent: true, depthTest: false }});
            const material = new THREE.SpriteMaterial({{ map: texNormal, transparent: true }});
            const sprite = new THREE.Sprite(material);
            
            // Position
            const px = p.x * SCALE_X;
            const py = p.y * SCALE_Y;
            const pz = p.z * SCALE_Z;
            
            sprite.position.set(px, py, pz);
            
            // Base Size
            sprite.scale.set(0.5, 0.5, 1); 
            // Store metadata & textures for raycaster/toggle
            sprite.userData = {{ 
                metadata: p.metadata,
                texNormal: texNormal,
                texTime: texTime
            }};
            
            scene.add(sprite);
            sprites.push(sprite);
                
                // --- PROJECTION LINES (Solid, Floor Drop Only) ---
                const materialLine = new THREE.LineBasicMaterial({{
                    color: 0x888888,
                    linewidth: 1,
                    opacity: 0.6,
                    transparent: true
                }});
                
                // 1. Vertical Drop to Floor (XZ plane at y=0)
                const lineDrop = new THREE.Line(new THREE.BufferGeometry().setFromPoints([
                    new THREE.Vector3(px, py, pz),
                    new THREE.Vector3(px, 0, pz)
                ]), materialLine);
                lineDrop.visible = false;
                scene.add(lineDrop);

                // 2. On Floor: To X-axis (along Z)
                const lineToX = new THREE.Line(new THREE.BufferGeometry().setFromPoints([
                    new THREE.Vector3(px, 0, pz),
                    new THREE.Vector3(px, 0, 0)
                ]), materialLine);
                lineToX.visible = false;
                scene.add(lineToX);

                // 3. On Floor: To Z-axis (along X)
                const lineToZ = new THREE.Line(new THREE.BufferGeometry().setFromPoints([
                    new THREE.Vector3(px, 0, pz),
                    new THREE.Vector3(0, 0, pz)
                ]), materialLine);
                lineToZ.visible = false;
                scene.add(lineToZ);

            // Store lines in userData to toggle
            sprite.userData.lines = [lineDrop, lineToX, lineToZ];

        }});

        // SLIDER CONTROL
        const slider = document.getElementById('sizeSlider');
        slider.addEventListener('input', (e) => {{
            const scale = parseFloat(e.target.value);
            sprites.forEach(s => {{
                s.scale.set(0.5 * scale, 0.5 * scale, 1);
            }});
        }});
        
        // COLOR TOGGLE CONTROL
        const colorToggle = document.getElementById('colorToggle');
        colorToggle.addEventListener('change', (e) => {{
            const showTime = e.target.checked;
            sprites.forEach(s => {{
                if (showTime && s.userData.texTime) {{
                    s.material.map = s.userData.texTime;
                }} else {{
                    s.material.map = s.userData.texNormal;
                }}
                s.material.needsUpdate = true;
            }});
        }});
        
        // INTERACTION (TOOLTIP)
        const raycaster = new THREE.Raycaster();
        const mouse = new THREE.Vector2();
        const tooltip = document.getElementById('tooltip');
        let hoveredSprite = null;
        
        function onMouseMove( event ) {{
            mouse.x = ( event.clientX / window.innerWidth ) * 2 - 1;
            mouse.y = - ( event.clientY / window.innerHeight ) * 2 + 1;
            
            tooltip.style.left = (event.clientX + 15) + 'px';
            tooltip.style.top = (event.clientY + 15) + 'px';
            
            checkIntersection();
        }}
        
        function checkIntersection() {{
            raycaster.setFromCamera( mouse, camera );
            const intersects = raycaster.intersectObjects( sprites );
            
            if ( intersects.length > 0 ) {{
                const obj = intersects[0].object;
                
                // If new hover
                if (hoveredSprite !== obj) {{
                    // Reset previous lines
                    if (hoveredSprite && hoveredSprite.userData.lines) {{
                        hoveredSprite.userData.lines.forEach(l => l.visible = false);
                    }}
                    hoveredSprite = obj;
                    
                    // Show current lines
                    if (hoveredSprite.userData.lines) {{
                        hoveredSprite.userData.lines.forEach(l => l.visible = true);
                    }}
                    
                    // Ghosting: Dim others
                    sprites.forEach(s => {{
                        s.material.opacity = (s === hoveredSprite) ? 1.0 : 0.2;
                    }});
                }}

                if (obj.userData && obj.userData.metadata) {{
                    tooltip.style.display = 'block';
                    let text = ``;
                    const meta = obj.userData.metadata;
                    for (const [key, value] of Object.entries(meta)) {{
                       text += `${{key}}: ${{value}}\\n`;
                    }}
                    tooltip.innerText = text;
                }}
            }} else {{
                tooltip.style.display = 'none';
                
                // Clear lines & Reset Opacity if we mouse off
                if (hoveredSprite) {{
                    if (hoveredSprite.userData.lines) {{
                        hoveredSprite.userData.lines.forEach(l => l.visible = false);
                    }}
                    // Reset Opacity
                    sprites.forEach(s => {{
                        s.material.opacity = 1.0;
                    }});
                    
                    hoveredSprite = null;
                }}
            }}
        }}

        window.addEventListener( 'mousemove', onMouseMove, false );

        // LIGHTS (Not strictly needed for sprites which are unlit, but good for context)
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        scene.add(ambientLight);

        // RENDER LOOP
        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }}
        animate();

        // RESIZE HANDLER
        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});
    </script>
</body>
</html>
    """

    with open(output_html_path, 'w') as f:
        f.write(html_content)
    print(f"Generated 3D Viz: {output_html_path} ({len(data_points)} shapes)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', required=True, help="Directory containing scan_results.csv")
    args = parser.parse_args()
    
    scan_csv = os.path.join(args.results_dir, "scan_results.csv")
    output_html = os.path.join(args.results_dir, "research_shapes", "viz_3d.html")
    
    create_viz_3d(scan_csv, output_html)
