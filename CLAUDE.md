# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

"Morpho" — a static WebGL art site where thousands of GPU-simulated butterflies flock and then settle into a portrait made of light. No build step, no framework, no package.json. The site lives at the repo root (`index.html` + `face-data.js`); `assets/` holds the dot-portrait reference images and research PDF. Published at https://github.com/SibasisRath/morpho with GitHub Pages serving the root.

**Intent.** The owner's vision: a dreamlike animated site that opens like the three.js GPGPU birds demo but with colorful butterflies, which then settle into a target face rendered as a dot-pattern portrait (see the dot-face reference JPEGs in `assets/`). It must stay browser- and device-friendly. Beyond the artifact itself, this is a study project — the owner wants to read and understand the techniques (GPGPU flocking, point-cloud portraits) to explore more 3D designs in this style. Favor readable, well-commented code over clever compression. `assets/` also contains Erra et al. 2004 (*Massive Simulation using GPU of a Distributed Behavioral Model of a Flock*), the research basis for texture-based GPU boids.

## Commands

Run the site (from the repo root):

```
python3 -m http.server 8000    # then open http://localhost:8000
```

Opening `index.html` directly in a browser also works. three.js and fonts load from CDNs, so network access is needed.

Regenerate the point cloud from a new portrait photo:

```
pip install numpy opencv-python-headless
python3 make_face_data.py your_portrait.jpg --points 16000 --out face-data.js
```

This also writes `face-preview.png` (dot/depth preview) for checking the mask before opening the site. `face-data.orig.js` is a backup of the original full-density (non-stippled) cloud; `restipple.js` (pure Node, no deps) re-weights it — MODE=tone is the approved halftone look. Optional flags: `--size 1024` (working resolution, helps if the mask misses hair), `--depth depth.png` (external monocular depth map for true relief), `--depth-scale`, `--seed`.

Useful URL switches while tuning: `?n=96` (compute texture side → n² butterflies, 16–256), `?bloom=0`, `?stats=1` (fps/count/settle readout). Press **R** or click Replay to relaunch the swarm. From the console, `SWARM.velU.uSettle.value = 0.5` freezes the morph mid-blend.

There are no tests or linters.

## Architecture

Two pieces: a Python baking tool and a single-file site.

**`make_face_data.py` (offline, photo → point cloud).** GrabCut isolates the head, a synthetic depth relief is built (silhouette-distance dome + brightness detail — or an external depth map via `--depth`), points are scattered with a jittered grid in stipple-portrait style — density comes only from edge strength and shadow tone, gated to zero on lit skin so the face reads through negative space, with a thin rim for the head contour — colors are sampled and saturation-lifted, then everything is quantized (xyz as uint16, rgb as uint8), base64-encoded, and written as `window.FACE_DATA` in `face-data.js`. Points are shuffled so any prefix is a uniform sub-sample — the site picks how many to use per device tier. Output space: head height ≈ 2.0 units, centered at origin, z in roughly [-0.3, 0.3], +z toward camera.

**`index.html` (the entire site, ~720 lines, all shaders inline).** Read it top to bottom; it is numbered in comment blocks 0–7:

- **0. Device tier** — picks butterfly count (72²–128²), bloom on/off, and pixel-ratio cap from cores/mobile detection (see the table in `README.md`).
- **1. Face data** — unpacks `FACE_DATA` into GPU textures (home positions + colors).
- **2. Renderer/scene/camera.**
- **3. GPGPU simulation** — the core. Each butterfly is one texel in two ping-ponged textures, *position* (xyz + wing phase) and *velocity*, advanced entirely by fragment shaders via `GPUComputationRenderer`; the CPU never touches a butterfly. The velocity shader blends two accelerations with a single uniform `uSettle` (0 = free flock, 1 = hold the face): **flock** (curl noise wind, a 12-sample pseudo-boids approximation, wandering attractor, soft boundary) vs **settle** (damped spring to each home position, k=9 damping=5, slightly under-damped; no idle force at rest, so the settled face is a true standstill — wing flap and camera sway also stop, and only the pointer disturbs it). The pointer adds a third force — radial push with a small lift toward the camera; the spring makes return-to-shape automatic.
- **4. Rendering** — one `InstancedBufferGeometry` (two wing quads × COUNT instances), each instance carrying a `reference` uv into the simulation textures. Vertex shader handles orientation (along velocity in flight, camera-facing at rest), wing flap from the phase in `position.w`, and LOD (past `uLodNear`, wings cross-fade to a billboard glow). Color: pastel rainbow in flight, the portrait's pixel color at rest.
- **5. Post** — desktop gets `UnrealBloomPass`; mobile tone-maps in the material (`DIRECT_OUTPUT`) so both paths match.
- **6. Pointer → world position** on the face plane.
- **7. Loop/UI.**

Wing size is derived from the gap between dots, so the face stays legible at any butterfly count. Simplex noise in the shaders is Ian McEwan / Ashima Arts (MIT).

## Deploy

Static hosting only — drag the folder onto Netlify or use GitHub Pages. No build step.
