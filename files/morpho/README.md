# Morpho — a swarm dreams a face

Thousands of GPU-simulated butterflies flock across the screen, then settle into a
portrait made of light. Move the pointer and they scatter; let go and they find their
way home.

## Run it

Everything is static. Either:

- open `index.html` directly in Chrome/Edge/Firefox/Safari, or
- serve the folder (`python3 -m http.server 8000` → http://localhost:8000).

Keep `index.html` and `face-data.js` next to each other. Fonts and three.js load from CDNs.

Useful URL switches while tuning:

| switch | effect |
|---|---|
| `?n=96` | compute texture side length → 96² = 9,216 butterflies (16–256) |
| `?bloom=0` | turn off the bloom pass |
| `?stats=1` | fps / count / settle readout, top-left |

Press **R** or click **Replay** to send them flying again.

## Use your own photo

```
pip install numpy opencv-python-headless
python3 make_face_data.py your_portrait.jpg --points 16000 --out face-data.js
```

Best results: a front-facing portrait, head filling the middle of the frame, plain-ish
background, hair darker than the background. It writes `face-preview.png` so you can check
the mask and depth relief before opening the site. If the mask misses hair, try `--size 1024`
or a tighter crop. For true 3D relief, run any monocular depth model (Depth Anything, MiDaS)
on the photo and pass its output with `--depth depth.png`.

## Deploy

Drag the folder onto Netlify, or push to GitHub and enable Pages. No build step.

## Device tiers

| device | butterflies | bloom | pixel ratio |
|---|---|---|---|
| phone, ≤7 cores | 72² = 5,184 | off | ≤1.25 |
| phone, 8+ cores | 96² = 9,216 | off | ≤1.5 |
| desktop, ≤7 cores | 112² = 12,544 | on | ≤1.5 |
| desktop, 8+ cores | 128² = 16,384 | on | ≤2 |

Wing size is derived from the gap between dots, so the face stays legible at any count.

## How it works — a study map

Read `index.html` top to bottom; it is numbered 0–7.

**GPGPU.** Every butterfly is one texel in two textures, *position* (xyz + wing phase)
and *velocity*. Each frame two fragment shaders read the previous textures and write the
next ones (`GPUComputationRenderer` handles the ping-pong). The CPU never touches a
butterfly.

**The velocity shader is the whole piece.** It computes two accelerations and blends them
with one number, `uSettle`:

- *Flock* — curl noise (a divergence-free wind, so the swarm never clumps), a cheap boids
  approximation that samples 12 pseudo-random texels for separation / alignment / cohesion,
  a wandering attractor so the swarm stays on screen, and a soft boundary.
- *Settle* — a damped spring to each butterfly's home position on the face
  (`k = 9`, `damping = 5`, slightly under-damped so they overshoot a little). No idle
  force is added at rest, so the face comes to a true standstill until the pointer disturbs it.

The pointer is a third force added on top: a radial push with a small lift toward the
camera. Because the spring is always on when settled, "return to shape" is free.

**Rendering.** One `InstancedBufferGeometry`: two wing quads, `COUNT` instances. Each
instance has a `reference` uv into the simulation textures. The vertex shader orients the
body along velocity while flying and toward the camera at rest, flaps the wings about the
body axis using the phase stored in `position.w`, and does LOD: past `uLodNear` the wing
geometry is cross-faded (in view space) into a camera-facing glow quad. Colour is a pastel
rainbow in flight and the portrait's pixel colour at rest, with a slow hue drift.

**Post.** On desktop an `UnrealBloomPass` gives the dream haze. On mobile the material
tone-maps itself (`DIRECT_OUTPUT`) so both paths look alike.

**Things to try first when you start exploring**

- `velU.uSettle` from the console (`SWARM.velU.uSettle.value = 0.5`) freezes the blend mid-morph.
- Change `curlNoise(pos * 0.55 …)` — the multiplier is the size of the wind eddies.
- Swap the wing canvas drawing for a photo of a real wing.
- Replace the face target texture with any point set — text, a logo, a 3D scan.
- Add a second target and cross-fade between faces.

## Files

- `index.html` — the site, all shaders inline and commented
- `face-data.js` — baked point cloud (~14,000 points, ~165 KB; stippled — dots only on features, shadows and the head rim)
- `make_face_data.py` — photo → point cloud tool
- `placeholder_face.jpg` — a StyleGAN-generated person (nobody real); `face-preview.png` its dot/depth preview

Simplex noise in the shaders is Ian McEwan / Ashima Arts (MIT).
