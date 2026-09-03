#!/usr/bin/env python3
"""
make_face_data.py — turn a portrait photo into a colourful 3D point cloud
that the Dream Swarm site can fly its butterflies into.

    python3 make_face_data.py photo.jpg --points 16000 --out face-data.js

What it does
  1. Isolates the head from the background (GrabCut, seeded with a centred ellipse).
  2. Builds a depth relief for the head: a smooth dome (distance from the
     silhouette edge) plus lit-surface detail from image brightness. This is a
     cheap stand-in for a real depth model — swap in a monocular depth map
     (Depth Anything, MiDaS) later via --depth depth.png if you want true relief.
  3. Scatters N points with a jittered grid (blue-noise-ish, no clumps), but only
     where facial detail lives — edges and shadow tones — so the portrait reads
     through negative space like a stipple drawing.
  4. Samples the photo colour at each point and lifts saturation a little.
  5. Shuffles the points so any prefix (first 4k, first 16k…) is a uniform
     sub-sample — the site picks the count per device tier.
  6. Packs x,y,z as uint16 and r,g,b as uint8, base64, into a tiny JS file.

Output coordinate space: head height ≈ 2.0 units, centred at the origin,
x follows the photo aspect, z (depth) in roughly [-0.3, 0.3], +z toward camera.
"""
import argparse, base64, json, struct
import numpy as np
import cv2


def load_rgb(path, size):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        raise SystemExit(f"could not read {path}")
    h, w = im.shape[:2]
    s = size / max(h, w)
    im = cv2.resize(im, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def head_mask(rgb, iters=8):
    """GrabCut with an elliptical prior: portraits put the head in the middle."""
    h, w = rgb.shape[:2]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    mask = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    # probable foreground: big centred ellipse; sure foreground: smaller one
    cv2.ellipse(mask, (w // 2, int(h * 0.50)), (int(w * 0.40), int(h * 0.47)), 0, 0, 360, cv2.GC_PR_FGD, -1)
    cv2.ellipse(mask, (w // 2, int(h * 0.48)), (int(w * 0.18), int(h * 0.26)), 0, 0, 360, cv2.GC_FGD, -1)
    # image border is background
    b = max(2, w // 40)
    mask[:b, :] = cv2.GC_BGD; mask[-b:, :] = cv2.GC_BGD; mask[:, :b] = cv2.GC_BGD; mask[:, -b:] = cv2.GC_BGD
    bgd = np.zeros((1, 65), np.float64); fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, mask, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    m = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    # keep the largest blob, close holes, soften the edge a touch
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 1:
        big = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        m = np.where(lab == big, 255, 0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    return m


def relief_depth(rgb, mask, external=None):
    """0..1 depth inside the mask. 1 = closest to camera."""
    if external is not None:
        d = cv2.imread(external, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        d = cv2.resize(d, (rgb.shape[1], rgb.shape[0]))
        return d
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    dome = np.sqrt(dist / (dist.max() + 1e-6))          # rounded head volume
    lum = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    broad = cv2.GaussianBlur(lum, (0, 0), 9)            # lit forms (nose, cheeks, brow)
    fine = cv2.GaussianBlur(lum, (0, 0), 2) - broad     # small features
    d = 0.62 * dome + 0.30 * broad + 0.9 * fine
    d = cv2.GaussianBlur(d, (0, 0), 1.2)
    inside = mask > 0
    lo, hi = np.percentile(d[inside], [1, 99])
    return np.clip((d - lo) / (hi - lo + 1e-6), 0, 1)


def feature_weight(rgb, mask):
    """Stipple-portrait density: dots only where facial detail lives — edges
    (eyes, lips, hair) and shadow tones — gated to zero on lit skin so the
    face reads through negative space. A thin rim keeps the head contour."""
    inside = mask > 0
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    edge = cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (0, 0), 2.5)
    edge = np.clip(edge / (np.percentile(edge[inside], 97) + 1e-6), 0, 1)
    # shadow tone: 1 at the darkest values, fading to 0 at the 45th percentile —
    # anything brighter (lit skin) contributes nothing
    l0 = np.percentile(g[inside], 45); l99 = np.percentile(g[inside], 1)
    shade = np.clip((l0 - g) / (l0 - l99 + 1e-6), 0, 1)
    shade = cv2.GaussianBlur(shade, (0, 0), 1.5)
    w = np.clip(edge + 0.75 * shade ** 1.3, 0, 1)
    gate = np.clip((w - 0.14) / (0.34 - 0.14), 0, 1)
    gate = gate * gate * (3.0 - 2.0 * gate)          # smoothstep: hard zero below, gentle knee above
    w = w * gate
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    rim = ((dist > 0) & (dist < 6)).astype(np.float32) * 0.5
    return np.maximum(w, rim) * inside


def scatter(weight, n, rng):
    """Jittered-grid sampling with acceptance by weight -> ~n blue-noise-ish points."""
    h, w = weight.shape
    inside = weight > 0
    mean_w = weight[inside].mean()
    cells_needed = n / mean_w
    cell = np.sqrt(inside.sum() / cells_needed)
    pts = []
    passes = 0
    while len(pts) < n and passes < 6:
        off = rng.random(2) * cell
        ys = np.arange(off[0], h, cell); xs = np.arange(off[1], w, cell)
        gy, gx = np.meshgrid(ys, xs, indexing='ij')
        py = gy + rng.random(gy.shape) * cell; px = gx + rng.random(gx.shape) * cell
        py = py.ravel(); px = px.ravel()
        ok = (py < h - 1) & (px < w - 1)
        py, px = py[ok], px[ok]
        wv = weight[py.astype(int), px.astype(int)]
        acc = rng.random(len(wv)) < wv
        pts.append(np.stack([px[acc], py[acc]], 1))
        passes += 1
        cell *= 0.97
    p = np.concatenate(pts)
    rng.shuffle(p)
    return p[:n]


def sample_bilinear(img, px, py):
    x0 = np.floor(px).astype(int); y0 = np.floor(py).astype(int)
    fx = (px - x0)[:, None]; fy = (py - y0)[:, None]
    x1 = np.minimum(x0 + 1, img.shape[1] - 1); y1 = np.minimum(y0 + 1, img.shape[0] - 1)
    img = img.astype(np.float32)
    if img.ndim == 2:
        img = img[..., None]; fx = fx[:, 0]; fy = fy[:, 0]
        return (img[y0, x0, 0] * (1 - fx) * (1 - fy) + img[y0, x1, 0] * fx * (1 - fy)
                + img[y1, x0, 0] * (1 - fx) * fy + img[y1, x1, 0] * fx * fy)
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x1] * fx * (1 - fy)
            + img[y1, x0] * (1 - fx) * fy + img[y1, x1] * fx * fy)


def lift_saturation(rgb01, amount=0.35, gamma=0.9):
    hsv = cv2.cvtColor((rgb01[None] * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * (1 + amount) + 12, 0, 255)
    hsv[..., 2] = np.clip(255 * (hsv[..., 2] / 255) ** gamma, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)[0].astype(np.float32) / 255


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('photo')
    ap.add_argument('--points', type=int, default=16000)   # matches the top device tier; dots cluster on features only
    ap.add_argument('--size', type=int, default=768, help='working resolution (long edge)')
    ap.add_argument('--depth', default=None, help='optional external depth map png (white = near)')
    ap.add_argument('--depth-scale', type=float, default=0.6, help='total z range in scene units')
    ap.add_argument('--out', default='face-data.js')
    ap.add_argument('--preview', default='face-preview.png')
    ap.add_argument('--seed', type=int, default=7)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    rgb = load_rgb(a.photo, a.size)
    h, w = rgb.shape[:2]
    mask = head_mask(rgb)
    depth = relief_depth(rgb, mask, a.depth)
    weight = feature_weight(rgb, mask)
    pts = scatter(weight, a.points, rng)
    px, py = pts[:, 0], pts[:, 1]

    col = sample_bilinear(rgb, px, py) / 255.0
    col = lift_saturation(col)
    z = sample_bilinear(depth, px, py)

    # normalise: head bbox height -> 2.0, centred
    ys, xs = np.where(mask > 0)
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    scale = 2.0 / (ys.max() - ys.min())
    X = (px - cx) * scale
    Y = -(py - cy) * scale
    Z = (z - 0.5) * a.depth_scale

    xyz = np.stack([X, Y, Z], 1).astype(np.float32)
    lo = xyz.min(0); hi = xyz.max(0)
    q = np.round((xyz - lo) / (hi - lo + 1e-9) * 65535).astype(np.uint16)
    c8 = np.round(np.clip(col, 0, 1) * 255).astype(np.uint8)

    payload = {
        'count': int(len(xyz)),
        'min': [float(v) for v in lo], 'max': [float(v) for v in hi],
        'aspect': float(w / h),
        'xyz': base64.b64encode(q.tobytes()).decode(),
        'rgb': base64.b64encode(c8.tobytes()).decode(),
    }
    with open(a.out, 'w') as f:
        f.write('// Generated by make_face_data.py — 3D colour point cloud of a portrait.\n')
        f.write('window.FACE_DATA = ' + json.dumps(payload) + ';\n')

    # preview: dots over black, coloured, size by depth
    prev = np.zeros((h, w, 3), np.uint8)
    for (x, y), c, d in zip(pts[::1], (col * 255).astype(np.uint8), z):
        cv2.circle(prev, (int(x), int(y)), 1 if d < 0.5 else 2, tuple(int(v) for v in c), -1)
    dep = (depth * 255 * (mask > 0)).astype(np.uint8)
    cv2.imwrite(a.preview, cv2.cvtColor(np.hstack([prev, cv2.cvtColor(dep, cv2.COLOR_GRAY2RGB)]), cv2.COLOR_RGB2BGR))
    print(f'{len(xyz)} points -> {a.out} ({payload["count"]} pts, {len(payload["xyz"]) // 1024 + len(payload["rgb"]) // 1024} KB), preview {a.preview}')


if __name__ == '__main__':
    main()
