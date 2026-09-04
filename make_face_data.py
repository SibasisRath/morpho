#!/usr/bin/env python3
"""
make_face_data.py — turn any portrait photo into the halftone 3D point cloud
that Morpho flies its butterflies into.

    .venv/bin/python make_face_data.py photo.jpg --out face-data.js

The photo can have any background — the pipeline isolates the person first.
Every stage is a flag you can see and tune (this project is a study vehicle):

  1. FACE CROP  auto-centers on the face with headroom            (--no-crop)
  2. MASK       photo alpha channel > rembg person segmentation >
                GrabCut fallback; or bring your own                (--mask m.png,
                                                                    --classic-mask)
  3. DENOISE    edge-preserving, BEFORE contrast — otherwise sensor
                grain gets amplified into fake detail dots         (--denoise 0..20)
  4. CONTRAST   grayscale -> CLAHE local contrast + unsharp relief:
                gives flat photos the light/shadow drama the
                halftone style needs                               (--clahe, --contrast)
  5. HALFTONE   dot density follows brightness — dense on lit skin,
                thinning into shadow, ZERO below the floor so dark
                areas stay dramatic voids; edges add detail; an
                optional rim traces dark hair against dark bg      (--tone-gamma,
                                                                    --floor, --edge, --rim)
  6. SCATTER + COLOR + DEPTH + PACK — jittered-grid blue-noise dots, photo
     colours (saturation lifted), dome-or-external depth relief (--depth d.png),
     shuffled so any prefix is a uniform sub-sample, packed to base64 JS.

Check face-preview.png before opening the site: left = mask contour on the
photo, middle = the dots, right = depth relief.
"""
import argparse, base64, json
import numpy as np
import cv2


# ----------------------------------------------------------------------
# 1. Load + face crop
# ----------------------------------------------------------------------

def load_image(path, size):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise SystemExit(f"could not read {path}")
    alpha = None
    if im.ndim == 3 and im.shape[2] == 4:
        alpha = im[:, :, 3]
        im = im[:, :, :3]
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    h, w = im.shape[:2]
    s = size / max(h, w)
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    im = cv2.resize(im, (round(w * s), round(h * s)), interpolation=interp)
    if alpha is not None:
        alpha = cv2.resize(alpha, (im.shape[1], im.shape[0]), interpolation=interp)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB), alpha


YUNET_URL = ('https://github.com/opencv/opencv_zoo/raw/main/models/'
             'face_detection_yunet/face_detection_yunet_2023mar.onnx')


def _yunet_model():
    import os, urllib.request
    cache = os.path.expanduser('~/.cache/morpho')
    os.makedirs(cache, exist_ok=True)
    p = os.path.join(cache, 'face_detection_yunet_2023mar.onnx')
    if not os.path.exists(p):
        print('downloading YuNet face detector (~230 KB, one time)')
        urllib.request.urlretrieve(YUNET_URL, p)
    return p


def detect_face(rgb):
    """Biggest face box (x, y, w, h) via YuNet, or None."""
    try:
        det = cv2.FaceDetectorYN.create(_yunet_model(), '', (rgb.shape[1], rgb.shape[0]), 0.6)
        _, faces = det.detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    except Exception as e:
        print(f'face detection unavailable ({e})')
        return None
    if faces is None or len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2] * f[3])[:4]


def face_crop(rgb, alpha):
    """Center the crop on the detected face so off-center photos still work.
    Generous margins: hair above, shoulders below. Returns the face box in
    cropped coordinates so later stages can prioritise the face."""
    face = detect_face(rgb)
    if face is None:
        print('face crop: no face found, using the full frame')
        return rgb, alpha, None
    x, y, w, h = face
    cx, cy = x + w / 2, y + h / 2
    half_w, above, below = w * 1.35, h * 1.15, h * 1.45  # headroom for hair, room for chin/neck
    x0 = max(0, int(cx - half_w)); x1 = min(rgb.shape[1], int(cx + half_w))
    y0 = max(0, int(cy - above));  y1 = min(rgb.shape[0], int(cy + below))
    face = (x - x0, y - y0, w, h)
    return rgb[y0:y1, x0:x1], (alpha[y0:y1, x0:x1] if alpha is not None else None), face


# ----------------------------------------------------------------------
# 2. Mask: alpha > external file > rembg > GrabCut
# ----------------------------------------------------------------------

def clean_mask(m):
    """Largest blob, closed holes — shared post-processing for every mask source."""
    m = (m > 127).astype(np.uint8) * 255
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 1:
        big = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        m = np.where(lab == big, 255, 0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)


def rembg_mask(rgb, subject='face'):
    """Segmentation that survives any background (local ONNX models):
    the person-tuned model for faces, the general salient-object one otherwise."""
    from rembg import remove, new_session
    names = ['u2net_human_seg', 'u2net'] if subject == 'face' else ['u2net']
    session = None
    for name in names:
        try:
            session = new_session(name)
            break
        except Exception:
            continue
    if session is None:
        raise RuntimeError('no rembg model available')
    out = remove(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), session=session, only_mask=True)
    return np.asarray(out)


def grabcut_mask(rgb, iters=8):
    """No-dependency fallback: assumes a roughly centred head (the face crop helps)."""
    h, w = rgb.shape[:2]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    mask = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    cv2.ellipse(mask, (w // 2, int(h * 0.50)), (int(w * 0.40), int(h * 0.47)), 0, 0, 360, cv2.GC_PR_FGD, -1)
    cv2.ellipse(mask, (w // 2, int(h * 0.48)), (int(w * 0.18), int(h * 0.26)), 0, 0, 360, cv2.GC_FGD, -1)
    b = max(2, w // 40)
    mask[:b, :] = cv2.GC_BGD; mask[-b:, :] = cv2.GC_BGD; mask[:, :b] = cv2.GC_BGD; mask[:, -b:] = cv2.GC_BGD
    bgd = np.zeros((1, 65), np.float64); fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, mask, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    return np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)


def head_mask(rgb, alpha, args):
    if alpha is not None and alpha.min() < 250:          # photo already cut out
        print('mask: using the photo\'s own alpha channel')
        return clean_mask(alpha)
    if args.mask:
        m = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise SystemExit(f"could not read mask {args.mask}")
        print(f'mask: external file {args.mask}')
        return clean_mask(cv2.resize(m, (rgb.shape[1], rgb.shape[0])))
    if not args.classic_mask:
        try:
            print('mask: rembg person segmentation')
            return clean_mask(rembg_mask(rgb, getattr(args, 'subject', 'face')))
        except Exception as e:
            print(f'mask: rembg unavailable ({e}); falling back to GrabCut')
    else:
        print('mask: GrabCut (classic)')
    return clean_mask(grabcut_mask(rgb))


# ----------------------------------------------------------------------
# 3+4. Denoise, grayscale, CLAHE, unsharp relief
# ----------------------------------------------------------------------

def preprocess(rgb, args):
    """Returns (clean_rgb for colour sampling, tonal gray 0..1 for dot density)."""
    clean = rgb
    if args.denoise > 0:
        clean = cv2.fastNlMeansDenoisingColored(rgb, None, args.denoise, args.denoise, 7, 21)
    g = cv2.cvtColor(clean, cv2.COLOR_RGB2GRAY)
    if args.clahe > 0:
        g = cv2.createCLAHE(clipLimit=args.clahe, tileGridSize=(8, 8)).apply(g)
    g = g.astype(np.float32) / 255.0
    if args.contrast > 0:
        # unsharp relief: amplify local light-vs-shadow so form (nose, brow,
        # cheek) carves into the dot density even on evenly lit photos
        sigma = 9.0 * max(g.shape) / 768.0
        broad = cv2.GaussianBlur(g, (0, 0), sigma)
        g = g + args.contrast * (g - broad)
    return clean, g


# ----------------------------------------------------------------------
# 5. Halftone weights — the style lives here
# ----------------------------------------------------------------------

def feature_weight(gray, mask, args, face=None):
    """Dot density = brightness through a tone curve, plus edge detail.
    A hard floor keeps shadows as true voids; --rim traces the silhouette
    so dark hair against a dark page still has an outline.

    Face-aware: the tone range is measured on the FACE, not the whole person —
    otherwise a white shirt eats the top of the range and the face goes flat.
    Density outside the face tapers to --body so clothing stays a whisper.

    Returns (weight, tone): weight drives WHERE dots land; tone (the
    normalised 0..1 brightness, pre-gamma) is baked per point so the site
    can drive dot SIZE and duotone colour from it — classic halftone
    modulates size with tone, not just spacing."""
    inside = mask > 0
    scope = inside
    emphasis = np.ones_like(gray)
    if face is not None and args.body < 1.0:
        x, y, fw, fh = (float(v) for v in face)
        fm = np.zeros(gray.shape, np.float32)
        cv2.ellipse(fm, (int(x + fw / 2), int(y + fh * 0.45)),
                    (int(fw * 0.75), int(fh * 1.05)), 0, 0, 360, 1.0, -1)
        soft = cv2.GaussianBlur(fm, (0, 0), 0.35 * max(fw, fh))
        soft = np.clip(soft / (soft.max() + 1e-6), 0, 1)
        emphasis = args.body + (1.0 - args.body) * soft
        face_scope = inside & (fm > 0)
        if face_scope.sum() > 100:
            scope = face_scope
    t_lo = np.percentile(gray[scope], args.tone_lo)
    t_hi = np.percentile(gray[scope], args.tone_hi)
    tone = np.clip((gray - t_lo) / (t_hi - t_lo + 1e-6), 0, 1)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge = cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (0, 0), 2.5)
    edge = np.clip(edge / (np.percentile(edge[inside], 97) + 1e-6), 0, 1)

    w = (tone ** args.tone_gamma + args.edge * edge) * emphasis
    w[w < args.floor] = 0.0                              # shadows stay empty
    w = np.clip(w, 0, 1)

    if args.rim > 0:
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        rim = ((dist > 0) & (dist < 6)).astype(np.float32) * args.rim
        w = np.maximum(w, rim)
    return w * inside, tone * inside


# ----------------------------------------------------------------------
# 6. Depth, scatter, colours, packing (the original machinery)
# ----------------------------------------------------------------------

def relief_depth(gray01, mask, external=None):
    """0..1 depth inside the mask. 1 = closest to camera."""
    if external is not None:
        d = cv2.imread(external, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        return cv2.resize(d, (mask.shape[1], mask.shape[0]))
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    dome = np.sqrt(dist / (dist.max() + 1e-6))
    broad = cv2.GaussianBlur(gray01, (0, 0), 9)
    fine = cv2.GaussianBlur(gray01, (0, 0), 2) - broad
    d = 0.62 * dome + 0.30 * broad + 0.9 * fine
    d = cv2.GaussianBlur(d, (0, 0), 1.2)
    inside = mask > 0
    lo, hi = np.percentile(d[inside], [1, 99])
    return np.clip((d - lo) / (hi - lo + 1e-6), 0, 1)


def scatter(weight, n, rng):
    """Jittered-grid sampling with acceptance by weight -> ~n blue-noise-ish points."""
    h, w = weight.shape
    inside = weight > 0
    if inside.sum() == 0:
        raise SystemExit('weight map is empty — lower --floor or check the mask preview')
    mean_w = weight[inside].mean()
    cell = np.sqrt(inside.sum() / (n / mean_w))
    pts = []
    passes = 0
    while sum(len(p) for p in pts) < n and passes < 8:
        off = rng.random(2) * cell
        ys = np.arange(off[0], h, cell); xs = np.arange(off[1], w, cell)
        gy, gx = np.meshgrid(ys, xs, indexing='ij')
        py = (gy + rng.random(gy.shape) * cell).ravel()
        px = (gx + rng.random(gx.shape) * cell).ravel()
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


# ----------------------------------------------------------------------
# 7. The pipeline as a library — bake() is what the CLI and the tuning
#    studio (studio.py) both call. DEFAULTS is the single source of truth
#    for every knob; the CLI flags below mirror it 1:1.
# ----------------------------------------------------------------------

DEFAULTS = dict(
    points=16000,      # matches the top device tier
    size=768,          # working resolution (long edge)
    seed=7,
    subject='face',    # 'face' (crop + face-priority) or 'object' (any subject)
    crop=None,         # manual crop 'x,y,w,h' as 0..1 fractions of the photo; overrides auto crop
    edits=None,        # painted corrections png: green strokes add dots, red strokes erase
    mask=None,         # external mask png (white = keep)
    classic_mask=False,  # skip rembg, use GrabCut
    no_crop=False,     # skip face-detection crop
    denoise=5.0,       # 0 = off; raise for grainy phone photos
    clahe=2.0,         # local contrast clip limit; 0 = off
    contrast=1.0,      # unsharp relief amount
    tone_lo=10.0,      # percentile mapped to black
    tone_hi=80.0,      # percentile mapped to white
    tone_gamma=2.6,    # higher = denser highlights, emptier midtones
    floor=0.18,        # weights below this become zero (voids)
    edge=0.15,         # extra density on edges/features
    rim=0.0,           # silhouette outline weight (helps dark hair), try 0.4
    body=0.35,         # density multiplier outside the face (1 = no face priority)
    depth=None,        # external depth map png (white = near)
    depth_scale=0.6,   # total z range in scene units
)


def apply_crop(rgb, alpha, spec):
    """Manual selection: 'x,y,w,h' as 0..1 fractions of the (resized) photo."""
    try:
        x, y, w, h = (float(v) for v in spec.split(','))
    except ValueError:
        raise SystemExit(f'bad --crop "{spec}", expected x,y,w,h as 0..1 fractions')
    H, W = rgb.shape[:2]
    x0 = max(0, int(x * W)); y0 = max(0, int(y * H))
    x1 = min(W, int((x + w) * W)); y1 = min(H, int((y + h) * H))
    if x1 - x0 < 32 or y1 - y0 < 32:
        raise SystemExit('crop selection is too small')
    return rgb[y0:y1, x0:x1], (alpha[y0:y1, x0:x1] if alpha is not None else None)


def apply_edits(weight, edits_path):
    """Painted corrections over the working image: green strokes force dots in
    (weight raised to stroke opacity), red strokes erase (weight -> 0).
    Painting edits the density FIELD, not individual dots — with a fixed seed,
    unpainted regions re-bake identically."""
    im = cv2.imread(edits_path, cv2.IMREAD_UNCHANGED)
    if im is None or im.ndim != 3 or im.shape[2] != 4:
        raise SystemExit(f'could not read edits png (need RGBA): {edits_path}')
    im = cv2.resize(im, (weight.shape[1], weight.shape[0]), interpolation=cv2.INTER_LINEAR)
    a = im[:, :, 3].astype(np.float32) / 255.0
    g = im[:, :, 1].astype(np.float32) / 255.0 * a       # add strokes
    r = im[:, :, 2].astype(np.float32) / 255.0 * a       # erase strokes (BGR order)
    # dodge & burn semantics: stroke opacity IS the strength, so a light pass
    # gently thickens/thins and a full-strength pass forces/erases outright
    weight = np.maximum(weight, g * 0.85)
    weight = weight * (1.0 - r)
    return weight


def bake(photo, **params):
    """Run the full photo -> point-cloud pipeline.

    photo: path to an image file. params override DEFAULTS (same names).
    Returns (payload, preview) where payload is the FACE_DATA dict ready to
    be JSON-serialised, and preview is a BGR image (mask | dots | depth).
    """
    from types import SimpleNamespace
    unknown = set(params) - set(DEFAULTS)
    if unknown:
        raise ValueError(f'unknown bake parameters: {sorted(unknown)}')
    a = SimpleNamespace(photo=photo, **{**DEFAULTS, **params})

    rng = np.random.default_rng(int(a.seed))
    rgb, alpha = load_image(a.photo, int(a.size))
    if a.crop:
        rgb, alpha = apply_crop(rgb, alpha, a.crop)
    face = None
    if a.subject == 'face':
        if not a.no_crop and not a.crop:
            rgb, alpha, face = face_crop(rgb, alpha)
        elif a.body < 1.0:
            face = detect_face(rgb)
    # object mode: no face logic at all — feature_weight falls back to
    # whole-mask tone percentiles and uniform emphasis
    mask = head_mask(rgb, alpha, a)
    clean, gray = preprocess(rgb, a)
    weight, tonemap = feature_weight(gray, mask, a, face)
    if a.edits:
        weight = apply_edits(weight, a.edits)
    depth = relief_depth(np.clip(gray, 0, 1), mask, a.depth)

    pts = scatter(weight, int(a.points), rng)
    px, py = pts[:, 0], pts[:, 1]
    col = lift_saturation(sample_bilinear(clean, px, py) / 255.0)
    z = sample_bilinear(depth, px, py)
    tone_pts = np.clip(sample_bilinear(tonemap, px, py), 0, 1)

    # normalise: mask bbox height -> 2.0 scene units, centred at the origin
    ys, xs = np.where(mask > 0)
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    scale = 2.0 / (ys.max() - ys.min())
    xyz = np.stack([(px - cx) * scale, -(py - cy) * scale, (z - 0.5) * a.depth_scale], 1).astype(np.float32)
    lo = xyz.min(0); hi = xyz.max(0)
    q = np.round((xyz - lo) / (hi - lo + 1e-9) * 65535).astype(np.uint16)
    c8 = np.round(np.clip(col, 0, 1) * 255).astype(np.uint8)

    payload = {
        'count': int(len(xyz)),
        'min': [float(v) for v in lo], 'max': [float(v) for v in hi],
        'aspect': float(rgb.shape[1] / rgb.shape[0]),
        'xyz': base64.b64encode(q.tobytes()).decode(),
        'rgb': base64.b64encode(c8.tobytes()).decode(),
        # per-point tone (0..255): drives dot size and duotone colour on the
        # site; older face-data files without it fall back to rgb luminance
        'tone': base64.b64encode(np.round(tone_pts * 255).astype(np.uint8).tobytes()).decode(),
    }

    # preview: mask contour on photo | dots | depth
    h, w = rgb.shape[:2]
    left = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(left, contours, -1, (80, 220, 80), 2)
    dots = np.zeros((h, w, 3), np.uint8)
    for (x, y), c, d in zip(pts, (col * 255).astype(np.uint8), z):
        cv2.circle(dots, (int(x), int(y)), 1 if d < 0.5 else 2, tuple(int(v) for v in c[::-1]), -1)
    dep = cv2.cvtColor((depth * 255 * (mask > 0)).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    preview = np.hstack([left, dots, dep])
    return payload, preview


def payload_to_js(payload):
    """FACE_DATA dict -> the face-data.js file contents the site loads."""
    return ('// Generated by make_face_data.py — halftone 3D colour point cloud of a portrait.\n'
            'window.FACE_DATA = ' + json.dumps(payload) + ';\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = DEFAULTS
    ap.add_argument('photo')
    ap.add_argument('--points', type=int, default=d['points'], help='matches the top device tier')
    ap.add_argument('--size', type=int, default=d['size'], help='working resolution (long edge)')
    ap.add_argument('--out', default='face-data.js')
    ap.add_argument('--preview', default='face-preview.png')
    ap.add_argument('--seed', type=int, default=d['seed'])
    # subject & framing
    ap.add_argument('--subject', choices=['face', 'object'], default=d['subject'],
                    help='face = crop + face-priority weighting; object = any subject (logo, sneaker, bonsai…)')
    ap.add_argument('--crop', default=d['crop'], help="manual selection 'x,y,w,h' as 0..1 fractions; overrides auto crop")
    ap.add_argument('--edits', default=d['edits'], help='painted corrections png (green = add dots, red = erase)')
    # masking
    ap.add_argument('--mask', default=d['mask'], help='external mask png (white = keep)')
    ap.add_argument('--classic-mask', action='store_true', help='skip rembg, use GrabCut')
    ap.add_argument('--no-crop', action='store_true', help='skip face-detection crop')
    # preprocessing
    ap.add_argument('--denoise', type=float, default=d['denoise'], help='0 = off; raise for grainy phone photos')
    ap.add_argument('--clahe', type=float, default=d['clahe'], help='local contrast clip limit; 0 = off')
    ap.add_argument('--contrast', type=float, default=d['contrast'], help='unsharp relief amount')
    # halftone
    ap.add_argument('--tone-lo', type=float, default=d['tone_lo'], help='percentile mapped to black')
    ap.add_argument('--tone-hi', type=float, default=d['tone_hi'], help='percentile mapped to white')
    ap.add_argument('--tone-gamma', type=float, default=d['tone_gamma'], help='higher = denser highlights, emptier midtones')
    ap.add_argument('--floor', type=float, default=d['floor'], help='weights below this become zero (voids)')
    ap.add_argument('--edge', type=float, default=d['edge'], help='extra density on edges/features')
    ap.add_argument('--rim', type=float, default=d['rim'], help='silhouette outline weight (helps dark hair), try 0.4')
    ap.add_argument('--body', type=float, default=d['body'], help='density multiplier outside the face (1 = no face priority)')
    # depth
    ap.add_argument('--depth', default=d['depth'], help='external depth map png (white = near)')
    ap.add_argument('--depth-scale', type=float, default=d['depth_scale'], help='total z range in scene units')
    a = ap.parse_args()

    params = {k: getattr(a, k) for k in DEFAULTS}
    payload, preview = bake(a.photo, **params)
    with open(a.out, 'w') as f:
        f.write(payload_to_js(payload))
    cv2.imwrite(a.preview, preview)
    kb = (len(payload['xyz']) + len(payload['rgb'])) // 1024
    print(f"{payload['count']} points -> {a.out} (~{kb} KB), preview -> {a.preview}")


if __name__ == '__main__':
    main()
