// Re-weight face-data.js for a stipple-portrait look: dots only where facial
// detail lives (edges + shadow tone) plus a thin rim, empty everywhere else.
// Pure Node — rasterizes the point cloud itself, no image libraries needed.
const fs = require('fs');
const zlib = require('zlib');

const SRC = process.argv[2];
const OUT = process.argv[3];
const PREVIEW = process.argv[4];
// Tunables
const EDGE_W = +(process.env.EDGE_W || 1.0);
const SHADE_W = +(process.env.SHADE_W || 0.75);
const GATE_LO = +(process.env.GATE_LO || 0.14);
const GATE_HI = +(process.env.GATE_HI || 0.34);
const RIM_W = +(process.env.RIM_W || 0.5);
const SHADE_P0 = +(process.env.SHADE_P0 || 45); // luminance percentile below which skin counts as lit → empty
const RES = +(process.env.RES || 300);          // raster height; higher = finer gaps survive
const SPLAT_R = +(process.env.SPLAT_R || 2);    // luminance raster blur
const HOLLOW = +(process.env.HOLLOW || 0);      // 0..1: hollow out the middle of large dark blobs

// ---- load ----
const txt = fs.readFileSync(SRC, 'utf8');
const json = JSON.parse(txt.slice(txt.indexOf('{'), txt.lastIndexOf('}') + 1));
const q = new Uint16Array(Buffer.from(json.xyz, 'base64').buffer.slice(0));
const rgb = Buffer.from(json.rgb, 'base64');
const n = json.count, lo = json.min, hi = json.max;
const X = new Float32Array(n), Y = new Float32Array(n), L = new Float32Array(n);
for (let i = 0; i < n; i++) {
  X[i] = lo[0] + (q[i * 3] / 65535) * (hi[0] - lo[0]);
  Y[i] = lo[1] + (q[i * 3 + 1] / 65535) * (hi[1] - lo[1]);
  L[i] = (0.2126 * rgb[i * 3] + 0.7152 * rgb[i * 3 + 1] + 0.0722 * rgb[i * 3 + 2]) / 255;
}

// ---- rasterize luminance from the points ----
const H = RES, W = Math.max(60, Math.round(H * (hi[0] - lo[0]) / (hi[1] - lo[1])));
const px = i => Math.min(W - 1, Math.max(0, ((X[i] - lo[0]) / (hi[0] - lo[0])) * (W - 1)));
const py = i => Math.min(H - 1, Math.max(0, (1 - (Y[i] - lo[1]) / (hi[1] - lo[1])) * (H - 1)));
const lumS = new Float32Array(W * H), cnt = new Float32Array(W * H);
for (let i = 0; i < n; i++) {
  const c = (py(i) | 0) * W + (px(i) | 0);
  lumS[c] += L[i]; cnt[c] += 1;
}
const blur = (a, r) => { // separable box blur
  const t = new Float32Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    let s = 0, m = 0;
    for (let k = -r; k <= r; k++) { const xx = x + k; if (xx >= 0 && xx < W) { s += a[y * W + xx]; m++; } }
    t[y * W + x] = s / m;
  }
  const o = new Float32Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    let s = 0, m = 0;
    for (let k = -r; k <= r; k++) { const yy = y + k; if (yy >= 0 && yy < H) { s += t[yy * W + x]; m++; } }
    o[y * W + x] = s / m;
  }
  return o;
};
const lumB = blur(blur(lumS, SPLAT_R), SPLAT_R), cntB = blur(blur(cnt, SPLAT_R), SPLAT_R);
const lum = new Float32Array(W * H), occ = new Uint8Array(W * H);
for (let c = 0; c < W * H; c++) { lum[c] = cntB[c] > 0.02 ? lumB[c] / cntB[c] : 0; occ[c] = cntB[c] > 0.02 ? 1 : 0; }

// ---- edge strength (Sobel) and shadow tone, percentile-normalized inside the head ----
const edge = new Float32Array(W * H);
for (let y = 1; y < H - 1; y++) for (let x = 1; x < W - 1; x++) {
  const g = (xx, yy) => lum[yy * W + xx];
  const gx = g(x + 1, y - 1) + 2 * g(x + 1, y) + g(x + 1, y + 1) - g(x - 1, y - 1) - 2 * g(x - 1, y) - g(x - 1, y + 1);
  const gy = g(x - 1, y + 1) + 2 * g(x, y + 1) + g(x + 1, y + 1) - g(x - 1, y - 1) - 2 * g(x, y - 1) - g(x + 1, y - 1);
  edge[y * W + x] = occ[y * W + x] ? Math.hypot(gx, gy) : 0;
}
const edgeB = blur(edge, 1);
const pct = (arr, mask, p) => {
  const v = []; for (let c = 0; c < W * H; c++) if (mask[c]) v.push(arr[c]);
  v.sort((a, b) => a - b); return v[Math.min(v.length - 1, Math.floor(v.length * p / 100))];
};
const e97 = pct(edgeB, occ, 97) + 1e-6;
const l0 = pct(lum, occ, SHADE_P0), l99 = pct(lum, occ, 1); // dark end
// shade: 1 at the darkest tones, 0 at/above the SHADE_P0 percentile
const shade = new Float32Array(W * H);
for (let c = 0; c < W * H; c++) shade[c] = occ[c] ? Math.min(1, Math.max(0, (l0 - lum[c]) / (l0 - l99 + 1e-6))) : 0;
let shadeB = blur(shade, 1);
if (HOLLOW > 0) {
  // deep inside a large dark blob (hair, brow mass) the wide-blurred shade saturates → fade the fill
  // there so the cluster is drawn by its outline, the way stipple portraits handle solid areas
  const deep = blur(blur(shadeB, 5), 5);
  const hollowed = new Float32Array(W * H);
  for (let c = 0; c < W * H; c++) {
    const d = Math.min(1, Math.max(0, (deep[c] - 0.55) / 0.35));
    hollowed[c] = shadeB[c] * (1 - HOLLOW * d * d * (3 - 2 * d));
  }
  shadeB = hollowed;
}

// ---- rim: occupied cells near the outside, so the head contour still reads ----
const rim = new Float32Array(W * H);
for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
  if (!occ[y * W + x]) continue;
  let empty = false;
  for (let dy = -2; dy <= 2 && !empty; dy++) for (let dx = -2; dx <= 2; dx++) {
    const xx = x + dx, yy = y + dy;
    if (xx < 0 || yy < 0 || xx >= W || yy >= H || !occ[yy * W + xx]) { empty = true; break; }
  }
  if (empty) rim[y * W + x] = RIM_W;
}

// ---- final weight ----
const smooth = (a, b, x) => { x = Math.min(1, Math.max(0, (x - a) / (b - a))); return x * x * (3 - 2 * x); };
const weight = new Float32Array(W * H);
if (process.env.MODE === 'tone') {
  // Halftone-portrait mode: density follows BRIGHTNESS — the lit surface of the
  // face is continuously covered, thinning smoothly into shadows (no hard gate),
  // so the portrait reads as one unified form. Edges add a little extra detail.
  // CONTRAST: unsharp mask on the luminance first — evenly lit portraits have a
  // narrow tonal range, and relief-style shading needs local light/shadow drama.
  const CONTRAST = +(process.env.CONTRAST || 0);
  let lumC = lum;
  if (CONTRAST > 0) {
    const broad = blur(blur(lum, 6), 6);
    lumC = new Float32Array(W * H);
    for (let c = 0; c < W * H; c++) lumC[c] = occ[c] ? lum[c] + CONTRAST * (lum[c] - broad[c]) : 0;
  }
  const t_lo = pct(lumC, occ, +(process.env.TONE_P_LO || 10));
  const t_hi = pct(lumC, occ, +(process.env.TONE_P_HI || 92));
  const GAMMA = +(process.env.GAMMA || 1.5);
  for (let c = 0; c < W * H; c++) {
    if (!occ[c]) continue;
    const tone = Math.min(1, Math.max(0, (lumC[c] - t_lo) / (t_hi - t_lo + 1e-6)));
    let w = Math.pow(tone, GAMMA) + EDGE_W * (edgeB[c] / e97);
    if (w < +(process.env.FLOOR || 0.045)) w = 0;          // shadows stay truly empty
    weight[c] = Math.min(1, Math.max(w, rim[c]));
  }
} else {
  for (let c = 0; c < W * H; c++) {
    const w = Math.min(1, EDGE_W * (edgeB[c] / e97) + SHADE_W * Math.pow(shadeB[c], 1.3));
    weight[c] = Math.max(w * smooth(GATE_LO, GATE_HI, w), rim[c] * occ[c]);
  }
}

// ---- accept/reject points (deterministic), preserving shuffled order ----
const rand = i => { let t = (i + 0x6D2B79F5) | 0; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
const keep = [];
for (let i = 0; i < n; i++) {
  const w = weight[(py(i) | 0) * W + (px(i) | 0)];
  if (rand(i) < w) keep.push(i);
}
console.log(`kept ${keep.length} / ${n} points (${(keep.length / n * 100).toFixed(1)}%)`);

// ---- write new face-data.js (reuse original quantized values — lossless) ----
const q2 = new Uint16Array(keep.length * 3), c2 = Buffer.alloc(keep.length * 3);
keep.forEach((i, k) => {
  q2[k * 3] = q[i * 3]; q2[k * 3 + 1] = q[i * 3 + 1]; q2[k * 3 + 2] = q[i * 3 + 2];
  c2[k * 3] = rgb[i * 3]; c2[k * 3 + 1] = rgb[i * 3 + 1]; c2[k * 3 + 2] = rgb[i * 3 + 2];
});
const out = { count: keep.length, min: json.min, max: json.max, aspect: json.aspect,
  xyz: Buffer.from(q2.buffer).toString('base64'), rgb: c2.toString('base64') };
fs.writeFileSync(OUT, '// Generated by make_face_data.py, re-stippled (features + rim only) for negative space.\nwindow.FACE_DATA = ' + JSON.stringify(out) + ';\n');

// ---- PNG preview of kept dots (minimal encoder, zlib built-in) ----
const SC = 3, PW = W * SC, PH = H * SC;
const img = Buffer.alloc(PH * (PW * 3 + 1));
for (let y = 0; y < PH; y++) img[y * (PW * 3 + 1)] = 0;
const put = (x, y, r, g, b) => { if (x < 0 || y < 0 || x >= PW || y >= PH) return; const o = y * (PW * 3 + 1) + 1 + x * 3; img[o] = r; img[o + 1] = g; img[o + 2] = b; };
keep.forEach(i => {
  const cx = Math.round(px(i) * SC), cy = Math.round(py(i) * SC);
  for (let dy = 0; dy <= 1; dy++) for (let dx = 0; dx <= 1; dx++) put(cx + dx, cy + dy, rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2]);
});
const crc32 = b => { let c = ~0; for (const x of b) { c ^= x; for (let k = 0; k < 8; k++) c = (c >>> 1) ^ (0xEDB88320 & -(c & 1)); } return ~c >>> 0; };
const chunk = (type, data) => { const len = Buffer.alloc(4); len.writeUInt32BE(data.length); const td = Buffer.concat([Buffer.from(type), data]); const crc = Buffer.alloc(4); crc.writeUInt32BE(crc32(td)); return Buffer.concat([len, td, crc]); };
const ihdr = Buffer.alloc(13); ihdr.writeUInt32BE(PW); ihdr.writeUInt32BE(PH, 4); ihdr[8] = 8; ihdr[9] = 2;
fs.writeFileSync(PREVIEW, Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk('IHDR', ihdr), chunk('IDAT', zlib.deflateSync(img)), chunk('IEND', Buffer.alloc(0))]));
console.log('preview:', PREVIEW);
