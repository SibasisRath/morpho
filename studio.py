#!/usr/bin/env python3
"""
studio.py — Morpho tuning studio (local development tool, never deployed).

    .venv/bin/python studio.py            # then open http://localhost:8765

One page, two halves:
  left  — upload a photo, tune every bake knob, hit Bake
  right — the real site in an iframe, showing the latest bake; look controls
          (background, size, speed, colour mode, glow) update it LIVE via
          the site's postMessage config API (index.html, section 8)

How it fits together:
  - Baking happens in-process by importing bake() from make_face_data.py —
    same code path as the CLI, so what you tune here is exactly what
    `make_face_data.py photo.jpg --flags` will reproduce.
  - The freshly baked cloud is kept in memory and served at
    /site/face-data.js, so the iframe sees it after a reload; the file on
    disk is only written when you click "Save to project" (or use the CLI).
  - Uploads and nothing else land in .studio/ (gitignored).

Routes:
  GET  /                  the studio UI (studio.html)
  GET  /site/<path>       the site, with face-data.js overridden by the bake
  POST /api/bake          multipart: photo? + any DEFAULTS keys as form fields
  GET  /api/preview.png   mask | dots | depth preview of the latest bake
  POST /api/save          write the latest bake to ./face-data.js
"""
import io
import json
from pathlib import Path

import cv2
from flask import Flask, request, jsonify, send_from_directory, send_file, Response

import make_face_data as mfd

ROOT = Path(__file__).resolve().parent
WORK = ROOT / '.studio'
WORK.mkdir(exist_ok=True)

app = Flask(__name__)

# Latest bake, kept in memory: {'js': str, 'preview_png': bytes, 'params': dict}
state = {}


@app.get('/')
def studio():
    return send_from_directory(ROOT, 'studio.html')


@app.get('/site/')
def site_index():
    return send_from_directory(ROOT, 'index.html')


@app.get('/site/<path:name>')
def site_file(name):
    if name == 'face-data.js' and 'js' in state:
        return Response(state['js'], mimetype='application/javascript')
    return send_from_directory(ROOT, name)


def _parse_params(form):
    """Pick out bake parameters from the form, casting to DEFAULTS' types."""
    params = {}
    for key, default in mfd.DEFAULTS.items():
        if key not in form or form[key] == '':
            continue
        raw = form[key]
        if isinstance(default, bool):
            params[key] = raw in ('1', 'true', 'on')
        elif isinstance(default, int):
            params[key] = int(float(raw))
        elif isinstance(default, float):
            params[key] = float(raw)
        else:                                   # str-or-None knobs (mask, depth)
            params[key] = raw or None
    return params


@app.post('/api/bake')
def api_bake():
    # a new photo is optional — without one we re-bake the previous upload
    photo = WORK / 'upload'
    if 'photo' in request.files and request.files['photo'].filename:
        request.files['photo'].save(photo)
    if not photo.exists():
        return jsonify(error='upload a photo first'), 400

    params = _parse_params(request.form)
    # painted corrections arrive as a file, not a form field
    if 'edits' in request.files and request.files['edits'].filename:
        edits_path = WORK / 'edits.png'
        request.files['edits'].save(edits_path)
        params['edits'] = str(edits_path)
    else:
        params.pop('edits', None)
    try:
        payload, preview = mfd.bake(str(photo), **params)
    except SystemExit as e:                      # pipeline aborts (empty mask etc.)
        return jsonify(error=str(e)), 422
    except Exception as e:
        return jsonify(error=f'{type(e).__name__}: {e}'), 500

    ok, png = cv2.imencode('.png', preview)
    state.update(js=mfd.payload_to_js(payload),
                 preview_png=png.tobytes() if ok else b'',
                 params=params)
    kb = (len(payload['xyz']) + len(payload['rgb'])) // 1024
    return jsonify(count=payload['count'], kb=kb, params=params)


@app.get('/api/preview.png')
def api_preview():
    if 'preview_png' not in state:
        return jsonify(error='no bake yet'), 404
    return send_file(io.BytesIO(state['preview_png']), mimetype='image/png')


EMBED_README = """Morpho embed
============
This folder is self-contained: index.html (the animation, design pre-applied)
+ face-data.js (the baked portrait). three.js and fonts load from CDNs.

Use it in a site either way:
  1. Copy the folder in and link to it:   <a href="/morpho/">…</a>
  2. Or embed it inline as an iframe:
     <iframe src="/morpho/" style="width:100%;height:100vh;border:0" title="Morpho"></iframe>

Tweak the design later by editing the window.MORPHO_DEFAULTS block at the
bottom of index.html — every key is documented in the configure() function.
Replace the portrait by baking a new face-data.js with Morpho Studio.
"""


@app.post('/api/export-embed')
def api_export_embed():
    """Build a drop-in embed zip: index.html with the studio's current design
    injected as window.MORPHO_DEFAULTS (applied at boot, section 8c) plus the
    latest baked face-data.js and a short README."""
    import zipfile
    config = request.get_json(silent=True) or {}
    html = (ROOT / 'index.html').read_text()
    anchor = '<script src="face-data.js"></script>'
    if anchor not in html:
        return jsonify(error='index.html changed — embed injection anchor missing'), 500
    inject = ('<script>\n// Design chosen in Morpho Studio — applied at boot (section 8c).\n'
              f'window.MORPHO_DEFAULTS = {json.dumps(config, indent=2)};\n</script>\n' + anchor)
    html = html.replace(anchor, inject)
    js = state.get('js') or (ROOT / 'face-data.js').read_text()

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('morpho/index.html', html)
        z.writestr('morpho/face-data.js', js)
        z.writestr('morpho/README.txt', EMBED_README)
    buf.seek(0)
    return send_file(buf, mimetype='application/zip', as_attachment=True,
                     download_name='morpho-embed.zip')


@app.post('/api/save')
def api_save():
    """Write the latest bake into the project's face-data.js (what git tracks)."""
    if 'js' not in state:
        return jsonify(error='no bake yet'), 404
    (ROOT / 'face-data.js').write_text(state['js'])
    # remember the winning recipe next to it, for reproducing via the CLI
    (WORK / 'last-saved-params.json').write_text(json.dumps(state['params'], indent=2))
    return jsonify(saved='face-data.js', params=state['params'])


if __name__ == '__main__':
    print('Morpho studio -> http://localhost:8765')
    app.run(host='127.0.0.1', port=8765, debug=False)
