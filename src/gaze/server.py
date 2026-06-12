from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse
import webbrowser

from .splits import SplitRequest, create_split
from .table import read_table


def serve(canonical_root: str | Path, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = False) -> str:
    root = Path(canonical_root).resolve()

    class Handler(GazeRequestHandler):
        canonical_root = root

    httpd = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{httpd.server_address[1]}"
    if open_browser:
        webbrowser.open(url)
    print(f"Serving {root} at {url}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return url


class GazeRequestHandler(BaseHTTPRequestHandler):
    canonical_root: Path

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_text(viewer_html(), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/episodes":
            self.send_json({"episodes": self.episodes()})
            return
        if parsed.path.startswith("/api/episodes/"):
            self.handle_episode_get(parsed.path)
            return
        self.send_error(404, "not found")

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/api/splits":
            payload = self.read_json()
            request = SplitRequest(
                name=payload.get("name", "default"),
                ratios=payload.get("ratios"),
                seed=int(payload.get("seed", 0)),
                mode=payload.get("mode", "heterogeneous"),
                include_datasets=set(payload.get("include_datasets") or []),
                include_modalities=set(payload.get("include_modalities") or []),
                group_by=payload.get("group_by", "dataset"),
                stratify_by=payload.get("stratify_by"),
            )
            self.send_json(create_split(self.canonical_root, request))
            return
        self.send_error(404, "not found")

    def handle_episode_get(self, path: str) -> None:
        parts = [unquote(part) for part in path.split("/") if part]
        if len(parts) < 3:
            self.send_error(404, "missing episode id")
            return
        joined = parts[2]
        if ":" not in joined:
            self.send_error(400, "episode id must be dataset:episode_id")
            return
        dataset, episode_id = joined.split(":", 1)
        episode_root = self.canonical_root / "episodes" / dataset / episode_id
        episode_doc_path = episode_root / "episode.json"
        if not episode_doc_path.exists():
            self.send_error(404, "episode not found")
            return
        doc = json.loads(episode_doc_path.read_text(encoding="utf-8"))
        if len(parts) == 3:
            self.send_json(doc)
            return
        key = parts[3]
        if key == "video":
            video = doc.get("files", {}).get("video")
            if not video:
                self.send_error(404, "video not available")
                return
            self.send_file(episode_root / video, "video/mp4")
            return
        if key in {"timeline", "gaze", "annotations", "annotation_intervals", "depth"}:
            table = doc.get("files", {}).get(key)
            if not table:
                self.send_json({"rows": []})
                return
            self.send_json({"rows": read_table(episode_root / table)})
            return
        self.send_error(404, "not found")

    def episodes(self) -> list[dict]:
        manifest = read_table(self.canonical_root / "manifest.parquet")
        return [
            {
                "id": f"{row.get('dataset')}:{row.get('episode_id')}",
                **row,
            }
            for row in manifest
        ]

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def send_json(self, payload: dict) -> None:
        self.send_text(json.dumps(payload, sort_keys=True), "application/json")

    def send_text(self, text: str, content_type: str) -> None:
        data = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_file(self, path: Path, content_type: str) -> None:
        if not path.exists():
            self.send_error(404, "file not found")
            return
        file_size = path.stat().st_size
        range_header = self.headers.get("Range")
        start = 0
        end = file_size - 1
        status = 200
        if range_header:
            try:
                unit, _, spec = range_header.partition("=")
                if unit.strip() != "bytes":
                    raise ValueError("unsupported range unit")
                start_text, _, end_text = spec.partition("-")
                if start_text:
                    start = int(start_text)
                    end = int(end_text) if end_text else end
                else:
                    suffix = int(end_text)
                    start = max(file_size - suffix, 0)
                if start < 0 or end < start or start >= file_size:
                    raise ValueError("invalid range")
                end = min(end, file_size - 1)
                status = 206
            except ValueError:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{file_size}")
                self.end_headers()
                return
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.end_headers()
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining > 0:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        return


def viewer_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Gaze Viewer</title>
  <style>
    :root { color-scheme: light dark; font-family: ui-sans-serif, system-ui, sans-serif; }
    body { margin: 0; display: grid; grid-template-columns: 340px 1fr; height: 100vh; overflow: hidden; }
    aside { border-right: 1px solid #9995; display: flex; flex-direction: column; min-height: 0; }
    .sidebar-head { padding: 12px 14px 8px; border-bottom: 1px solid #9994; position: sticky; top: 0; background: Canvas; z-index: 3; }
    .sidebar-head h1 { font-size: 16px; margin: 0 0 8px; }
    .episode-list { overflow-y: auto; flex: 1 1 auto; padding: 6px 8px 16px; min-height: 0; }
    main { padding: 16px; display: grid; gap: 12px; align-content: start; overflow-y: auto; height: 100vh; }
    button.ep { display: block; width: 100%; text-align: left; padding: 6px 8px; margin: 3px 0; border: 1px solid #9996; background: transparent; border-radius: 6px; font-size: 12px; line-height: 1.3; cursor: pointer; word-break: break-all; }
    button.ep:hover { border-color: #ff4757aa; }
    button.ep.selected { background: #ff475722; border-color: #ff4757; font-weight: 600; }
    button.ep .ep-sub { color: #888; font-weight: 400; font-size: 11px; }
    select, input[type=search] { width: 100%; padding: 7px; margin: 0 0 8px; border: 1px solid #9996; border-radius: 6px; background: Canvas; color: CanvasText; box-sizing: border-box; font-size: 13px; }
    .navbar { display: flex; gap: 8px; align-items: center; }
    .navbar button { flex: 0 0 auto; padding: 6px 12px; border: 1px solid #9996; background: transparent; border-radius: 6px; cursor: pointer; }
    .navbar button:disabled { opacity: 0.4; cursor: default; }
    .navbar button.on { background: #2ecc7133; border-color: #2ecc71; font-weight: 600; }
    #speed { min-width: 52px; text-align: center; font-variant-numeric: tabular-nums; }
    .count { font-size: 12px; color: #888; margin: 2px 0 8px; }
    .stage { position: relative; width: min(100%, 960px); aspect-ratio: 16 / 9; background: #111; overflow: hidden; }
    video, canvas, .fallback { position: absolute; inset: 0; width: 100%; height: 100%; }
    video { display: block; object-fit: contain; }
    canvas { pointer-events: none; z-index: 2; }
    .fallback { display: none; place-items: center; color: #ddd; font-size: 14px; z-index: 1; }
    .stage.no-video video { display: none; }
    .stage.no-video .fallback { display: grid; }
    .now { width: min(100%, 960px); border: 1px solid #9994; padding: 10px; border-radius: 6px; min-height: 44px; }
    .muted { color: #777; }
    .hint { font-size: 11px; color: #999; }
    .active-row { outline: 2px solid #ff4757; outline-offset: -2px; }
    table { border-collapse: collapse; width: min(100%, 960px); }
    td, th { border: 1px solid #9994; padding: 4px 6px; font-size: 13px; }
    tr.role-source td { font-weight: 600; }
    tr.role-auxiliary td { color: #888; }
    .badge { display: inline-block; font-size: 10px; font-weight: 700; padding: 1px 6px; border-radius: 10px; vertical-align: middle; }
    .badge.src { background: #ff475733; color: #ff6b78; border: 1px solid #ff4757; }
    .badge.aux { background: #6c8cff22; color: #8aa2ff; border: 1px solid #6c8cff88; }
  </style>
</head>
<body>
  <aside>
    <div class="sidebar-head">
      <h1>Episodes</h1>
      <select id="datasetFilter"></select>
      <input type="search" id="search" placeholder="Filter episodes (id substring)…" autocomplete="off">
      <div class="count" id="count"></div>
    </div>
    <div class="episode-list" id="episodes"></div>
  </aside>
  <main>
    <div class="navbar">
      <button id="prevBtn" title="Previous (← or k)">◀ Prev</button>
      <button id="nextBtn" title="Next (→ or j)">Next ▶</button>
      <button id="speedDown" title="Slower ([ or -)">−</button>
      <span id="speed" title="Playback speed (applies to all clips)">1.00×</span>
      <button id="speedUp" title="Faster (] or +)">+</button>
      <button id="autoAdvanceBtn" title="Auto-advance to next clip when current ends (a)">Auto-advance: OFF</button>
      <h2 id="title" style="margin:0 0 0 8px; font-size:16px;">Select an episode</h2>
    </div>
    <div class="hint">Keys: space play/pause · ←/k prev · →/j next · [ / ] speed · a auto-advance · / focus search</div>
    <div class="stage no-video" id="stage"><video id="video" controls></video><div class="fallback" id="fallback">No playable video for this episode</div><canvas id="overlay"></canvas></div>
    <div class="now" id="currentAnnotation"><span class="muted">No annotation selected</span></div>
    <table><thead><tr><th>role</th><th>start_s</th><th>end_s</th><th>channel</th><th>text</th></tr></thead><tbody id="annotations"></tbody></table>
  </main>
  <script>
    const episodesEl = document.querySelector('#episodes');
    const datasetFilter = document.querySelector('#datasetFilter');
    const searchEl = document.querySelector('#search');
    const countEl = document.querySelector('#count');
    const prevBtn = document.querySelector('#prevBtn');
    const nextBtn = document.querySelector('#nextBtn');
    const speedEl = document.querySelector('#speed');
    const speedDown = document.querySelector('#speedDown');
    const speedUp = document.querySelector('#speedUp');
    const autoAdvanceBtn = document.querySelector('#autoAdvanceBtn');
    const stage = document.querySelector('#stage');
    const video = document.querySelector('#video');
    const fallback = document.querySelector('#fallback');
    const currentAnnotation = document.querySelector('#currentAnnotation');
    const canvas = document.querySelector('#overlay');
    const ctx = canvas.getContext('2d');
    let episodes = [];
    let visibleEpisodes = [];   // current filtered+searched list (nav order)
    let currentId = null;       // selected episode id
    let gaze = [];
    let annotations = [];
    let mediaState = 'empty';
    let episodeDuration = 0;
    let playbackRate = 1.0;     // GLOBAL playback speed; carries across clip switches
    let autoAdvance = false;    // when a clip ends, auto-load the next clip (toggle: 'a')
    let frameSide = 0;  // square pixel side for x_px/y_px gaze (e.g. 378); 0 = unknown
    let syntheticStart = performance.now();
    let loadToken = 0;
    async function json(url) { return (await fetch(url)).json(); }
    function setMediaState(state, message) {
      mediaState = state;
      stage.classList.toggle('no-video', state !== 'video');
      if (message) fallback.textContent = message;
      if (state === 'no-video') syntheticStart = performance.now();
    }
    video.addEventListener('loadedmetadata', () => {
      setMediaState('video');
      video.playbackRate = playbackRate;   // re-apply the global speed to the new clip
      // Match the stage to the video's aspect ratio so the overlay maps 1:1 (square
      // molmo2 clips, 4:3 egtea/egome, etc.) -- avoids letterbox + dot drift.
      if (video.videoWidth && video.videoHeight) {
        stage.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;
      }
    });
    video.addEventListener('canplay', () => {
      setMediaState('video');
      video.playbackRate = playbackRate;
      // Autoplay the clip as soon as it's selected (clips have no audio track, so the
      // browser's autoplay policy permits this). play() rejects if blocked -> ignore.
      video.play().catch(() => {});
    });
    // The browser resets playbackRate to 1.0 on a new source; if it differs from our
    // global rate (e.g. user changed speed mid-load), snap it back.
    video.addEventListener('ratechange', () => {
      if (Math.abs(video.playbackRate - playbackRate) > 1e-3) video.playbackRate = playbackRate;
    });
    video.addEventListener('error', () => setMediaState('no-video', 'No playable video for this episode'));
    video.addEventListener('timeupdate', updateCurrentAnnotation);
    video.addEventListener('seeked', updateCurrentAnnotation);
    // Auto-advance: when a clip ends, move to the next one (if the mode is on).
    video.addEventListener('ended', () => { if (autoAdvance) step(1); });
    async function loadEpisodes() {
      const payload = await json('/api/episodes');
      episodes = payload.episodes;
      const counts = {};
      episodes.forEach(ep => { if (ep.dataset) counts[ep.dataset] = (counts[ep.dataset] || 0) + 1; });
      const datasets = Object.keys(counts).sort();
      datasetFilter.innerHTML = `<option value="">All datasets (${episodes.length})</option>` +
        datasets.map(name => `<option value="${escapeHtml(name)}">${escapeHtml(name)} (${counts[name]})</option>`).join('');
      datasetFilter.onchange = renderEpisodeList;
      searchEl.oninput = renderEpisodeList;
      prevBtn.onclick = () => step(-1);
      nextBtn.onclick = () => step(1);
      speedDown.onclick = () => bumpRate(-RATE_STEP);
      speedUp.onclick = () => bumpRate(RATE_STEP);
      autoAdvanceBtn.onclick = () => setAutoAdvance(!autoAdvance);
      setRate(playbackRate);          // initialize the speed indicator
      setAutoAdvance(autoAdvance);    // initialize the auto-advance indicator (OFF)
      renderEpisodeList();
    }
    function renderEpisodeList() {
      const selected = datasetFilter.value;
      const q = (searchEl.value || '').trim().toLowerCase();
      visibleEpisodes = episodes.filter(ep =>
        (!selected || ep.dataset === selected) &&
        (!q || String(ep.id).toLowerCase().includes(q)));
      countEl.textContent = `${visibleEpisodes.length} of ${episodes.length} episode(s)`;
      episodesEl.innerHTML = '';
      const frag = document.createDocumentFragment();
      visibleEpisodes.forEach(ep => {
        const button = document.createElement('button');
        button.className = 'ep' + (ep.id === currentId ? ' selected' : '');
        button.dataset.id = ep.id;
        button.innerHTML = `${escapeHtml(ep.id)}<br><span class="ep-sub">${escapeHtml(ep.modalities || '')}</span>`;
        button.onclick = () => loadEpisode(ep.id);
        frag.appendChild(button);
      });
      episodesEl.appendChild(frag);
      updateNavButtons();
    }
    function updateNavButtons() {
      const i = visibleEpisodes.findIndex(ep => ep.id === currentId);
      prevBtn.disabled = !(i > 0);
      nextBtn.disabled = !(i >= 0 && i < visibleEpisodes.length - 1);
    }
    function step(delta) {
      if (!visibleEpisodes.length) return;
      let i = visibleEpisodes.findIndex(ep => ep.id === currentId);
      if (i < 0) { i = delta > 0 ? -1 : visibleEpisodes.length; }
      const ni = i + delta;
      if (ni < 0 || ni >= visibleEpisodes.length) return;
      loadEpisode(visibleEpisodes[ni].id);
    }
    function markSelected() {
      [...episodesEl.querySelectorAll('button.ep')].forEach(b =>
        b.classList.toggle('selected', b.dataset.id === currentId));
      const sel = episodesEl.querySelector('button.ep.selected');
      if (sel) sel.scrollIntoView({ block: 'nearest' });
      updateNavButtons();
    }
    function togglePlay() {
      if (mediaState !== 'video') return;
      if (video.paused) video.play(); else video.pause();
    }
    const MIN_RATE = 0.25, MAX_RATE = 4.0, RATE_STEP = 0.25;
    function setRate(r) {
      playbackRate = Math.min(MAX_RATE, Math.max(MIN_RATE, Math.round(r / RATE_STEP) * RATE_STEP));
      video.playbackRate = playbackRate;   // applies to the current clip immediately
      speedEl.textContent = playbackRate.toFixed(2) + '×';
    }
    function bumpRate(delta) { setRate(playbackRate + delta); }
    function setAutoAdvance(on) {
      autoAdvance = !!on;
      autoAdvanceBtn.textContent = 'Auto-advance: ' + (autoAdvance ? 'ON' : 'OFF');
      autoAdvanceBtn.classList.toggle('on', autoAdvance);
    }
    document.addEventListener('keydown', (e) => {
      if (e.target === searchEl) { if (e.key === 'Escape') searchEl.blur(); return; }
      if (e.key === 'ArrowRight' || e.key === 'j') { e.preventDefault(); step(1); }
      else if (e.key === 'ArrowLeft' || e.key === 'k') { e.preventDefault(); step(-1); }
      else if (e.key === '/') { e.preventDefault(); searchEl.focus(); }
      else if (e.key === ' ' || e.code === 'Space') {
        // Space toggles play/pause. preventDefault stops the page from scrolling and,
        // when the <video controls> element is focused, stops the browser's own Space
        // handler from also toggling (which would cancel ours out).
        e.preventDefault();
        togglePlay();
      }
      // Playback speed in 0.25 steps; GLOBAL (carries across clip switches). ']'/'+'/'='
      // faster, '['/'-' slower.
      else if (e.key === ']' || e.key === '+' || e.key === '=') { e.preventDefault(); bumpRate(RATE_STEP); }
      else if (e.key === '[' || e.key === '-') { e.preventDefault(); bumpRate(-RATE_STEP); }
      // 'a' toggles auto-advance to the next clip when the current one ends.
      else if (e.key === 'a') { e.preventDefault(); setAutoAdvance(!autoAdvance); }
    });
    async function loadEpisode(id) {
      const token = ++loadToken;
      currentId = id;
      markSelected();
      document.querySelector('#title').textContent = id;
      document.querySelector('#annotations').innerHTML = '';
      currentAnnotation.innerHTML = '<span class="muted">Loading annotations</span>';
      gaze = [];
      annotations = [];
      video.pause();
      video.removeAttribute('src');
      video.load();
      stage.style.aspectRatio = '';  // reset; re-set from intrinsic dims on loadedmetadata
      fallback.textContent = 'Loading episode...';
      setMediaState('loading');
      const ep = await json(`/api/episodes/${encodeURIComponent(id)}`);
      if (token !== loadToken) return;
      episodeDuration = ep.duration_s || (ep.metadata && ep.metadata.clip_end_time) || 0;
      frameSide = Number(ep.resolution) || 0;  // for x_px/y_px gaze normalization
      if (ep.files.video) {
        setMediaState('loading', 'Loading video...');
        video.src = `/api/episodes/${encodeURIComponent(id)}/video`;
      } else {
        setMediaState('no-video', 'No video file for this episode');
      }
      const gazePayload = await json(`/api/episodes/${encodeURIComponent(id)}/gaze`);
      if (token !== loadToken) return;
      gaze = gazePayload.rows || [];
      const intervalRows = (await json(`/api/episodes/${encodeURIComponent(id)}/annotation_intervals`)).rows || [];
      if (token !== loadToken) return;
      annotations = intervalRows.length ? intervalRows : sampledToIntervals((await json(`/api/episodes/${encodeURIComponent(id)}/annotations`)).rows || []);
      // Sort source first, then auxiliary; stable on start time within each group.
      annotations.sort((a, b) => (roleRank(a) - roleRank(b)) || (Number(a.start_s) - Number(b.start_s)));
      document.querySelector('#annotations').innerHTML = annotations.slice(0, 500).map((row, index) => {
        const role = (row.role || (row.label ? 'source' : '')).toLowerCase();
        const badge = role === 'source' ? '<span class="badge src">source</span>'
                    : role === 'auxiliary' ? '<span class="badge aux">aux</span>' : '';
        const chan = row.channel || row.label || '';
        return `<tr data-index="${index}" class="role-${role}"><td>${badge}</td><td>${formatTime(row.start_s)}</td><td>${formatTime(row.end_s)}</td><td>${escapeHtml(chan)}</td><td>${escapeHtml(row.text || '')}</td></tr>`;
      }).join('');
      updateCurrentAnnotation();
      draw();
    }
    function roleRank(row) { return (row.role || (row.label ? 'source' : '')).toLowerCase() === 'auxiliary' ? 1 : 0; }
    function sampledToIntervals(rows) {
      return rows.map((row, index) => ({ start_s: row.time_s, end_s: rows[index + 1]?.time_s ?? episodeDuration, role: row.role, channel: row.channel, label: row.label, text: row.text }));
    }
    function activeTime() {
      if (mediaState === 'video') return video.currentTime || 0;
      if (mediaState === 'no-video') return episodeDuration ? ((performance.now() - syntheticStart) / 1000) % episodeDuration : 0;
      return 0;
    }
    function activeAnnotations(t) {
      // All rows covering t (annotations is sorted source-first).
      return annotations.filter(row => Number(row.start_s) <= t && t < Number(row.end_s));
    }
    function isSource(row) { return (row.role || (row.label ? 'source' : '')).toLowerCase() !== 'auxiliary'; }
    function updateCurrentAnnotation() {
      const t = activeTime();
      const active = activeAnnotations(t);
      const activeIdx = new Set(active.map(r => annotations.indexOf(r)));
      [...document.querySelectorAll('#annotations tr')].forEach(tr => tr.classList.toggle('active-row', activeIdx.has(Number(tr.dataset.index))));
      if (!active.length) {
        currentAnnotation.innerHTML = `<span class="muted">${formatTime(t)}s: no active annotation</span>`;
        return;
      }
      const src = active.find(isSource);
      const aux = active.filter(r => !isSource(r));
      let html = `<strong>${formatTime(t)}s</strong>`;
      if (src) html += ` <span class="badge src">source</span> <strong>${escapeHtml(src.channel || src.label || '')}</strong>: ${escapeHtml(src.text || '')}`;
      for (const a of aux) html += `<br><span class="badge aux">aux</span> <span class="muted">${escapeHtml(a.channel || a.label || '')}</span>: ${escapeHtml(a.text || '')}`;
      currentAnnotation.innerHTML = html;
    }
    function draw() {
      const rect = stage.getBoundingClientRect();
      canvas.width = rect.width;
      canvas.height = rect.height;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const t = activeTime();
      if (mediaState === 'no-video') {
        ctx.strokeStyle = '#ffffff22';
        ctx.lineWidth = 1;
        for (let x = 0; x < canvas.width; x += 80) {
          ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
        }
        for (let y = 0; y < canvas.height; y += 80) {
          ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke();
        }
      }
      if (mediaState === 'loading') {
        requestAnimationFrame(draw);
        return;
      }
      const row = gaze.reduce((best, item) => Math.abs(item.time_s - t) < Math.abs((best?.time_s || 999999) - t) ? item : best, null);
      const norm = gazeNorm(row);  // {x,y} in [0,1] on the video frame, or null
      if (norm) {
        const r = videoContentRect();  // displayed video rect inside the stage (handles letterbox)
        const cx = r.x + norm.x * r.w;
        const cy = r.y + norm.y * r.h;
        ctx.fillStyle = '#ff4757';
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(cx, cy, 9, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        // small crosshair for precision
        ctx.beginPath();
        ctx.moveTo(cx - 16, cy); ctx.lineTo(cx + 16, cy);
        ctx.moveTo(cx, cy - 16); ctx.lineTo(cx, cy + 16);
        ctx.stroke();
      }
      updateCurrentAnnotation();
      requestAnimationFrame(draw);
    }
    function gazeNorm(row) {
      // Normalize a gaze row to [0,1] on the VIDEO FRAME, supporting both forms:
      //   x_norm/y_norm (rectify/canonical) or x_px/y_px on a `frameSide` square (molmo2).
      if (!row) return null;
      if (row.x_norm != null && row.y_norm != null) return { x: row.x_norm, y: row.y_norm };
      if (row.x_px != null && row.y_px != null) {
        const side = frameSide || (video.videoWidth && video.videoHeight ? Math.max(video.videoWidth, video.videoHeight) : 0);
        if (side > 0) return { x: row.x_px / side, y: row.y_px / side };
      }
      return null;
    }
    function videoContentRect() {
      // The rect (in canvas px) where the video content is actually drawn inside the
      // stage, accounting for object-fit: contain letterboxing. Falls back to the full
      // canvas when intrinsic video dims are unknown (e.g. no-video synthetic mode).
      const vw = video.videoWidth, vh = video.videoHeight;
      if (!vw || !vh || mediaState !== 'video') {
        return { x: 0, y: 0, w: canvas.width, h: canvas.height };
      }
      const scale = Math.min(canvas.width / vw, canvas.height / vh);
      const w = vw * scale, h = vh * scale;
      return { x: (canvas.width - w) / 2, y: (canvas.height - h) / 2, w, h };
    }
    function formatTime(value) {
      const number = Number(value);
      return Number.isFinite(number) ? number.toFixed(2).replace(/\\.?0+$/, '') : '';
    }
    function escapeHtml(value) {
      return String(value).replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
    }
    loadEpisodes();
  </script>
</body>
</html>"""
