from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse
import webbrowser

from .source import LocalSource, Source
from .splits import SplitRequest, create_split

# Table keys the viewer fetches per episode; "gaze_pred" is the eval-only predicted-gaze track.
_TABLE_KEYS = {"timeline", "gaze", "gaze_pred", "annotations", "annotation_intervals", "depth"}


def serve(canonical_root: str | Path, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = False) -> str:
    return serve_source(LocalSource(canonical_root), host=host, port=port, open_browser=open_browser)


def serve_source(source: Source, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = False) -> str:
    class Handler(GazeRequestHandler):
        pass

    Handler.source = source

    httpd = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{httpd.server_address[1]}"
    if open_browser:
        webbrowser.open(url)
    print(f"Serving {source.describe()} at {url}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return url


class GazeRequestHandler(BaseHTTPRequestHandler):
    source: Source

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_text(viewer_html(), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/episodes":
            self.send_json({"episodes": self.source.episodes()})
            return
        if parsed.path.startswith("/api/episodes/"):
            self.handle_episode_get(parsed.path)
            return
        self.send_error(404, "not found")

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/api/splits":
            root = getattr(self.source, "canonical_root", None)
            if root is None:
                self.send_error(400, "splits are only supported for a local canonical root")
                return
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
            self.send_json(create_split(root, request))
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
        doc = self.source.episode_doc(dataset, episode_id)
        if doc is None:
            self.send_error(404, "episode not found")
            return
        if len(parts) == 3:
            self.send_json(doc)
            return
        key = parts[3]
        if key == "video":
            handle = self.source.open_video(dataset, episode_id)
            if handle is None:
                self.send_error(404, "video not available")
                return
            self.send_video(handle, "video/mp4")
            return
        if key in _TABLE_KEYS:
            self.send_json({"rows": self.source.table_rows(dataset, episode_id, key)})
            return
        self.send_error(404, "not found")

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
        from .s3fetch import LocalVideoHandle

        if not Path(path).exists():
            self.send_error(404, "file not found")
            return
        self.send_video(LocalVideoHandle(path), content_type)

    def send_video(self, handle, content_type: str) -> None:
        """Stream a range-capable video handle (local file or S3) with HTTP 206 support."""
        file_size = handle.size
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
        try:
            for chunk in handle.read_range(start, length):
                self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError):
            # The browser routinely aborts an in-flight range request when the user scrubs
            # the timeline or switches clips. The socket is gone, so just stop streaming --
            # this is expected, not an error.
            return

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
    tr.role-final td { font-weight: 700; }
    tr.role-source td { font-weight: 600; }
    tr.role-auxiliary td { color: #888; }
    .badge { display: inline-block; font-size: 10px; font-weight: 700; padding: 1px 6px; border-radius: 10px; vertical-align: middle; }
    .badge.fin { background: #2ecc7133; color: #2ecc71; border: 1px solid #2ecc71; }
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
      <div class="dur-filter" style="display:flex; gap:6px; align-items:center; margin:0 0 8px;">
        <input type="number" id="minDur" min="0" step="0.5" placeholder="min s" style="margin:0; -moz-appearance:textfield;" title="Minimum episode duration (seconds)">
        <span style="color:#888; font-size:12px;">–</span>
        <input type="number" id="maxDur" min="0" step="0.5" placeholder="max s" style="margin:0;" title="Maximum episode duration (seconds)">
      </div>
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
      <button id="downloadBtn" title="Download this clip as a video with the gaze overlay + annotation text burned in (d)">⬇ Download</button>
      <h2 id="title" style="margin:0 0 0 8px; font-size:16px;">Select an episode</h2>
    </div>
    <div class="hint">Keys: space play/pause · ←/k prev · →/j next · [ / ] speed · a auto-advance · d download · / focus search</div>
    <div class="stage no-video" id="stage"><video id="video" controls></video><div class="fallback" id="fallback">No playable video for this episode</div><canvas id="overlay"></canvas></div>
    <div class="now" id="evalMetrics" style="display:none"></div>
    <div class="now" id="currentAnnotation"><span class="muted">No annotation selected</span></div>
    <table><thead><tr><th>role</th><th>start_s</th><th>end_s</th><th>channel</th><th>text</th></tr></thead><tbody id="annotations"></tbody></table>
  </main>
  <script>
    const episodesEl = document.querySelector('#episodes');
    const datasetFilter = document.querySelector('#datasetFilter');
    const searchEl = document.querySelector('#search');
    const minDurEl = document.querySelector('#minDur');
    const maxDurEl = document.querySelector('#maxDur');
    const countEl = document.querySelector('#count');
    const prevBtn = document.querySelector('#prevBtn');
    const nextBtn = document.querySelector('#nextBtn');
    const speedEl = document.querySelector('#speed');
    const speedDown = document.querySelector('#speedDown');
    const speedUp = document.querySelector('#speedUp');
    const autoAdvanceBtn = document.querySelector('#autoAdvanceBtn');
    const downloadBtn = document.querySelector('#downloadBtn');
    const stage = document.querySelector('#stage');
    const video = document.querySelector('#video');
    const fallback = document.querySelector('#fallback');
    const currentAnnotation = document.querySelector('#currentAnnotation');
    const evalMetricsEl = document.querySelector('#evalMetrics');
    const canvas = document.querySelector('#overlay');
    const ctx = canvas.getContext('2d');
    let episodes = [];
    let visibleEpisodes = [];   // current filtered+searched list (nav order)
    let currentId = null;       // selected episode id
    let gaze = [];
    let gazePred = [];          // predicted gaze track (eval mode only)
    let evalMode = false;       // true when the episode carries model predictions + metrics
    let currentEpisodeDoc = null;  // the loaded episode.json (carries metrics in eval mode)
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
      minDurEl.oninput = renderEpisodeList;
      maxDurEl.oninput = renderEpisodeList;
      prevBtn.onclick = () => step(-1);
      nextBtn.onclick = () => step(1);
      speedDown.onclick = () => bumpRate(-RATE_STEP);
      speedUp.onclick = () => bumpRate(RATE_STEP);
      autoAdvanceBtn.onclick = () => setAutoAdvance(!autoAdvance);
      downloadBtn.onclick = () => downloadOverlayVideo();
      setRate(playbackRate);          // initialize the speed indicator
      setAutoAdvance(autoAdvance);    // initialize the auto-advance indicator (OFF)
      renderEpisodeList();
    }
    function renderEpisodeList() {
      const selected = datasetFilter.value;
      const q = (searchEl.value || '').trim().toLowerCase();
      // Duration range filter (seconds). Blank = unbounded; episodes with no duration_s
      // are only excluded when a bound is actually set.
      const minD = minDurEl.value === '' ? null : Number(minDurEl.value);
      const maxD = maxDurEl.value === '' ? null : Number(maxDurEl.value);
      const inDuration = (ep) => {
        if (minD == null && maxD == null) return true;
        const d = Number(ep.duration_s);
        if (!Number.isFinite(d)) return false;
        if (minD != null && d < minD) return false;
        if (maxD != null && d > maxD) return false;
        return true;
      };
      visibleEpisodes = episodes.filter(ep =>
        (!selected || ep.dataset === selected) &&
        (!q || String(ep.id).toLowerCase().includes(q)) &&
        inDuration(ep));
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
      // Don't hijack keys while typing in the duration filter inputs.
      if (e.target === minDurEl || e.target === maxDurEl) { if (e.key === 'Escape') e.target.blur(); return; }
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
      // 'd' downloads the current clip with the overlay + annotation text burned in.
      else if (e.key === 'd') { e.preventDefault(); downloadOverlayVideo(); }
    });
    async function loadEpisode(id) {
      const token = ++loadToken;
      currentId = id;
      markSelected();
      document.querySelector('#title').textContent = id;
      document.querySelector('#annotations').innerHTML = '';
      currentAnnotation.innerHTML = '<span class="muted">Loading annotations</span>';
      gaze = [];
      gazePred = [];
      evalMode = false;
      currentEpisodeDoc = null;
      evalMetricsEl.style.display = 'none';
      annotations = [];
      video.pause();
      video.removeAttribute('src');
      video.load();
      stage.style.aspectRatio = '';  // reset; re-set from intrinsic dims on loadedmetadata
      fallback.textContent = 'Loading episode...';
      setMediaState('loading');
      const ep = await json(`/api/episodes/${encodeURIComponent(id)}`);
      if (token !== loadToken) return;
      currentEpisodeDoc = ep;
      episodeDuration = ep.duration_s || (ep.metadata && ep.metadata.clip_end_time) || 0;
      frameSide = Number(ep.resolution) || 0;  // for x_px/y_px gaze normalization
      evalMode = !!ep.eval_mode;
      if (evalMode) renderEvalMetrics(ep);
      if (ep.files && ep.files.video) {
        setMediaState('loading', 'Loading video...');
        video.src = `/api/episodes/${encodeURIComponent(id)}/video`;
      } else {
        setMediaState('no-video', 'No video file for this episode');
      }
      const gazePayload = await json(`/api/episodes/${encodeURIComponent(id)}/gaze`);
      if (token !== loadToken) return;
      gaze = gazePayload.rows || [];
      if (evalMode) {
        const predPayload = await json(`/api/episodes/${encodeURIComponent(id)}/gaze_pred`);
        if (token !== loadToken) return;
        gazePred = predPayload.rows || [];
      }
      const intervalRows = (await json(`/api/episodes/${encodeURIComponent(id)}/annotation_intervals`)).rows || [];
      if (token !== loadToken) return;
      annotations = intervalRows.length ? intervalRows : sampledToIntervals((await json(`/api/episodes/${encodeURIComponent(id)}/annotations`)).rows || []);
      // Sort source first, then auxiliary; stable on start time within each group.
      annotations.sort((a, b) => (roleRank(a) - roleRank(b)) || (Number(a.start_s) - Number(b.start_s)));
      document.querySelector('#annotations').innerHTML = annotations.slice(0, 500).map((row, index) => {
        const role = (row.role || (row.label ? 'source' : '')).toLowerCase();
        const badge = role === 'final' ? '<span class="badge fin">final</span>'
                    : role === 'source' ? '<span class="badge src">source</span>'
                    : role === 'auxiliary' ? '<span class="badge aux">aux</span>' : '';
        const chan = row.channel || row.label || '';
        return `<tr data-index="${index}" class="role-${role}"><td>${badge}</td><td>${formatTime(row.start_s)}</td><td>${formatTime(row.end_s)}</td><td>${escapeHtml(chan)}</td><td>${escapeHtml(row.text || '')}</td></tr>`;
      }).join('');
      updateCurrentAnnotation();
      draw();
    }
    function roleRank(row) {
      const r = (row.role || (row.label ? 'source' : '')).toLowerCase();
      return r === 'final' ? 0 : r === 'source' ? 1 : 2;   // final, then source, then auxiliary
    }
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
    function isSource(row) {
      // 'final' and 'source' both render in the prominent (non-auxiliary) slot.
      return (row.role || (row.label ? 'source' : '')).toLowerCase() !== 'auxiliary';
    }
    function updateCurrentAnnotation() {
      const t = activeTime();
      const active = activeAnnotations(t);
      const activeIdx = new Set(active.map(r => annotations.indexOf(r)));
      [...document.querySelectorAll('#annotations tr')].forEach(tr => tr.classList.toggle('active-row', activeIdx.has(Number(tr.dataset.index))));
      if (!active.length) {
        currentAnnotation.innerHTML = `<span class="muted">${formatTime(t)}s: no active annotation</span>`;
        return;
      }
      const roleOf = r => (r.role || (r.label ? 'source' : '')).toLowerCase();
      const fin = active.find(r => roleOf(r) === 'final');
      const src = active.find(r => roleOf(r) === 'source');
      const aux = active.filter(r => roleOf(r) === 'auxiliary');
      let html = `<strong>${formatTime(t)}s</strong>`;
      if (fin) html += ` <span class="badge fin">final</span> ${escapeHtml(fin.text || '')}`;
      if (src) html += `<br><span class="badge src">source</span> <strong>${escapeHtml(src.channel || src.label || '')}</strong>: ${escapeHtml(src.text || '')}`;
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
      const r = videoContentRect();  // displayed video rect inside the stage (handles letterbox)
      // Ground-truth gaze (red). In eval mode the model's prediction (blue) is overlaid too.
      drawGazeDot(gaze, t, r, '#ff4757');
      if (evalMode) drawGazeDot(gazePred, t, r, '#1e90ff');
      updateCurrentAnnotation();
      requestAnimationFrame(draw);
    }
    function nearestByTime(rows, t) {
      return rows.reduce((best, item) => Math.abs(item.time_s - t) < Math.abs((best?.time_s ?? 999999) - t) ? item : best, null);
    }
    function drawGazeDot(rows, t, r, color) {
      const norm = gazeNorm(nearestByTime(rows, t));  // {x,y} in [0,1] on the video frame, or null
      if (!norm) return;
      const cx = r.x + norm.x * r.w;
      const cy = r.y + norm.y * r.h;
      ctx.fillStyle = color;
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
    function renderEvalMetrics(ep) {
      const m = ep.metrics || {};
      const fmt = (v) => (v == null ? '—' : Number(v).toFixed(3));
      const pct = (v) => (v == null ? '—' : (Number(v) * 100).toFixed(1) + '%');
      evalMetricsEl.innerHTML =
        '<strong>Eval</strong> '
        + '<span class="badge" style="background:#ff475733;color:#ff6b78;border:1px solid #ff4757">GT</span>'
        + ' <span class="badge" style="background:#1e90ff33;color:#7ec0ff;border:1px solid #1e90ff">pred</span>'
        + ` &nbsp; L2: <strong>${fmt(m.l2)}</strong>`
        + ` &nbsp; acc@5: ${pct(m['acc@5'])}`
        + ` &nbsp; acc@10: ${pct(m['acc@10'])}`
        + ` &nbsp; acc@15: ${pct(m['acc@15'])}`
        + ` &nbsp; valid: ${m.valid == null ? '—' : (Number(m.valid) ? 'yes' : 'no')}`;
      evalMetricsEl.style.display = '';
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
    // ---- Download: re-render the clip to an offscreen canvas with the gaze overlay + the
    // active annotation text burned in, capture it via MediaRecorder, and save as a video. ----
    function activeAnnotationLines(t) {
      // The text shown over the frame at time t: GT gaze (eval), then final/source/aux rows.
      const active = activeAnnotations(t);
      const roleOf = r => (r.role || (r.label ? 'source' : '')).toLowerCase();
      const lines = [];
      const fin = active.find(r => roleOf(r) === 'final');
      const src = active.find(r => roleOf(r) === 'source');
      if (fin && fin.text) lines.push(fin.text);
      if (src) { const c = src.channel || src.label || ''; lines.push((c ? c + ': ' : '') + (src.text || '')); }
      for (const a of active.filter(r => roleOf(r) === 'auxiliary')) {
        const c = a.channel || a.label || ''; lines.push((c ? c + ': ' : '') + (a.text || ''));
      }
      return lines.filter(Boolean);
    }
    function wrapText(c2d, text, maxWidth) {
      const words = String(text).split(/\\s+/);
      const out = []; let line = '';
      for (const w of words) {
        const trial = line ? line + ' ' + w : w;
        if (c2d.measureText(trial).width > maxWidth && line) { out.push(line); line = w; }
        else line = trial;
      }
      if (line) out.push(line);
      return out;
    }
    function drawDotOn(c2d, rows, t, w, h, color) {
      const norm = gazeNorm(nearestByTime(rows, t));
      if (!norm) return;
      const cx = norm.x * w, cy = norm.y * h;
      const rad = Math.max(6, Math.round(Math.min(w, h) * 0.018));
      c2d.fillStyle = color; c2d.strokeStyle = '#ffffff'; c2d.lineWidth = Math.max(2, rad / 4.5);
      c2d.beginPath(); c2d.arc(cx, cy, rad, 0, Math.PI * 2); c2d.fill(); c2d.stroke();
      const arm = rad * 1.8;
      c2d.beginPath();
      c2d.moveTo(cx - arm, cy); c2d.lineTo(cx + arm, cy);
      c2d.moveTo(cx, cy - arm); c2d.lineTo(cx, cy + arm);
      c2d.stroke();
    }
    function pickMime() {
      const cands = ['video/mp4;codecs=avc1.42E01E', 'video/mp4', 'video/webm;codecs=vp9', 'video/webm'];
      for (const m of cands) { if (window.MediaRecorder && MediaRecorder.isTypeSupported(m)) return m; }
      return '';
    }
    let downloading = false;
    async function downloadOverlayVideo() {
      if (downloading) return;
      if (mediaState !== 'video' || !video.videoWidth) { alert('No playable video to download for this episode.'); return; }
      if (!window.MediaRecorder) { alert('Your browser does not support MediaRecorder; cannot export.'); return; }
      downloading = true;
      const label = downloadBtn.textContent; downloadBtn.disabled = true;
      const wasPaused = video.paused, savedTime = video.currentTime, savedRate = video.playbackRate;
      try {
        const w = video.videoWidth, h = video.videoHeight;
        const off = document.createElement('canvas'); off.width = w; off.height = h;
        const c2d = off.getContext('2d');
        const fontPx = Math.max(14, Math.round(h * 0.035));
        const mime = pickMime();
        const stream = off.captureStream(30);
        const chunks = [];
        const rec = new MediaRecorder(stream, mime ? { mimeType: mime, videoBitsPerSecond: 8_000_000 } : undefined);
        rec.ondataavailable = (e) => { if (e.data && e.data.size) chunks.push(e.data); };
        const done = new Promise(res => { rec.onstop = res; });

        const renderFrame = () => {
          const t = video.currentTime;
          c2d.clearRect(0, 0, w, h);
          c2d.drawImage(video, 0, 0, w, h);
          drawDotOn(c2d, gaze, t, w, h, '#ff4757');
          if (evalMode) drawDotOn(c2d, gazePred, t, w, h, '#1e90ff');
          // Annotation text, wrapped, bottom-anchored with a legibility shadow.
          c2d.font = `${fontPx}px sans-serif`;
          const lines = [];
          for (const ln of activeAnnotationLines(t)) lines.push(...wrapText(c2d, ln, w * 0.94));
          c2d.textBaseline = 'bottom'; c2d.textAlign = 'left';
          let y = h - Math.round(fontPx * 0.5);
          for (let i = lines.length - 1; i >= 0; i--) {
            const x = Math.round(w * 0.03);
            c2d.lineWidth = Math.max(3, fontPx / 6); c2d.strokeStyle = 'rgba(0,0,0,0.85)';
            c2d.strokeText(lines[i], x, y); c2d.fillStyle = '#ffffff'; c2d.fillText(lines[i], x, y);
            y -= Math.round(fontPx * 1.25);
          }
          // Eval legend + metrics, top-left.
          if (evalMode) {
            c2d.font = `${Math.round(fontPx*0.85)}px sans-serif`; c2d.textBaseline = 'top';
            const m = (currentEpisodeDoc && currentEpisodeDoc.metrics) || {};
            const tag = `GT(red) pred(blue)  L2 ${m.l2==null?'—':Number(m.l2).toFixed(2)}  acc@5 ${m['acc@5']==null?'—':(m['acc@5']*100).toFixed(0)+'%'}`;
            const ty = Math.round(fontPx*0.4), tx = Math.round(w*0.03);
            c2d.lineWidth = Math.max(3, fontPx/6); c2d.strokeStyle = 'rgba(0,0,0,0.85)';
            c2d.strokeText(tag, tx, ty); c2d.fillStyle = '#ffffff'; c2d.fillText(tag, tx, ty);
          }
        };

        let raf = 0;
        const pump = () => { renderFrame(); raf = requestAnimationFrame(pump); };
        // Play from the start at 1x so the capture covers the whole clip in real time.
        video.pause();
        if (video.currentTime > 0.01) {
          await new Promise(r => { const onSeek = () => { video.removeEventListener('seeked', onSeek); r(); }; video.addEventListener('seeked', onSeek); video.currentTime = 0; });
        }
        video.playbackRate = 1.0;
        rec.start();
        pump();
        await video.play();
        downloadBtn.textContent = '● Recording…';
        // Stop at clip end; guard with a timeout (clip duration + 2s) in case 'ended' never fires.
        const capMs = ((video.duration || episodeDuration || 10) + 2) * 1000 / (video.playbackRate || 1);
        await new Promise(res => {
          let to = setTimeout(() => { video.removeEventListener('ended', onEnd); res(); }, capMs);
          function onEnd() { clearTimeout(to); video.removeEventListener('ended', onEnd); res(); }
          video.addEventListener('ended', onEnd);
        });
        cancelAnimationFrame(raf);
        rec.stop();
        await done;

        const type = (mime || 'video/webm').split(';')[0];
        const ext = type.includes('mp4') ? 'mp4' : 'webm';
        const blob = new Blob(chunks, { type });
        const safe = String(currentId || 'clip').replace(/[^a-zA-Z0-9._-]+/g, '_');
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = `${safe}.overlay.${ext}`;
        document.body.appendChild(a); a.click(); a.remove();
        setTimeout(() => URL.revokeObjectURL(a.href), 10000);
      } catch (err) {
        console.error(err); alert('Download failed: ' + err);
      } finally {
        // Restore the live player.
        try { video.pause(); video.currentTime = savedTime; video.playbackRate = savedRate || playbackRate; if (!wasPaused) video.play().catch(() => {}); } catch (e) {}
        downloadBtn.disabled = false; downloadBtn.textContent = label; downloading = false;
      }
    }
    loadEpisodes();
  </script>
</body>
</html>"""
