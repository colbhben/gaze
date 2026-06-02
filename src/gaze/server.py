from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse
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
        if key in {"timeline", "gaze", "annotations", "depth"}:
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
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

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
    body { margin: 0; display: grid; grid-template-columns: 320px 1fr; min-height: 100vh; }
    aside { border-right: 1px solid #9995; padding: 16px; overflow: auto; }
    main { padding: 16px; display: grid; gap: 12px; align-content: start; }
    button { width: 100%; text-align: left; padding: 8px; margin: 4px 0; border: 1px solid #9996; background: transparent; border-radius: 6px; }
    .stage { position: relative; width: min(100%, 960px); aspect-ratio: 16 / 9; background: #111; overflow: hidden; }
    video, canvas, .fallback { position: absolute; inset: 0; width: 100%; height: 100%; }
    video { display: block; object-fit: contain; }
    canvas { pointer-events: none; z-index: 2; }
    .fallback { display: none; place-items: center; color: #ddd; font-size: 14px; z-index: 1; }
    .stage.no-video video { display: none; }
    .stage.no-video .fallback { display: grid; }
    table { border-collapse: collapse; width: min(100%, 960px); }
    td, th { border: 1px solid #9994; padding: 4px 6px; font-size: 13px; }
  </style>
</head>
<body>
  <aside><h1>Episodes</h1><div id="episodes"></div></aside>
  <main>
    <h2 id="title">Select an episode</h2>
    <div class="stage no-video" id="stage"><video id="video" controls></video><div class="fallback" id="fallback">No playable video for this episode</div><canvas id="overlay"></canvas></div>
    <table><thead><tr><th>time_s</th><th>label</th><th>text</th></tr></thead><tbody id="annotations"></tbody></table>
  </main>
  <script>
    const episodesEl = document.querySelector('#episodes');
    const stage = document.querySelector('#stage');
    const video = document.querySelector('#video');
    const fallback = document.querySelector('#fallback');
    const canvas = document.querySelector('#overlay');
    const ctx = canvas.getContext('2d');
    let gaze = [];
    let noVideo = true;
    let episodeDuration = 0;
    let syntheticStart = performance.now();
    let loadToken = 0;
    async function json(url) { return (await fetch(url)).json(); }
    function setNoVideo(enabled) {
      noVideo = enabled;
      stage.classList.toggle('no-video', enabled);
      if (enabled) {
        video.removeAttribute('src');
        video.load();
        syntheticStart = performance.now();
      }
    }
    video.addEventListener('loadedmetadata', () => setNoVideo(false));
    video.addEventListener('error', () => setNoVideo(true));
    async function loadEpisodes() {
      const payload = await json('/api/episodes');
      episodesEl.innerHTML = '';
      payload.episodes.forEach(ep => {
        const button = document.createElement('button');
        button.textContent = `${ep.id} (${ep.modalities || ''})`;
        button.onclick = () => loadEpisode(ep.id);
        episodesEl.appendChild(button);
      });
    }
    async function loadEpisode(id) {
      const token = ++loadToken;
      document.querySelector('#title').textContent = id;
      document.querySelector('#annotations').innerHTML = '';
      gaze = [];
      fallback.textContent = 'Loading episode...';
      const ep = await json(`/api/episodes/${encodeURIComponent(id)}`);
      if (token !== loadToken) return;
      episodeDuration = ep.duration_s || 0;
      setNoVideo(true);
      fallback.textContent = ep.files.video ? 'No playable video for this episode' : 'No video file for this episode';
      if (ep.files.video) video.src = `/api/episodes/${encodeURIComponent(id)}/video`;
      const gazePayload = await json(`/api/episodes/${encodeURIComponent(id)}/gaze`);
      if (token !== loadToken) return;
      gaze = gazePayload.rows || [];
      const ann = (await json(`/api/episodes/${encodeURIComponent(id)}/annotations`)).rows || [];
      if (token !== loadToken) return;
      document.querySelector('#annotations').innerHTML = ann.slice(0, 200).map(row => `<tr><td>${row.time_s}</td><td>${row.label || ''}</td><td>${row.text || ''}</td></tr>`).join('');
      draw();
    }
    function draw() {
      const rect = stage.getBoundingClientRect();
      canvas.width = rect.width;
      canvas.height = rect.height;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const elapsed = (performance.now() - syntheticStart) / 1000;
      const t = noVideo ? (episodeDuration ? elapsed % episodeDuration : elapsed) : (video.currentTime || 0);
      if (noVideo) {
        ctx.strokeStyle = '#ffffff22';
        ctx.lineWidth = 1;
        for (let x = 0; x < canvas.width; x += 80) {
          ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
        }
        for (let y = 0; y < canvas.height; y += 80) {
          ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke();
        }
      }
      const row = gaze.reduce((best, item) => Math.abs(item.time_s - t) < Math.abs((best?.time_s || 999999) - t) ? item : best, null);
      if (row && row.x_norm != null && row.y_norm != null) {
        ctx.fillStyle = '#ff4757';
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(row.x_norm * canvas.width, row.y_norm * canvas.height, 9, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }
      requestAnimationFrame(draw);
    }
    loadEpisodes();
  </script>
</body>
</html>"""
