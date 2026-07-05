import argparse
import json
import mimetypes
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import urllib.parse

import cv2


parser = argparse.ArgumentParser(
    description="Current-frame video + map viewer. Compares only the frame currently shown in the video."
)

parser.add_argument("-v", "--video", required=True, help="Path to the video file.")
parser.add_argument("-db", "--database", default="graph.db", help="Path to graph.db.")
parser.add_argument("-r", "--root", default=".", help="Project root for DB image paths.")
parser.add_argument(
    "--global-estimator-script",
    default="dataset/estimate_image_position_fixed.py",
    help="Global estimator script. Used only when no previous anchor is known.",
)
parser.add_argument(
    "--bfs-estimator-script",
    default="dataset/estimate_image_position_bfs_only_fixed.py",
    help="BFS-only estimator script. Used only after a previous anchor is known.",
)
parser.add_argument("-p", "--port", type=int, default=8010)
parser.add_argument("--temp-dir", default="dataset/server_temp")

# Estimator settings
parser.add_argument("--feature-max-size", type=int, default=1000)
parser.add_argument("--max-features", type=int, default=600)
parser.add_argument("--ratio", type=float, default=0.75)
parser.add_argument("--ransac", type=float, default=5.0)
parser.add_argument("--min-good", type=int, default=20)
parser.add_argument("--min-inliers", type=int, default=12)
parser.add_argument("--min-inlier-ratio", type=float, default=0.20)
parser.add_argument("--min-coverage", type=float, default=0.02)
parser.add_argument("--min-confidence", type=float, default=0.15)
parser.add_argument("--max-reprojection-error", type=float, default=10.0)
parser.add_argument("--workers", type=int, default=6)
parser.add_argument("--needed-matches", type=int, default=2)

# Global lock settings
parser.add_argument(
    "--global-candidate-steps",
    default="0",
    help="Candidate steps for first/global lock. 0 means full global search.",
)
parser.add_argument("--global-timeout", type=float, default=30.0)

# Local BFS settings
parser.add_argument("--bfs-depth", type=int, default=2)
parser.add_argument("--bfs-neighbor-limit", type=int, default=40)
parser.add_argument("--bfs-max-candidates", type=int, default=250)
parser.add_argument("--local-timeout", type=float, default=8.0)

# UI/loop settings
parser.add_argument(
    "--auto-every",
    type=float,
    default=2.0,
    help="Browser auto-estimate interval in seconds.",
)
parser.add_argument(
    "--jpeg-quality",
    type=int,
    default=92,
    help="Temporary extracted-frame JPEG quality.",
)
parser.add_argument(
    "--max-estimator-log-lines",
    type=int,
    default=12,
    help="Max estimator stdout lines copied into browser log.",
)

args = parser.parse_args()

VIDEO_PATH = Path(args.video).resolve()
DB_PATH = Path(args.database).resolve()
ROOT = Path(args.root).resolve()
GLOBAL_ESTIMATOR_SCRIPT = Path(args.global_estimator_script).resolve()
BFS_ESTIMATOR_SCRIPT = Path(args.bfs_estimator_script).resolve()
TEMP_DIR = Path(args.temp_dir).resolve()

if not VIDEO_PATH.exists():
    raise FileNotFoundError(f"Video does not exist: {VIDEO_PATH}")

if not DB_PATH.exists():
    raise FileNotFoundError(f"Database does not exist: {DB_PATH}")

if not GLOBAL_ESTIMATOR_SCRIPT.exists():
    raise FileNotFoundError(f"Global estimator does not exist: {GLOBAL_ESTIMATOR_SCRIPT}")

if not BFS_ESTIMATOR_SCRIPT.exists():
    raise FileNotFoundError(f"BFS estimator does not exist: {BFS_ESTIMATOR_SCRIPT}")

TEMP_DIR.mkdir(parents=True, exist_ok=True)

video_probe = cv2.VideoCapture(str(VIDEO_PATH))

if not video_probe.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

VIDEO_FPS = video_probe.get(cv2.CAP_PROP_FPS) or 30.0
VIDEO_FRAME_COUNT = int(video_probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
VIDEO_WIDTH = int(video_probe.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
VIDEO_HEIGHT = int(video_probe.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
video_probe.release()

CLIENT_ABORT_ERRORS = (
    ConnectionAbortedError,
    ConnectionResetError,
    BrokenPipeError,
)

state_lock = threading.Lock()
estimate_thread = None

state = {
    "estimating": False,
    "error": None,
    "video": str(VIDEO_PATH),
    "fps": VIDEO_FPS,
    "frame_count": VIDEO_FRAME_COUNT,
    "width": VIDEO_WIDTH,
    "height": VIDEO_HEIGHT,
    "last_known_path": None,
    "last_estimate": None,
    "last_requested_time": None,
    "last_requested_frame": None,
    "positions": [],
    "logs": [],
}


HTML = r'''<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Current-frame drone locator</title>

  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">

  <style>
    * { box-sizing: border-box; }

    body {
      margin: 0;
      background: #111;
      color: #eee;
      font-family: Arial, sans-serif;
      overflow: hidden;
    }

    header {
      height: 70px;
      background: #1b1b1b;
      border-bottom: 1px solid #333;
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 10px 14px;
      flex-wrap: wrap;
    }

    button, input {
      background: #222;
      color: #eee;
      border: 1px solid #444;
      border-radius: 7px;
      padding: 8px 10px;
    }

    button { cursor: pointer; }
    button:hover { border-color: #70a7ff; }

    .layout {
      height: calc(100vh - 70px);
      display: grid;
      grid-template-columns: 1fr 1fr;
    }

    .left, .right {
      min-width: 0;
      min-height: 0;
      position: relative;
    }

    .left {
      background: #050505;
      border-right: 1px solid #333;
      display: grid;
      grid-template-rows: 1fr 190px;
    }

    video {
      width: 100%;
      height: 100%;
      object-fit: contain;
      background: #050505;
    }

    #map {
      width: 100%;
      height: 100%;
      background: #222;
    }

    .panel {
      border-top: 1px solid #333;
      background: #181818;
      overflow: auto;
      padding: 10px;
      font-size: 13px;
      line-height: 1.5;
    }

    .status { color: #aaa; font-size: 13px; }
    .good { color: #72d38a; }
    .bad { color: #ff7777; }
    .warn { color: #ffcc66; }

    .floating {
      position: absolute;
      top: 12px;
      left: 12px;
      z-index: 500;
      background: rgba(20, 20, 20, 0.88);
      border: 1px solid #333;
      border-radius: 8px;
      padding: 10px;
      font-size: 13px;
      line-height: 1.5;
      max-width: 420px;
    }

    code {
      background: #222;
      padding: 2px 5px;
      border-radius: 4px;
    }

    .log {
      color: #aaa;
      white-space: pre-wrap;
    }
  </style>
</head>

<body>
<header>
  <button onclick="estimateCurrentFrame()">Estimate current frame</button>
  <button id="autoButton" onclick="toggleAuto()">Start auto</button>
  <button onclick="clearLock()">Clear lock</button>
  <button onclick="refreshNow()">Refresh</button>

  <label class="status">
    auto every
    <input id="autoEvery" type="number" value="AUTO_EVERY_PLACEHOLDER" min="0.5" step="0.5" style="width:72px;">
    sec
  </label>

  <span class="status" id="status">Loading...</span>
</header>

<div class="layout">
  <section class="left">
    <video id="video" controls muted>
      <source src="/video" type="video/mp4">
    </video>

    <div class="panel">
      <div><b>Last estimate</b></div>
      <div id="details" class="status">No estimate yet.</div>
      <div style="margin-top:8px;"><b>Logs</b></div>
      <div id="logs" class="log"></div>
    </div>
  </section>

  <section class="right">
    <div id="map"></div>

    <div class="floating">
      <div><b>Map path</b></div>
      <div id="mapInfo" class="status">Waiting for positions...</div>
      <div class="status" style="margin-top:8px;">
        This compares the current displayed video frame.<br>
        No lock = global estimator.<br>
        Lock exists = BFS-only estimator.
      </div>
    </div>
  </section>
</div>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>

<script>
const video = document.getElementById("video");
const statusEl = document.getElementById("status");
const detailsEl = document.getElementById("details");
const logsEl = document.getElementById("logs");
const mapInfoEl = document.getElementById("mapInfo");
const autoButton = document.getElementById("autoButton");
const autoEveryInput = document.getElementById("autoEvery");

const map = L.map("map").setView([32.1, 35.2], 15);

L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 22,
  attribution: "&copy; OpenStreetMap contributors"
}).addTo(map);

let pathLine = L.polyline([], { weight: 4 }).addTo(map);
let latestMarker = null;
let hasFitMap = false;
let autoTimer = null;

function n(value, digits = 6) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toFixed(digits);
}

async function getJson(url, options = {}) {
  const res = await fetch(url, options);

  if (!res.ok) {
    throw new Error(await res.text());
  }

  return await res.json();
}

async function estimateCurrentFrame() {
  await getJson("/api/estimate-current", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      time: video.currentTime
    })
  });

  await refreshNow();
}

function toggleAuto() {
  if (autoTimer) {
    clearInterval(autoTimer);
    autoTimer = null;
    autoButton.textContent = "Start auto";
    return;
  }

  const everySeconds = Math.max(0.5, Number(autoEveryInput.value || 2));

  autoTimer = setInterval(async () => {
    try {
      const s = await getJson("/api/state");

      if (!s.estimating && !video.paused && !video.ended) {
        await estimateCurrentFrame();
      } else {
        await refreshNow();
      }
    } catch (err) {
      console.error(err);
    }
  }, everySeconds * 1000);

  autoButton.textContent = "Stop auto";
}

async function clearLock() {
  await getJson("/api/clear-lock", {method: "POST"});
  await refreshNow();
}

async function refreshNow() {
  const s = await getJson("/api/state");

  statusEl.innerHTML =
    `estimating=${s.estimating} | positions=${s.positions.length} | lock=${s.last_known_path ? "yes" : "no"}`;

  if (s.error) {
    statusEl.innerHTML += ` | <span class="bad">${s.error}</span>`;
  }

  logsEl.textContent = (s.logs || []).slice(-10).join("\n");

  if (s.last_estimate) {
    const e = s.last_estimate;

    detailsEl.innerHTML = `
      mode: <b>${e.mode}</b><br>
      video time: <b>${n(e.video_time, 2)}s</b><br>
      frame: <b>${e.source_frame}</b><br>
      lat: <span class="good">${n(e.estimated_lat, 8)}</span><br>
      lon: <span class="good">${n(e.estimated_lon, 8)}</span><br>
      alt: ${e.estimated_alt === null ? "-" : n(e.estimated_alt, 2) + "m"}<br>
      confidence: ${n(e.confidence, 3)}<br>
      used matches: ${e.used_matches}/${e.total_matches}<br>
      checked anchors: ${e.checked_anchors}/${e.anchors_in_db}<br>
      last anchor: <code>${e.last_known_path || "-"}</code>
    `;
  }

  const points = (s.positions || [])
    .filter(p => p.estimated_lat !== null && p.estimated_lon !== null)
    .map(p => [p.estimated_lat, p.estimated_lon]);

  pathLine.setLatLngs(points);

  if (points.length > 0) {
    const latest = points[points.length - 1];

    if (!latestMarker) {
      latestMarker = L.circleMarker(latest, {
        radius: 8,
        fillOpacity: 0.9,
        color: "red"
      }).addTo(map);
    } else {
      latestMarker.setLatLng(latest);
    }

    if (!hasFitMap) {
      map.setView(latest, 17);
      hasFitMap = true;
    }

    mapInfoEl.innerHTML = `
      points: <b>${points.length}</b><br>
      latest: ${n(latest[0], 7)}, ${n(latest[1], 7)}
    `;
  }
}

setInterval(refreshNow, 1500);
refreshNow();
</script>
</body>
</html>
'''.replace("AUTO_EVERY_PLACEHOLDER", str(args.auto_every))


def add_log(text):
    with state_lock:
        timestamp = time.strftime("%H:%M:%S")
        state["logs"].append(f"[{timestamp}] {text}")
        state["logs"] = state["logs"][-120:]


def json_response(handler, data, status=200):
    raw = json.dumps(data).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(raw)))
    handler.end_headers()
    handler.wfile.write(raw)


def text_response(handler, text, status=400):
    raw = text.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "text/plain; charset=utf-8")
    handler.send_header("Content-Length", str(len(raw)))
    handler.end_headers()
    handler.wfile.write(raw)


def html_response(handler):
    raw = HTML.encode("utf-8")
    handler.send_response(200)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Content-Length", str(len(raw)))
    handler.end_headers()
    handler.wfile.write(raw)


def stream_file(handler, path):
    if not path.exists():
        text_response(handler, "file not found", 404)
        return

    content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    size = path.stat().st_size
    range_header = handler.headers.get("Range")

    try:
        if range_header:
            range_value = range_header.replace("bytes=", "")
            start_text, end_text = range_value.split("-", 1)
            start = int(start_text) if start_text else 0
            end = int(end_text) if end_text else size - 1
            end = min(end, size - 1)

            if start > end:
                raise ValueError("bad range")

            handler.send_response(206)
            handler.send_header("Content-Type", content_type)
            handler.send_header("Accept-Ranges", "bytes")
            handler.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            handler.send_header("Content-Length", str(end - start + 1))
            handler.end_headers()

            with path.open("rb") as file:
                file.seek(start)
                remaining = end - start + 1

                while remaining > 0:
                    chunk = file.read(min(1024 * 512, remaining))

                    if not chunk:
                        break

                    handler.wfile.write(chunk)
                    remaining -= len(chunk)

            return

        handler.send_response(200)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Content-Length", str(size))
        handler.send_header("Accept-Ranges", "bytes")
        handler.end_headers()

        with path.open("rb") as file:
            while True:
                chunk = file.read(1024 * 512)

                if not chunk:
                    break

                handler.wfile.write(chunk)

    except CLIENT_ABORT_ERRORS:
        return


def extract_video_frame(video_time_seconds):
    frame_number = max(1, int(video_time_seconds * VIDEO_FPS) + 1)

    if VIDEO_FRAME_COUNT > 0:
        frame_number = min(frame_number, VIDEO_FRAME_COUNT)

    capture = cv2.VideoCapture(str(VIDEO_PATH))

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
    success, frame = capture.read()
    capture.release()

    if not success:
        raise RuntimeError(f"Could not read video frame: {frame_number}")

    frame_path = TEMP_DIR / f"current_frame_{frame_number}.jpg"
    cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])

    return frame_number, frame_path


def build_estimator_command(frame_path, output_json, last_known_path):
    is_local = bool(last_known_path)

    if is_local:
        mode = "LOCAL-BFS"
        script_path = BFS_ESTIMATOR_SCRIPT
        timeout_seconds = args.local_timeout

        command = [
            sys.executable,
            str(script_path),
            "-i",
            str(frame_path),
            "-db",
            str(DB_PATH),
            "-r",
            str(ROOT),
            "--last-known-path",
            last_known_path,
            "--feature-max-size",
            str(args.feature_max_size),
            "--max-features",
            str(args.max_features),
            "--ratio",
            str(args.ratio),
            "--ransac",
            str(args.ransac),
            "--min-good",
            str(args.min_good),
            "--min-inliers",
            str(args.min_inliers),
            "--min-inlier-ratio",
            str(args.min_inlier_ratio),
            "--min-coverage",
            str(args.min_coverage),
            "--min-confidence",
            str(args.min_confidence),
            "--max-reprojection-error",
            str(args.max_reprojection_error),
            "--workers",
            str(args.workers),
            "--needed-matches",
            str(args.needed_matches),
            "--bfs-depth",
            str(args.bfs_depth),
            "--bfs-neighbor-limit",
            str(args.bfs_neighbor_limit),
            "--bfs-max-candidates",
            str(args.bfs_max_candidates),
            "-o",
            str(output_json),
        ]

    else:
        mode = "GLOBAL"
        script_path = GLOBAL_ESTIMATOR_SCRIPT
        timeout_seconds = args.global_timeout

        command = [
            sys.executable,
            str(script_path),
            "-i",
            str(frame_path),
            "-db",
            str(DB_PATH),
            "-r",
            str(ROOT),
            "--feature-max-size",
            str(args.feature_max_size),
            "--max-features",
            str(args.max_features),
            "--ratio",
            str(args.ratio),
            "--ransac",
            str(args.ransac),
            "--min-good",
            str(args.min_good),
            "--min-inliers",
            str(args.min_inliers),
            "--min-inlier-ratio",
            str(args.min_inlier_ratio),
            "--min-coverage",
            str(args.min_coverage),
            "--min-confidence",
            str(args.min_confidence),
            "--max-reprojection-error",
            str(args.max_reprojection_error),
            "--workers",
            str(args.workers),
            "--needed-matches",
            str(args.needed_matches),
            "--candidate-steps",
            args.global_candidate_steps,
            "-o",
            str(output_json),
        ]

    return mode, timeout_seconds, command


def run_estimator(frame_path, output_json, last_known_path):
    mode, timeout_seconds, command = build_estimator_command(
        frame_path,
        output_json,
        last_known_path,
    )

    add_log(f"{mode} estimator start | timeout={timeout_seconds:.1f}s | frame={frame_path.name}")

    output_lines = []
    copied_lines = 0

    process = subprocess.Popen(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )

    def read_stdout():
        nonlocal copied_lines

        try:
            for line in process.stdout:
                line = line.rstrip()

                if not line:
                    continue

                output_lines.append(line)

                if copied_lines < args.max_estimator_log_lines:
                    add_log(f"est: {line[:260]}")
                    copied_lines += 1

        except Exception:
            return

    reader = threading.Thread(target=read_stdout, daemon=True)
    reader.start()

    start_time = time.time()

    while True:
        return_code = process.poll()

        if return_code is not None:
            break

        if time.time() - start_time > timeout_seconds:
            try:
                process.kill()
            except Exception:
                pass

            raise RuntimeError(
                f"{mode} estimator timed out after {timeout_seconds:.1f}s. "
                f"Last output:\\n" + "\\n".join(output_lines[-20:])
            )

        time.sleep(0.15)

    reader.join(timeout=1.0)

    if process.returncode != 0:
        raise RuntimeError(
            f"{mode} estimator failed with exit code {process.returncode}. "
            f"Last output:\\n" + "\\n".join(output_lines[-30:])
        )

    if not output_json.exists():
        raise RuntimeError(f"{mode} estimator finished but did not create output JSON.")

    add_log(f"{mode} estimator done in {time.time() - start_time:.1f}s")
    return mode, json.loads(output_json.read_text(encoding="utf-8"))


def normalize_estimate_payload(payload, mode, source_frame, video_time):
    top_matches = payload.get("top_matches") or []
    best_path = top_matches[0].get("anchor_path") if top_matches else None

    return {
        "mode": mode,
        "video_time": video_time,
        "source_frame": source_frame,
        "estimated_lat": payload.get("estimated_lat"),
        "estimated_lon": payload.get("estimated_lon"),
        "estimated_alt": payload.get("estimated_alt"),
        "confidence": payload.get("confidence"),
        "used_matches": payload.get("used_matches"),
        "total_matches": payload.get("total_matches"),
        "checked_anchors": payload.get("checked_anchors"),
        "anchors_in_db": payload.get("anchors_in_db"),
        "last_known_path": best_path,
        "top_matches": top_matches[:5],
    }


def estimate_current_worker(video_time):
    with state_lock:
        if state["estimating"]:
            return

        state["estimating"] = True
        state["error"] = None
        last_known_path = state["last_known_path"]

    try:
        frame_number, frame_path = extract_video_frame(video_time)
        output_json = TEMP_DIR / f"estimate_current_{frame_number}.json"

        add_log(
            f"Comparing CURRENT frame {frame_number} at {video_time:.2f}s "
            f"last={last_known_path or 'GLOBAL'}"
        )

        mode, payload = run_estimator(frame_path, output_json, last_known_path)
        estimate = normalize_estimate_payload(payload, mode, frame_number, video_time)

        if estimate["last_known_path"]:
            last_known_path = estimate["last_known_path"]

        with state_lock:
            state["last_known_path"] = last_known_path
            state["last_estimate"] = estimate
            state["positions"].append(estimate)
            state["last_requested_time"] = video_time
            state["last_requested_frame"] = frame_number

        add_log(
            f"OK {mode} frame {frame_number}: "
            f"{estimate['estimated_lat']:.7f}, {estimate['estimated_lon']:.7f} "
            f"conf={estimate['confidence']:.2f}"
        )

    except Exception as error:
        add_log(f"FAILED current frame: {error}")

        with state_lock:
            state["error"] = str(error)

            # If local BFS failed, remove the lock so the next current-frame request
            # can use the reliable global estimator.
            if last_known_path:
                state["last_known_path"] = None

    finally:
        with state_lock:
            state["estimating"] = False


def start_estimate_current(video_time):
    global estimate_thread

    with state_lock:
        if state["estimating"]:
            return False

    estimate_thread = threading.Thread(
        target=estimate_current_worker,
        args=(float(video_time),),
        daemon=True,
    )
    estimate_thread.start()
    return True


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        route = parsed.path

        try:
            if route == "/":
                html_response(self)
            elif route == "/video":
                stream_file(self, VIDEO_PATH)
            elif route == "/api/state":
                with state_lock:
                    json_response(self, state)
            else:
                text_response(self, "not found", 404)

        except CLIENT_ABORT_ERRORS:
            return

        except Exception as error:
            try:
                text_response(self, str(error), 500)
            except CLIENT_ABORT_ERRORS:
                return

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        route = parsed.path

        try:
            if route == "/api/estimate-current":
                content_length = int(self.headers.get("Content-Length", "0") or "0")
                body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
                payload = json.loads(body or "{}")

                video_time = float(payload.get("time", 0.0))
                started = start_estimate_current(video_time)
                json_response(self, {"started": started})

            elif route == "/api/clear-lock":
                with state_lock:
                    state["last_known_path"] = None
                    state["error"] = None
                add_log("Lock cleared")
                json_response(self, {"cleared": True})

            else:
                text_response(self, "not found", 404)

        except CLIENT_ABORT_ERRORS:
            return

        except Exception as error:
            try:
                text_response(self, str(error), 500)
            except CLIENT_ABORT_ERRORS:
                return

    def log_message(self, *args):
        return


print(f"Video:     {VIDEO_PATH}")
print(f"DB:        {DB_PATH}")
print(f"Root:      {ROOT}")
print(f"Global:    {GLOBAL_ESTIMATOR_SCRIPT}")
print(f"BFS:       {BFS_ESTIMATOR_SCRIPT}")
print(f"Temp:      {TEMP_DIR}")
print(f"FPS:       {VIDEO_FPS}")
print(f"Frames:    {VIDEO_FRAME_COUNT}")
print(f"Open:      http://127.0.0.1:{args.port}")
print()
print("Press Ctrl+C to stop.")

server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
server.serve_forever()
