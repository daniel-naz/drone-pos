import argparse
import json
import mimetypes
import sqlite3
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import urllib.parse

import cv2


parser = argparse.ArgumentParser(
    description="Video + map viewer that estimates drone position frame-by-frame."
)

parser.add_argument("-v", "--video", required=True, help="Path to the video file.")
parser.add_argument("-db", "--database", default="graph.db", help="Path to graph.db.")
parser.add_argument("-r", "--root", default=".", help="Project root for DB image paths.")
parser.add_argument(
    "--global-estimator-script",
    default="dataset/estimate_image_position.py",
    help="Path to the reliable global estimator script. Used only when no previous anchor is known.",
)
parser.add_argument(
    "--bfs-estimator-script",
    default="dataset/estimate_image_position_bfs_only.py",
    help="Path to the BFS-only estimator script. Used only after a previous anchor is known.",
)
parser.add_argument("-p", "--port", type=int, default=8010)

parser.add_argument("--sample-every", type=int, default=30, help="Estimate every N video frames.")

parser.add_argument(
    "--failed-skip-multiplier",
    type=int,
    default=5,
    help=(
        "When a frame cannot be localized, skip this many sample intervals before trying again. "
        "Example: sample_every=30 and multiplier=5 means skip 150 frames after failure."
    ),
)

parser.add_argument(
    "--failed-skip-frames",
    type=int,
    default=0,
    help=(
        "Exact number of frames to skip after localization failure. "
        "If 0, uses sample_every * failed_skip_multiplier."
    ),
)

parser.add_argument(
    "--start-time",
    type=float,
    default=0.0,
    help="Start processing at this video time in seconds. Overrides --start-frame when > 0.",
)

parser.add_argument("--temp-dir", default="dataset/server_temp")
parser.add_argument("--start-frame", type=int, default=1)
parser.add_argument("--stop-frame", type=int, default=0, help="0 means until video ends.")

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
parser.add_argument("--needed-matches", type=int, default=3)

# Local BFS settings
parser.add_argument("--bfs-depth", type=int, default=2)
parser.add_argument("--bfs-neighbor-limit", type=int, default=40)
parser.add_argument("--bfs-max-candidates", type=int, default=250)

# Global fallback / realtime settings
parser.add_argument(
    "--candidate-steps",
    default="0",
    help=(
        "Backward-compatible global candidate steps. Default 0 means use the reliable "
        "all-anchor global search for the first lock."
    ),
)

parser.add_argument(
    "--global-candidate-steps",
    default=None,
    help=(
        "Candidate steps used when there is no last-known anchor. "
        "If omitted, uses --candidate-steps. Use 0 if full global SIFT is your reliable method."
    ),
)

parser.add_argument(
    "--local-candidate-steps",
    default="100,300",
    help="Candidate steps passed during local/BFS mode if global fallback is allowed.",
)

parser.add_argument(
    "--allow-local-global-fallback",
    action="store_true",
    help=(
        "When a last-known anchor exists, allow the estimator to fall back to global search "
        "inside the same subprocess. Off by default for realtime behavior."
    ),
)

parser.add_argument(
    "--local-timeout",
    type=float,
    default=12.0,
    help="Timeout in seconds for a local/BFS estimate after lock.",
)

parser.add_argument(
    "--global-timeout",
    type=float,
    default=25.0,
    help="Timeout in seconds for a global first-lock estimate.",
)

parser.add_argument(
    "--heartbeat-seconds",
    type=float,
    default=3.0,
    help="Log a heartbeat while the estimator subprocess is still running.",
)

parser.add_argument(
    "--max-estimator-log-lines",
    type=int,
    default=12,
    help="Maximum estimator stdout lines to copy into the browser logs per sampled frame.",
)

parser.add_argument(
    "--local-only",
    action="store_true",
    help=(
        "Deprecated alias. Local mode is already local-only by default unless "
        "--allow-local-global-fallback is set."
    ),
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
    raise FileNotFoundError(f"Global estimator script does not exist: {GLOBAL_ESTIMATOR_SCRIPT}")

if not BFS_ESTIMATOR_SCRIPT.exists():
    raise FileNotFoundError(f"BFS estimator script does not exist: {BFS_ESTIMATOR_SCRIPT}")

TEMP_DIR.mkdir(parents=True, exist_ok=True)

video_probe = cv2.VideoCapture(str(VIDEO_PATH))

if not video_probe.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

VIDEO_FPS = video_probe.get(cv2.CAP_PROP_FPS) or 30.0
VIDEO_FRAME_COUNT = int(video_probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
VIDEO_WIDTH = int(video_probe.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
VIDEO_HEIGHT = int(video_probe.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
video_probe.release()


state_lock = threading.Lock()
processor_thread = None
stop_requested = False

CLIENT_ABORT_ERRORS = (
    ConnectionAbortedError,
    ConnectionResetError,
    BrokenPipeError,
)

state = {
    "running": False,
    "done": False,
    "error": None,
    "video": str(VIDEO_PATH),
    "fps": VIDEO_FPS,
    "frame_count": VIDEO_FRAME_COUNT,
    "width": VIDEO_WIDTH,
    "height": VIDEO_HEIGHT,
    "sample_every": args.sample_every,
    "failed_skip_multiplier": args.failed_skip_multiplier,
    "failed_skip_frames": args.failed_skip_frames,
    "start_time": args.start_time,
    "current_frame": 0,
    "processed_count": 0,
    "last_known_path": None,
    "last_estimate": None,
    "positions": [],
    "logs": [],
}


HTML = r'''
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Drone video position viewer</title>

  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">

  <style>
    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      background: #111;
      color: #eee;
      font-family: Arial, sans-serif;
      overflow: hidden;
    }

    header {
      height: 64px;
      background: #1b1b1b;
      border-bottom: 1px solid #333;
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 10px 14px;
    }

    button {
      background: #222;
      color: #eee;
      border: 1px solid #444;
      border-radius: 7px;
      padding: 8px 10px;
      cursor: pointer;
    }

    button:hover {
      border-color: #70a7ff;
    }

    .layout {
      height: calc(100vh - 64px);
      display: grid;
      grid-template-columns: 1fr 1fr;
    }

    .left,
    .right {
      min-width: 0;
      min-height: 0;
      position: relative;
    }

    .left {
      background: #050505;
      border-right: 1px solid #333;
      display: grid;
      grid-template-rows: 1fr 170px;
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

    .status {
      color: #aaa;
      font-size: 13px;
    }

    .good {
      color: #72d38a;
    }

    .bad {
      color: #ff7777;
    }

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
      max-width: 390px;
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
  <button onclick="startProcessing()">Start processing</button>
  <button onclick="stopProcessing()">Stop</button>
  <button onclick="refreshNow()">Refresh now</button>

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
        Blue path = estimated drone route.<br>
        Red dot = latest estimate.
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

const map = L.map("map").setView([32.1, 35.2], 15);

L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 22,
  attribution: "&copy; OpenStreetMap contributors"
}).addTo(map);

let pathLine = L.polyline([], {
  weight: 4
}).addTo(map);

let latestMarker = null;
let hasFitMap = false;

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

async function startProcessing() {
  await getJson("/api/start", {method: "POST"});
  await refreshNow();
}

async function stopProcessing() {
  await getJson("/api/stop", {method: "POST"});
  await refreshNow();
}

async function refreshNow() {
  const s = await getJson("/api/state");

  const progress =
    s.frame_count > 0
      ? `${s.current_frame}/${s.frame_count}`
      : `${s.current_frame}`;

  statusEl.innerHTML =
    `running=${s.running} | done=${s.done} | frame=${progress} | estimates=${s.positions.length}`;

  if (s.error) {
    statusEl.innerHTML += ` | <span class="bad">${s.error}</span>`;
  }

  logsEl.textContent = (s.logs || []).slice(-8).join("\n");

  if (s.last_estimate) {
    const e = s.last_estimate;

    detailsEl.innerHTML = `
      frame: <b>${e.source_frame}</b><br>
      lat: <span class="good">${n(e.estimated_lat, 8)}</span><br>
      lon: <span class="good">${n(e.estimated_lon, 8)}</span><br>
      alt: ${e.estimated_alt === null ? "-" : n(e.estimated_alt, 2) + "m"}<br>
      confidence: ${n(e.confidence, 3)}<br>
      used matches: ${e.used_matches}/${e.total_matches}<br>
      checked anchors: ${e.checked_anchors}/${e.anchors_in_db}<br>
      last anchor: <code>${e.last_known_path || "-"}</code>
    `;

    if (video.duration && s.fps) {
      const targetTime = Math.max(0, (e.source_frame - 1) / s.fps);
      if (Math.abs(video.currentTime - targetTime) > 1.0) {
        video.currentTime = targetTime;
      }
    }
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
'''


def add_log(text):
    with state_lock:
        timestamp = time.strftime("%H:%M:%S")
        state["logs"].append(f"[{timestamp}] {text}")
        state["logs"] = state["logs"][-80:]


def safe_json_response(handler, data, status=200):
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


def run_estimator(frame_path, output_json, last_known_path):
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

        log_extra = f"last={last_known_path}"

    else:
        mode = "GLOBAL"
        script_path = GLOBAL_ESTIMATOR_SCRIPT
        timeout_seconds = args.global_timeout
        candidate_steps = args.global_candidate_steps or args.candidate_steps

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
            candidate_steps,
            "-o",
            str(output_json),
        ]

        log_extra = f"steps={candidate_steps}"

    add_log(
        f"{mode} estimator start | timeout={timeout_seconds:.1f}s | "
        f"{log_extra} | frame={frame_path.name}"
    )

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
    last_heartbeat = start_time

    while True:
        return_code = process.poll()

        if return_code is not None:
            break

        now = time.time()

        if now - start_time > timeout_seconds:
            try:
                process.kill()
            except Exception:
                pass

            raise RuntimeError(
                f"{mode} estimator timed out after {timeout_seconds:.1f}s. "
                f"Last output:\\n" + "\\n".join(output_lines[-20:])
            )

        if now - last_heartbeat >= args.heartbeat_seconds:
            add_log(f"{mode} estimator still running... {now - start_time:.1f}s")
            last_heartbeat = now

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

    return json.loads(output_json.read_text(encoding="utf-8"))

def normalize_estimate_payload(payload, source_frame):
    best_path = None
    top_matches = payload.get("top_matches") or []

    if top_matches:
        best_path = top_matches[0].get("anchor_path")

    return {
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


def processing_loop():
    global stop_requested

    add_log("Processor started")

    with state_lock:
        state["running"] = True
        state["done"] = False
        state["error"] = None

    video = cv2.VideoCapture(str(VIDEO_PATH))

    if not video.isOpened():
        with state_lock:
            state["running"] = False
            state["error"] = f"Could not open video: {VIDEO_PATH}"
        return

    if args.start_time > 0:
        current_frame = max(1, int(args.start_time * VIDEO_FPS) + 1)
    else:
        current_frame = max(1, args.start_frame)

    stop_frame = args.stop_frame if args.stop_frame > 0 else VIDEO_FRAME_COUNT

    last_known_path = None

    with state_lock:
        if state["last_known_path"]:
            last_known_path = state["last_known_path"]

    while True:
        with state_lock:
            if stop_requested:
                add_log("Stop requested")
                break

        if stop_frame > 0 and current_frame > stop_frame:
            break

        video.set(cv2.CAP_PROP_POS_FRAMES, current_frame - 1)
        success, frame = video.read()

        if not success:
            break

        frame_path = TEMP_DIR / f"frame_{current_frame}.jpg"
        output_json = TEMP_DIR / f"estimate_{current_frame}.json"

        cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])

        with state_lock:
            state["current_frame"] = current_frame

        frame_step = max(1, args.sample_every)

        try:
            add_log(f"Estimating frame {current_frame} last={last_known_path or 'GLOBAL'}")

            payload = run_estimator(frame_path, output_json, last_known_path)
            estimate = normalize_estimate_payload(payload, current_frame)

            if estimate["last_known_path"]:
                last_known_path = estimate["last_known_path"]

            with state_lock:
                state["last_known_path"] = last_known_path
                state["last_estimate"] = estimate
                state["positions"].append(estimate)
                state["processed_count"] += 1

            lat = estimate.get("estimated_lat")
            lon = estimate.get("estimated_lon")
            conf = estimate.get("confidence")

            add_log(
                f"OK frame {current_frame}: "
                f"{lat:.7f}, {lon:.7f} conf={conf:.2f}"
            )

        except Exception as error:
            add_log(f"FAILED frame {current_frame}: {error}")

            with state_lock:
                state["error"] = str(error)

            # If local/global search failed, lose lock and jump ahead.
            # This is useful for dead-time at the start of a video where the camera is not aerial yet.
            last_known_path = None

            if args.failed_skip_frames > 0:
                frame_step = max(1, args.failed_skip_frames)
            else:
                frame_step = max(
                    max(1, args.sample_every),
                    max(1, args.sample_every) * max(1, args.failed_skip_multiplier),
                )

            add_log(f"Skipping ahead {frame_step} frames after failure")

        current_frame += frame_step

    video.release()

    with state_lock:
        state["running"] = False
        state["done"] = True

    add_log("Processor finished")


def start_processor():
    global processor_thread, stop_requested

    with state_lock:
        if state["running"]:
            return False

        state["done"] = False
        state["error"] = None

    stop_requested = False
    processor_thread = threading.Thread(target=processing_loop, daemon=True)
    processor_thread.start()

    return True


def stop_processor():
    global stop_requested
    stop_requested = True
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
                    safe_json_response(self, state)

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
            if route == "/api/start":
                started = start_processor()
                safe_json_response(self, {"started": started})

            elif route == "/api/stop":
                stopped = stop_processor()
                safe_json_response(self, {"stopped": stopped})

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
print(f"Global estimator: {GLOBAL_ESTIMATOR_SCRIPT}")
print(f"BFS estimator:    {BFS_ESTIMATOR_SCRIPT}")
print(f"Temp:             {TEMP_DIR}")
print(f"FPS:       {VIDEO_FPS}")
print(f"Frames:    {VIDEO_FRAME_COUNT}")
print(f"Sample:    every {args.sample_every} frames")
print(f"Fail skip: {args.failed_skip_frames if args.failed_skip_frames > 0 else args.sample_every * args.failed_skip_multiplier} frames")
print(f"Global:    steps={args.global_candidate_steps or args.candidate_steps}, timeout={args.global_timeout}s")
print(f"Local:     timeout={args.local_timeout}s, local-only={not args.allow_local_global_fallback}")
print(f"Open:      http://127.0.0.1:{args.port}")
print()
print("Press Ctrl+C to stop.")

server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
server.serve_forever()
