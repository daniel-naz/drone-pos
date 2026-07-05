import argparse
import json
import math
import mimetypes
import sqlite3
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


parser = argparse.ArgumentParser(description="View graph.db in browser")
parser.add_argument("-db", "--database", default="graph.db", help="Path to graph.db")
parser.add_argument(
    "-r",
    "--root",
    default=".",
    help="Project root folder. Image paths in the DB are resolved from here.",
)
parser.add_argument("-p", "--port", type=int, default=8000, help="Server port")
args = parser.parse_args()

DB_PATH = Path(args.database).resolve()
ROOT = Path(args.root).resolve()

if not DB_PATH.exists():
    raise FileNotFoundError(f"Database does not exist: {DB_PATH}")

if not ROOT.exists():
    raise FileNotFoundError(f"Root folder does not exist: {ROOT}")


HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>graph.db viewer</title>

  <style>
    :root {
      --bg: #111;
      --panel: #181818;
      --panel2: #222;
      --border: #333;
      --text: #eee;
      --muted: #aaa;
      --accent: #70a7ff;
      --good: #72d38a;
      --blob: #ffcc66;
      --bad: #ff7777;
    }

    * {
      box-sizing: border-box;
    }

    body {
      font-family: Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      margin: 0;
    }

    header {
      padding: 16px;
      background: #1b1b1b;
      position: sticky;
      top: 0;
      z-index: 10;
      border-bottom: 1px solid var(--border);
    }

    h1 {
      margin: 0 0 12px;
      font-size: 22px;
    }

    main {
      padding: 16px;
    }

    button,
    select,
    input {
      background: var(--panel2);
      color: var(--text);
      border: 1px solid #444;
      padding: 7px;
      border-radius: 6px;
      margin: 3px;
    }

    button {
      cursor: pointer;
    }

    button:hover {
      border-color: var(--accent);
    }

    .tabs {
      margin-top: 12px;
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }

    .tabs button.active {
      background: #2f3a52;
      border-color: var(--accent);
    }

    .info {
      margin: 12px 0;
      color: var(--muted);
    }

    .table-wrap {
      overflow: auto;
      max-height: calc(100vh - 245px);
      border: 1px solid var(--border);
      border-radius: 10px;
      background: var(--panel);
    }

    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }

    th,
    td {
      border-bottom: 1px solid var(--border);
      padding: 7px;
      white-space: nowrap;
      max-width: 420px;
      overflow: hidden;
      text-overflow: ellipsis;
      vertical-align: middle;
    }

    th {
      position: sticky;
      top: 0;
      background: var(--panel2);
      z-index: 2;
    }

    tr:hover {
      background: var(--panel2);
    }

    .blob {
      color: var(--blob);
    }

    .null {
      color: #777;
    }

    .path {
      color: var(--good);
    }

    .pager {
      margin: 12px 0;
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }

    code {
      background: var(--panel2);
      padding: 2px 5px;
      border-radius: 4px;
    }

    .thumb-cell {
      min-width: 180px;
      max-width: 230px;
      white-space: normal;
    }

    .thumb {
      width: 160px;
      height: 95px;
      object-fit: cover;
      display: block;
      border: 1px solid #444;
      border-radius: 6px;
      background: #050505;
      margin-bottom: 5px;
    }

    .thumb-path {
      font-size: 11px;
      color: var(--muted);
      overflow-wrap: anywhere;
      white-space: normal;
      max-width: 210px;
    }

    .bad {
      color: var(--bad);
    }

    .good {
      color: var(--good);
    }

    .small {
      color: var(--muted);
      font-size: 12px;
      margin-top: 6px;
    }


    .transform-cell {
      min-width: 190px;
      max-width: 230px;
      white-space: normal;
      font-family: Consolas, monospace;
      font-size: 12px;
      line-height: 1.35;
    }

    .transform-button {
      font-size: 12px;
      padding: 5px 7px;
      margin-top: 6px;
    }

    .modal-backdrop {
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.72);
      z-index: 100;
      align-items: center;
      justify-content: center;
      padding: 20px;
    }

    .modal {
      width: min(980px, 96vw);
      max-height: 92vh;
      overflow: auto;
      background: #181818;
      border: 1px solid #444;
      border-radius: 12px;
      box-shadow: 0 20px 80px rgba(0, 0, 0, 0.5);
    }

    .modal-header {
      position: sticky;
      top: 0;
      background: #222;
      border-bottom: 1px solid #333;
      padding: 12px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
    }

    .modal-body {
      padding: 12px;
    }

    .modal-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }

    .modal-grid img {
      width: 100%;
      max-height: 340px;
      object-fit: contain;
      background: #050505;
      border: 1px solid #333;
      border-radius: 8px;
    }

    .modal-path {
      color: var(--muted);
      font-size: 12px;
      overflow-wrap: anywhere;
      margin-bottom: 6px;
    }

    .transform-details {
      margin-top: 12px;
      display: grid;
      grid-template-columns: repeat(4, minmax(120px, 1fr));
      gap: 10px;
    }

    .detail-box {
      background: #202020;
      border: 1px solid #333;
      border-radius: 8px;
      padding: 10px;
    }

    .detail-box .label {
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 4px;
    }

    .detail-box .value {
      font-size: 18px;
      font-weight: bold;
    }

    @media (max-width: 900px) {
      .modal-grid,
      .transform-details {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>

<body>
<header>
  <h1>graph.db viewer</h1>

  <div>
    DB table:
    <select id="tableSelect"></select>

    Rows per page:
    <input id="limitInput" type="number" value="100" min="1" max="1000">

    <button onclick="reloadTable()">Reload</button>
  </div>

  <div id="tabs" class="tabs"></div>

  <div class="small">
    The <code>image_matches</code> table is shown as a joined view with thumbnails and homography transform values.
  </div>
</header>

<main>
  <div id="info" class="info">Loading...</div>

  <div class="pager">
    <button onclick="firstPage()">First</button>
    <button onclick="prevPage()">Prev</button>
    <span id="pageInfo">Page 1</span>
    <button onclick="nextPage()">Next</button>
    <button onclick="lastPage()">Last</button>
  </div>

  <div id="tableArea" class="table-wrap"></div>

  <div class="pager">
    <button onclick="firstPage()">First</button>
    <button onclick="prevPage()">Prev</button>
    <span id="pageInfoBottom">Page 1</span>
    <button onclick="nextPage()">Next</button>
    <button onclick="lastPage()">Last</button>
  </div>
</main>

<div id="transformModal" class="modal-backdrop" onclick="closeTransformModal(event)">
  <div class="modal" onclick="event.stopPropagation()">
    <div class="modal-header">
      <b>Image transform</b>
      <button onclick="closeTransformModal()">Close</button>
    </div>
    <div class="modal-body" id="transformModalBody"></div>
  </div>
</div>

<script>
let currentTable = null;
let currentPage = 0;
let totalRows = 0;
let tables = [];

const tableSelect = document.getElementById("tableSelect");
const limitInput = document.getElementById("limitInput");
const tabs = document.getElementById("tabs");
const info = document.getElementById("info");
const tableArea = document.getElementById("tableArea");
const pageInfo = document.getElementById("pageInfo");
const pageInfoBottom = document.getElementById("pageInfoBottom");
const transformModal = document.getElementById("transformModal");
const transformModalBody = document.getElementById("transformModalBody");

function esc(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function formatNumber(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toFixed(digits);
}

function cellClass(value) {
  if (value === null) return "null";
  if (typeof value === "string" && value.startsWith("[BLOB")) return "blob";

  if (typeof value === "string") {
    const lower = value.toLowerCase();

    if (
      lower.includes("dataset\\\\") ||
      lower.includes("dataset/") ||
      lower.endsWith(".jpeg") ||
      lower.endsWith(".jpg") ||
      lower.endsWith(".jpen") ||
      lower.endsWith(".png")
    ) {
      return "path";
    }
  }

  if (typeof value === "number" && value >= 0.7) return "good";

  return "";
}

function isImagePath(value) {
  if (typeof value !== "string") return false;

  const lower = value.toLowerCase();

  return (
    lower.endsWith(".jpg") ||
    lower.endsWith(".jpeg") ||
    lower.endsWith(".jpen") ||
    lower.endsWith(".png")
  );
}

function fileUrl(path) {
  return `/file?path=${encodeURIComponent(path)}`;
}

function renderTransformCell(row) {
  if (row.dx_px === null || row.dx_px === undefined) {
    return `<td class="null">No transform</td>`;
  }

  const rowJson = encodeURIComponent(JSON.stringify(row));

  return `
    <td class="transform-cell">
      dx: <b>${formatNumber(row.dx_px, 1)}px</b><br>
      dy: <b>${formatNumber(row.dy_px, 1)}px</b><br>
      move: <b>${formatNumber(row.move_px, 1)}px</b><br>
      scale: <b>${formatNumber(row.scale_a_to_b, 4)}x</b><br>
      rot: <b>${formatNumber(row.rotation_deg, 2)}°</b><br>
      <button class="transform-button" onclick="showTransformModalFromEncoded('${rowJson}')">
        View transform
      </button>
    </td>
  `;
}

function renderCell(col, value, row) {
  if (col === "transform") {
    return renderTransformCell(row);
  }

  if (value === null) {
    return `<td class="null">NULL</td>`;
  }

  if (
    (col === "image_a_path" || col === "image_b_path" || col === "path") &&
    isImagePath(value)
  ) {
    return `
      <td class="thumb-cell" title="${esc(value)}">
        <img class="thumb" src="${fileUrl(value)}" onerror="this.style.display='none'">
        <div class="thumb-path">${esc(value)}</div>
      </td>
    `;
  }

  return `
    <td class="${cellClass(value)}" title="${esc(value)}">
      ${esc(value)}
    </td>
  `;
}

function showTransformModalFromEncoded(encodedRow) {
  const row = JSON.parse(decodeURIComponent(encodedRow));
  showTransformModal(row);
}

function showTransformModal(row) {
  const imageA = row.image_a_path;
  const imageB = row.image_b_path;

  transformModalBody.innerHTML = `
    <div class="modal-grid">
      <div>
        <div class="modal-path"><b>A:</b> ${esc(imageA)}</div>
        <img src="${fileUrl(imageA)}">
      </div>
      <div>
        <div class="modal-path"><b>B:</b> ${esc(imageB)}</div>
        <img src="${fileUrl(imageB)}">
      </div>
    </div>

    <div class="transform-details">
      <div class="detail-box">
        <div class="label">Horizontal shift</div>
        <div class="value">${formatNumber(row.dx_px, 2)} px</div>
      </div>

      <div class="detail-box">
        <div class="label">Vertical shift</div>
        <div class="value">${formatNumber(row.dy_px, 2)} px</div>
      </div>

      <div class="detail-box">
        <div class="label">Total movement</div>
        <div class="value">${formatNumber(row.move_px, 2)} px</div>
      </div>

      <div class="detail-box">
        <div class="label">Scale A → B</div>
        <div class="value">${formatNumber(row.scale_a_to_b, 5)}x</div>
      </div>

      <div class="detail-box">
        <div class="label">Rotation</div>
        <div class="value">${formatNumber(row.rotation_deg, 3)}°</div>
      </div>

      <div class="detail-box">
        <div class="label">Confidence</div>
        <div class="value">${formatNumber(row.confidence, 3)}</div>
      </div>

      <div class="detail-box">
        <div class="label">Inlier ratio</div>
        <div class="value">${formatNumber(row.inlier_ratio, 3)}</div>
      </div>

      <div class="detail-box">
        <div class="label">Reprojection error</div>
        <div class="value">${formatNumber(row.mean_reprojection_error, 3)} px</div>
      </div>
    </div>

    <div class="small">
      dx/dy are measured by mapping the center of image A through the homography into image B,
      then comparing it to the center of image B. Scale/rotation are estimated locally around the image center.
    </div>
  `;

  transformModal.style.display = "flex";
}

function closeTransformModal(event) {
  if (event && event.target !== transformModal) return;
  transformModal.style.display = "none";
}

async function getJson(url) {
  const res = await fetch(url);

  if (!res.ok) {
    throw new Error(await res.text());
  }

  return await res.json();
}

async function loadTables() {
  tables = await getJson("/api/tables");

  tableSelect.innerHTML = tables
    .map(t => `<option value="${esc(t)}">${esc(t)}</option>`)
    .join("");

  tabs.innerHTML = tables
    .map(t => `<button onclick="selectTable('${esc(t)}')" id="tab-${esc(t)}">${esc(t)}</button>`)
    .join("");

  if (tables.includes("image_matches")) {
    selectTable("image_matches");
  } else if (tables.length > 0) {
    selectTable(tables[0]);
  } else {
    info.innerHTML = "No tables found.";
  }
}

function selectTable(name) {
  currentTable = name;
  currentPage = 0;
  tableSelect.value = name;

  for (const button of tabs.querySelectorAll("button")) {
    button.classList.remove("active");
  }

  const active = document.getElementById("tab-" + name);
  if (active) active.classList.add("active");

  loadTablePage();
}

function reloadTable() {
  currentPage = 0;
  currentTable = tableSelect.value;
  selectTable(currentTable);
}

function currentLimit() {
  return Math.max(1, Math.min(Number(limitInput.value || 100), 1000));
}

function maxPage() {
  return Math.max(0, Math.ceil(totalRows / currentLimit()) - 1);
}

async function loadTablePage() {
  if (!currentTable) return;

  const limit = currentLimit();
  const offset = currentPage * limit;

  const data = await getJson(
    `/api/table?name=${encodeURIComponent(currentTable)}&limit=${limit}&offset=${offset}`
  );

  totalRows = data.total;

  const max = Math.max(0, Math.ceil(totalRows / limit) - 1);

  if (currentPage > max) {
    currentPage = max;
    return loadTablePage();
  }

  const start = totalRows === 0 ? 0 : offset + 1;
  const end = Math.min(offset + limit, totalRows);

  info.innerHTML = `
    Table <code>${esc(currentTable)}</code> —
    showing <b>${start}</b> to <b>${end}</b> of <b>${totalRows}</b> rows
  `;

  pageInfo.innerHTML = `Page ${currentPage + 1} / ${max + 1}`;
  pageInfoBottom.innerHTML = `Page ${currentPage + 1} / ${max + 1}`;

  if (data.rows.length === 0) {
    tableArea.innerHTML = "<div style='padding:12px;color:#aaa;'>No rows.</div>";
    return;
  }

  const columns = data.columns;

  let html = "<table><thead><tr>";

  for (const col of columns) {
    html += `<th>${esc(col)}</th>`;
  }

  html += "</tr></thead><tbody>";

  for (const row of data.rows) {
    html += "<tr>";

    for (const col of columns) {
      html += renderCell(col, row[col], row);
    }

    html += "</tr>";
  }

  html += "</tbody></table>";

  tableArea.innerHTML = html;
}

function nextPage() {
  if (currentPage < maxPage()) {
    currentPage++;
    loadTablePage();
  }
}

function prevPage() {
  if (currentPage > 0) {
    currentPage--;
    loadTablePage();
  }
}

function firstPage() {
  currentPage = 0;
  loadTablePage();
}

function lastPage() {
  currentPage = maxPage();
  loadTablePage();
}

tableSelect.addEventListener("change", () => {
  selectTable(tableSelect.value);
});

limitInput.addEventListener("change", () => {
  currentPage = 0;
  loadTablePage();
});

loadTables();
</script>
</body>
</html>
"""


def connect():
    return sqlite3.connect(DB_PATH)


def send_json(handler, data):
    raw = json.dumps(data).encode("utf-8")
    handler.send_response(200)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(raw)))
    handler.end_headers()
    handler.wfile.write(raw)


def send_text(handler, text, status=400):
    raw = text.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "text/plain")
    handler.send_header("Content-Length", str(len(raw)))
    handler.end_headers()
    handler.wfile.write(raw)


def table_names(conn):
    cur = conn.cursor()
    cur.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
          AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        """
    )

    return [row[0] for row in cur.fetchall()]


def table_columns(conn, table):
    cur = conn.cursor()
    cur.execute(f'PRAGMA table_info("{table}")')
    return [row[1] for row in cur.fetchall()]


def safe_table_name(conn, table):
    names = table_names(conn)

    if table not in names:
        raise ValueError(f"Unknown table: {table}")

    return table


def convert_value(value):
    if isinstance(value, bytes):
        return f"[BLOB {len(value)} bytes]"

    return value


def make_row(columns, db_row):
    row = {}

    for index, col in enumerate(columns):
        row[col] = convert_value(db_row[index])

    return row


def apply_homography(H, x, y):
    mapped_x = H[0][0] * x + H[0][1] * y + H[0][2]
    mapped_y = H[1][0] * x + H[1][1] * y + H[1][2]
    mapped_w = H[2][0] * x + H[2][1] * y + H[2][2]

    if abs(mapped_w) < 1e-9:
        return None

    return mapped_x / mapped_w, mapped_y / mapped_w


def transform_from_homography(homography_json, width_a, height_a, width_b, height_b):
    if not homography_json:
        return {
            "dx_px": None,
            "dy_px": None,
            "move_px": None,
            "scale_a_to_b": None,
            "rotation_deg": None,
        }

    try:
        H = json.loads(homography_json)

        center_a_x = width_a / 2.0
        center_a_y = height_a / 2.0
        center_b_x = width_b / 2.0
        center_b_y = height_b / 2.0

        mapped_center = apply_homography(H, center_a_x, center_a_y)

        if mapped_center is None:
            raise ValueError("bad center mapping")

        mapped_x, mapped_y = mapped_center

        dx_px = center_b_x - mapped_x
        dy_px = center_b_y - mapped_y
        move_px = math.hypot(dx_px, dy_px)

        step = 10.0

        p0 = apply_homography(H, center_a_x, center_a_y)
        px = apply_homography(H, center_a_x + step, center_a_y)
        py = apply_homography(H, center_a_x, center_a_y + step)

        if p0 is None or px is None or py is None:
            raise ValueError("bad local mapping")

        scale_x = math.dist(p0, px) / step
        scale_y = math.dist(p0, py) / step
        scale = math.sqrt(scale_x * scale_y)

        rotation_deg = math.degrees(math.atan2(px[1] - p0[1], px[0] - p0[0]))

        return {
            "dx_px": dx_px,
            "dy_px": dy_px,
            "move_px": move_px,
            "scale_a_to_b": scale,
            "rotation_deg": rotation_deg,
        }

    except Exception:
        return {
            "dx_px": None,
            "dy_px": None,
            "move_px": None,
            "scale_a_to_b": None,
            "rotation_deg": None,
        }


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        route = parsed.path
        q = urllib.parse.parse_qs(parsed.query)

        try:
            if route == "/":
                self.html()
            elif route == "/api/tables":
                self.api_tables()
            elif route == "/api/table":
                self.api_table(q)
            elif route == "/file":
                self.file(q)
            else:
                send_text(self, "not found", 404)

        except Exception as error:
            send_text(self, str(error), 500)

    def log_message(self, *args):
        return

    def html(self):
        raw = HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def api_tables(self):
        with connect() as conn:
            send_json(self, table_names(conn))

    def api_table(self, q):
        name = q.get("name", [None])[0]

        if not name:
            send_text(self, "missing table name", 400)
            return

        limit = int(q.get("limit", ["100"])[0])
        offset = int(q.get("offset", ["0"])[0])

        limit = max(1, min(limit, 1000))
        offset = max(0, offset)

        with connect() as conn:
            name = safe_table_name(conn, name)
            cur = conn.cursor()

            if name == "image_matches":
                match_cols = table_columns(conn, "image_matches")

                select_parts = [
                    'm."id" AS "id"',
                    'a."path" AS "image_a_path"',
                    'b."path" AS "image_b_path"',
                    'a."width" AS "_width_a"',
                    'a."height" AS "_height_a"',
                    'b."width" AS "_width_b"',
                    'b."height" AS "_height_b"',
                ]

                for col in match_cols:
                    if col != "id":
                        select_parts.append(f'm."{col}" AS "{col}"')

                raw_cols = [
                    "id",
                    "image_a_path",
                    "image_b_path",
                    "_width_a",
                    "_height_a",
                    "_width_b",
                    "_height_b",
                ] + [col for col in match_cols if col != "id"]

                cols = ["id", "image_a_path", "image_b_path", "transform"] + [
                    col for col in match_cols if col != "id"
                ]

                cur.execute('SELECT COUNT(*) FROM "image_matches"')
                total = cur.fetchone()[0]

                sql = f"""
                    SELECT {", ".join(select_parts)}
                    FROM image_matches m
                    JOIN images a ON a.id = m.image_a_id
                    JOIN images b ON b.id = m.image_b_id
                    ORDER BY m.id ASC
                    LIMIT ? OFFSET ?
                """

                cur.execute(sql, [limit, offset])

                rows = []

                for db_row in cur.fetchall():
                    row = make_row(raw_cols, db_row)

                    transform = transform_from_homography(
                        row.get("homography_json"),
                        row.get("_width_a"),
                        row.get("_height_a"),
                        row.get("_width_b"),
                        row.get("_height_b"),
                    )

                    row.update(transform)

                    for hidden_col in ("_width_a", "_height_a", "_width_b", "_height_b"):
                        row.pop(hidden_col, None)

                    rows.append(row)

            else:
                cols = table_columns(conn, name)

                cur.execute(f'SELECT COUNT(*) FROM "{name}"')
                total = cur.fetchone()[0]

                order_sql = ""

                if "id" in cols:
                    order_sql = 'ORDER BY "id" ASC'

                cur.execute(
                    f'SELECT * FROM "{name}" {order_sql} LIMIT ? OFFSET ?',
                    [limit, offset],
                )

                rows = [make_row(cols, db_row) for db_row in cur.fetchall()]

        send_json(
            self,
            {
                "table": name,
                "columns": cols,
                "total": total,
                "limit": limit,
                "offset": offset,
                "rows": rows,
            },
        )

    def file(self, q):
        if "path" not in q:
            send_text(self, "missing path", 400)
            return

        raw_path = q["path"][0].replace("\\", "/")
        requested_path = Path(raw_path)

        if requested_path.is_absolute():
            file_path = requested_path.resolve()
        else:
            file_path = (ROOT / requested_path).resolve()

        try:
            file_path.relative_to(ROOT)
        except ValueError:
            send_text(self, "blocked path outside root", 403)
            return

        if not file_path.exists():
            send_text(self, f"file not found: {file_path}", 404)
            return

        if not file_path.is_file():
            send_text(self, f"path is not a file: {file_path}", 404)
            return

        data = file_path.read_bytes()
        content_type = (
            mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        )

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)

print(f"DB:   {DB_PATH}")
print(f"Root: {ROOT}")
print(f"Open: http://127.0.0.1:{args.port}")
print("Press Ctrl+C to stop.")

server.serve_forever()
