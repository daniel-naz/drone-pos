import argparse
import json
import math
import os
import sqlite3
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".jpen", ".png"}
EARTH_RADIUS = 6378137.0

parser = argparse.ArgumentParser(
    description="Fast first-frame position estimate: cheap candidate search + SIFT verification."
)

parser.add_argument("-i", "--input", required=True, help="Image to estimate.")
parser.add_argument("-db", "--database", default="graph.db", help="Path to graph.db.")
parser.add_argument("-r", "--root", default=".", help="Project root used to resolve image paths stored in the DB.")

# Candidate search / cache
parser.add_argument(
    "--candidate-steps",
    default="100,300,700,0",
    help="Comma-separated candidate counts to try. 0 means all anchors. Example: 100,300,0",
)
parser.add_argument("--needed-matches", type=int, default=5, help="Stop expanding after this many usable SIFT matches.")
parser.add_argument("--rebuild-index", action="store_true", help="Rebuild cheap image-search index table.")
parser.add_argument("--index-size", type=int, default=32, help="Small image size used for cheap descriptors.")
parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1), help="Parallel SIFT verification workers.")

# Local graph/BFS mode for real-time tracking after a previous known image
parser.add_argument(
    "--last-known-id",
    type=int,
    default=None,
    help="images.id of the last known/locked anchor image. The script will search graph-nearby images first.",
)
parser.add_argument(
    "--last-known-path",
    default=None,
    help="Alternative to --last-known-id: path of the last known/locked anchor image.",
)
parser.add_argument(
    "--bfs-depth",
    type=int,
    default=2,
    help="How many graph hops to traverse from the last known image before falling back to global search.",
)
parser.add_argument(
    "--bfs-neighbor-limit",
    type=int,
    default=40,
    help="Maximum neighbors to expand per graph node, ordered by match quality.",
)
parser.add_argument(
    "--bfs-max-candidates",
    type=int,
    default=250,
    help="Maximum local BFS anchor candidates to SIFT-check before global fallback.",
)
parser.add_argument(
    "--local-only",
    action="store_true",
    help="Only use BFS/local graph candidates. Do not fall back to cheap global candidate search.",
)

# SIFT settings
parser.add_argument("--feature-max-size", type=int, default=1000, help="Must match the --max-size used when building graph.db features.")
parser.add_argument("--max-features", type=int, default=600)
parser.add_argument("--ratio", type=float, default=0.75)
parser.add_argument("--ransac", type=float, default=5.0)
parser.add_argument("--min-good", type=int, default=20)
parser.add_argument("--min-inliers", type=int, default=12)
parser.add_argument("--min-inlier-ratio", type=float, default=0.20)
parser.add_argument("--min-coverage", type=float, default=0.02)
parser.add_argument("--min-confidence", type=float, default=0.15)
parser.add_argument("--max-reprojection-error", type=float, default=10.0)

# Position settings
parser.add_argument("--query-fov", type=float, default=60.0)
parser.add_argument("--anchor-fov", type=float, default=60.0)
parser.add_argument("--min-calibration-confidence", type=float, default=0.35)

parser.add_argument("--save", action="store_true")
parser.add_argument("-o", "--output", default=None)

args = parser.parse_args()

DB_PATH = Path(args.database).resolve()
ROOT = Path(args.root).resolve()
QUERY_PATH = Path(args.input)

if not QUERY_PATH.is_absolute():
    QUERY_PATH = (Path.cwd() / QUERY_PATH).resolve()

if not DB_PATH.exists():
    raise FileNotFoundError(f"Database does not exist: {DB_PATH}")

if not QUERY_PATH.exists():
    raise FileNotFoundError(f"Input image does not exist: {QUERY_PATH}")

if QUERY_PATH.suffix.lower() not in IMAGE_EXTENSIONS:
    raise ValueError(f"Input image must be one of: {sorted(IMAGE_EXTENSIONS)}")


def parse_candidate_steps(text):
    steps = []
    for part in text.split(","):
        part = part.strip()
        if part:
            steps.append(int(part))
    if not steps:
        raise ValueError("--candidate-steps cannot be empty")
    return steps


def connect():
    return sqlite3.connect(DB_PATH)


def table_exists(conn, table_name):
    cur = conn.cursor()
    cur.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
          AND name = ?
        """,
        [table_name],
    )
    return cur.fetchone() is not None


def table_columns(conn, table_name):
    cur = conn.cursor()
    cur.execute(f'PRAGMA table_info("{table_name}")')
    return [row[1] for row in cur.fetchall()]


def column_exists(conn, table_name, column_name):
    return column_name in table_columns(conn, table_name)


def latlon_to_local_meters(lat, lon, origin_lat, origin_lon):
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    origin_lat_rad = math.radians(origin_lat)
    origin_lon_rad = math.radians(origin_lon)

    east = (lon_rad - origin_lon_rad) * math.cos(origin_lat_rad) * EARTH_RADIUS
    north = (lat_rad - origin_lat_rad) * EARTH_RADIUS
    return east, north


def local_meters_to_latlon(east, north, origin_lat, origin_lon):
    origin_lat_rad = math.radians(origin_lat)
    lat = origin_lat + math.degrees(north / EARTH_RADIUS)
    lon = origin_lon + math.degrees(east / (EARTH_RADIUS * math.cos(origin_lat_rad)))
    return lat, lon


def resolve_db_image_path(path_text):
    path = Path(path_text.replace("\\", "/"))
    if path.is_absolute():
        return path.resolve()
    return (ROOT / path).resolve()


def image_scale(width, height, max_size):
    if max_size <= 0:
        return 1.0
    largest = max(width, height)
    if largest <= max_size:
        return 1.0
    return max_size / largest


def load_gray(path, max_size):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")

    height, width = image.shape[:2]
    scale = image_scale(width, height, max_size)
    if scale != 1.0:
        image = cv2.resize(
            image,
            (round(width * scale), round(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    return image, width, height, scale


def extract_features(path, sift, max_size):
    gray, width, height, scale = load_gray(path, max_size)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    if descriptors is not None:
        descriptors = descriptors.astype(np.float32)
    return {
        "path": str(path),
        "width": width,
        "height": height,
        "scale": scale,
        "keypoints": keypoints,
        "descriptors": descriptors,
    }


def deserialize_keypoints(keypoints_json):
    """
    Supports both keypoint formats used by your scripts:

    Old/verbose:
        {"x": 12.3, "y": 45.6, "size": ..., ...}

    Compact/current graph builder:
        [x, y, size, angle, response, octave, class_id]
    """
    items = json.loads(keypoints_json)
    keypoints = []

    for item in items:
        if isinstance(item, dict):
            x = float(item["x"])
            y = float(item["y"])
            size = float(item.get("size", 1.0))
            angle = float(item.get("angle", -1.0))
            response = float(item.get("response", 0.0))
            octave = int(item.get("octave", 0))
            class_id = int(item.get("class_id", -1))
        else:
            x = float(item[0])
            y = float(item[1])
            size = float(item[2])
            angle = float(item[3])
            response = float(item[4])
            octave = int(item[5])
            class_id = int(item[6])

        keypoints.append(
            cv2.KeyPoint(
                x,
                y,
                size,
                angle,
                response,
                octave,
                class_id,
            )
        )

    return keypoints

def descriptors_from_blob(blob, rows, cols, dtype):
    if blob is None or rows is None or cols is None or rows <= 0 or cols <= 0:
        return None
    array = np.frombuffer(blob, dtype=np.dtype(dtype)).reshape((rows, cols))
    return array.astype(np.float32)


def normalize_l2(vector):
    vector = vector.astype(np.float32).reshape(-1)
    vector -= vector.mean()
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector /= norm
    return vector.astype(np.float32)


def normalize_l1(vector):
    vector = vector.astype(np.float32).reshape(-1)
    total = float(vector.sum())
    if total > 0:
        vector /= total
    return vector.astype(np.float32)


def cheap_descriptor_from_image(path, size):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None

    small = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    gray_vector = normalize_l2(gray)
    edges = cv2.Canny(gray, 60, 160)
    edge_vector = normalize_l2(edges)

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 4], [0, 180, 0, 256, 0, 256])
    hist_vector = normalize_l1(hist)

    height, width = image.shape[:2]
    return {"width": width, "height": height, "gray": gray_vector, "edge": edge_vector, "hist": hist_vector}


def create_search_index_table(conn):
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS image_search_index (
            path TEXT PRIMARY KEY,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            index_size INTEGER NOT NULL,
            gray_blob BLOB NOT NULL,
            edge_blob BLOB NOT NULL,
            hist_blob BLOB NOT NULL
        )
        """
    )
    conn.commit()


def array_to_blob(array):
    return sqlite3.Binary(array.astype(np.float32).tobytes())


def blob_to_array(blob):
    return np.frombuffer(blob, dtype=np.float32)


def load_anchor_positions(conn):
    if not table_exists(conn, "edge_transforms"):
        raise RuntimeError("edge_transforms table does not exist. Run analyze_transforms.py first.")

    cur = conn.cursor()
    cur.execute(
        """
        SELECT image_a_path, lat_a, lon_a, alt_a
        FROM edge_transforms
        UNION
        SELECT image_b_path, lat_b, lon_b, alt_b
        FROM edge_transforms
        """
    )

    anchors = {}
    for path, lat, lon, alt in cur.fetchall():
        if path not in anchors:
            anchors[path] = {"path": path, "lat": float(lat), "lon": float(lon), "alt": float(alt)}
    return anchors


def build_or_load_search_index(conn, anchors):
    create_search_index_table(conn)
    cur = conn.cursor()

    if args.rebuild_index:
        print("Rebuilding cheap image search index...")
        cur.execute("DELETE FROM image_search_index")
        conn.commit()

    wanted_paths = list(anchors.keys())
    cur.execute(
        """
        SELECT path, width, height, gray_blob, edge_blob, hist_blob
        FROM image_search_index
        WHERE index_size = ?
        """,
        [args.index_size],
    )

    indexed = {}
    for path, width, height, gray_blob, edge_blob, hist_blob in cur.fetchall():
        indexed[path] = {
            "path": path,
            "width": int(width),
            "height": int(height),
            "gray": blob_to_array(gray_blob),
            "edge": blob_to_array(edge_blob),
            "hist": blob_to_array(hist_blob),
        }

    missing = [path for path in wanted_paths if path not in indexed]
    if missing:
        print(f"Building cheap descriptors for missing anchors: {len(missing)}")

    inserted = 0
    for db_path in missing:
        image_path = resolve_db_image_path(db_path)
        if not image_path.exists():
            continue

        desc = cheap_descriptor_from_image(image_path, args.index_size)
        if desc is None:
            continue

        cur.execute(
            """
            INSERT OR REPLACE INTO image_search_index (
                path, width, height, index_size, gray_blob, edge_blob, hist_blob
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                db_path,
                desc["width"],
                desc["height"],
                args.index_size,
                array_to_blob(desc["gray"]),
                array_to_blob(desc["edge"]),
                array_to_blob(desc["hist"]),
            ),
        )
        indexed[db_path] = {"path": db_path, **desc}
        inserted += 1

        if inserted % 100 == 0:
            conn.commit()
            print(f"  indexed {inserted}/{len(missing)}")

    conn.commit()
    return indexed


def cheap_score(query_desc, anchor_desc):
    gray_score = (float(np.dot(query_desc["gray"], anchor_desc["gray"])) + 1.0) / 2.0
    edge_score = (float(np.dot(query_desc["edge"], anchor_desc["edge"])) + 1.0) / 2.0
    hist_score = float(np.minimum(query_desc["hist"], anchor_desc["hist"]).sum())

    gray_score = max(0.0, min(gray_score, 1.0))
    edge_score = max(0.0, min(edge_score, 1.0))
    hist_score = max(0.0, min(hist_score, 1.0))
    return 0.40 * gray_score + 0.35 * edge_score + 0.25 * hist_score


def rank_candidates(query_path, anchors, search_index):
    query_desc = cheap_descriptor_from_image(query_path, args.index_size)
    if query_desc is None:
        raise RuntimeError(f"Could not make cheap descriptor for {query_path}")

    scored = []
    for anchor in anchors.values():
        desc = search_index.get(anchor["path"])
        if desc is None:
            continue
        score = cheap_score(query_desc, desc)
        scored.append((score, anchor))

    scored.sort(key=lambda item: item[0], reverse=True)
    return scored


def normalize_db_path(path_text):
    return str(path_text).replace("\\", "/")


def get_image_id_path_maps(conn):
    cur = conn.cursor()
    cur.execute('SELECT id, path FROM images')

    id_to_path = {}
    path_to_id = {}
    normalized_to_id = {}

    for image_id, path in cur.fetchall():
        id_to_path[int(image_id)] = path
        path_to_id[path] = int(image_id)
        normalized_to_id[normalize_db_path(path)] = int(image_id)

    return id_to_path, path_to_id, normalized_to_id


def resolve_last_known_image(conn):
    id_to_path, path_to_id, normalized_to_id = get_image_id_path_maps(conn)

    if args.last_known_id is not None:
        if args.last_known_id not in id_to_path:
            raise ValueError(f"No image found with id: {args.last_known_id}")

        return args.last_known_id, id_to_path[args.last_known_id]

    if args.last_known_path is None:
        return None, None

    raw_path = str(args.last_known_path)
    normalized = normalize_db_path(raw_path)

    if raw_path in path_to_id:
        image_id = path_to_id[raw_path]
        return image_id, id_to_path[image_id]

    if normalized in normalized_to_id:
        image_id = normalized_to_id[normalized]
        return image_id, id_to_path[image_id]

    # Helpful fallback: allow passing only the tail of the path.
    matches = [
        (image_id, path)
        for image_id, path in id_to_path.items()
        if normalize_db_path(path).endswith(normalized)
    ]

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        raise ValueError(
            f"last-known-path is ambiguous. {len(matches)} DB images end with: {raw_path}"
        )

    raise ValueError(f"No image found with path: {raw_path}")


def graph_neighbors(conn, image_id):
    has_confidence = column_exists(conn, "image_matches", "confidence")
    has_inliers = column_exists(conn, "image_matches", "inliers")

    confidence_expr = "COALESCE(confidence, 0)" if has_confidence else "0"
    inliers_expr = "COALESCE(inliers, 0)" if has_inliers else "0"

    cur = conn.cursor()
    cur.execute(
        f"""
        SELECT
            CASE
                WHEN image_a_id = ? THEN image_b_id
                ELSE image_a_id
            END AS neighbor_id,
            {confidence_expr} AS edge_confidence,
            {inliers_expr} AS edge_inliers
        FROM image_matches
        WHERE image_a_id = ? OR image_b_id = ?
        ORDER BY edge_confidence DESC, edge_inliers DESC
        LIMIT ?
        """,
        [image_id, image_id, image_id, args.bfs_neighbor_limit],
    )

    return [
        {
            "image_id": int(row[0]),
            "edge_confidence": float(row[1] or 0),
            "edge_inliers": int(row[2] or 0),
        }
        for row in cur.fetchall()
    ]


def bfs_local_candidates(conn, anchors):
    start_id, start_path = resolve_last_known_image(conn)

    if start_id is None:
        return [], None

    id_to_path, _, _ = get_image_id_path_maps(conn)

    visited = {start_id}
    queue = deque([(start_id, 0)])
    candidates = []
    candidate_paths = set()

    while queue and len(candidates) < args.bfs_max_candidates:
        current_id, depth = queue.popleft()
        current_path = id_to_path.get(current_id)

        if current_path in anchors and current_path not in candidate_paths:
            candidates.append(anchors[current_path])
            candidate_paths.add(current_path)

            if len(candidates) >= args.bfs_max_candidates:
                break

        if depth >= args.bfs_depth:
            continue

        for neighbor in graph_neighbors(conn, current_id):
            neighbor_id = neighbor["image_id"]

            if neighbor_id in visited:
                continue

            visited.add(neighbor_id)
            queue.append((neighbor_id, depth + 1))

    return candidates, {
        "start_id": start_id,
        "start_path": start_path,
        "visited_nodes": len(visited),
        "candidates": len(candidates),
    }


def fit_pixel_to_meter_model(conn):
    cur = conn.cursor()
    cur.execute(
        """
        SELECT dx_px, dy_px, gps_east_delta, gps_north_delta, confidence
        FROM edge_transforms
        WHERE dx_px IS NOT NULL
          AND dy_px IS NOT NULL
          AND gps_east_delta IS NOT NULL
          AND gps_north_delta IS NOT NULL
          AND confidence >= ?
        """,
        [args.min_calibration_confidence],
    )

    rows = cur.fetchall()
    if len(rows) < 4:
        raise RuntimeError("Not enough calibration rows in edge_transforms.")

    X = []
    Y = []
    W = []
    for dx, dy, east, north, confidence in rows:
        X.append([float(dx), float(dy)])
        Y.append([float(east), float(north)])
        W.append(max(float(confidence or 0.1), 0.1))

    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    W = np.sqrt(np.asarray(W, dtype=np.float64)).reshape(-1, 1)

    matrix, _, _, _ = np.linalg.lstsq(X * W, Y * W, rcond=None)
    predictions = X @ matrix
    errors = np.linalg.norm(predictions - Y, axis=1)

    return {
        "matrix": matrix,
        "rows": len(rows),
        "mean_error_m": float(errors.mean()),
        "median_error_m": float(np.median(errors)),
    }


def load_cached_anchor_features(conn, candidate_paths):
    features_by_path = {}
    if not table_exists(conn, "image_features") or not candidate_paths:
        return features_by_path

    placeholders = ",".join("?" for _ in candidate_paths)
    cur = conn.cursor()
    cur.execute(
        f"""
        SELECT
            i.path,
            i.width,
            i.height,
            f.keypoints_json,
            f.descriptors_blob,
            f.descriptor_rows,
            f.descriptor_cols,
            f.descriptor_dtype
        FROM images i
        JOIN image_features f ON f.image_id = i.id
        WHERE i.path IN ({placeholders})
        """,
        candidate_paths,
    )

    for row in cur.fetchall():
        path, width, height, keypoints_json, descriptors_blob, rows, cols, dtype = row
        try:
            scale = image_scale(width, height, args.feature_max_size)
            features_by_path[path] = {
                "path": path,
                "width": int(width),
                "height": int(height),
                "scale": scale,
                "keypoints": deserialize_keypoints(keypoints_json),
                "descriptors": descriptors_from_blob(descriptors_blob, rows, cols, dtype),
            }
        except Exception as error:
            print(f"Could not load cached features for {path}: {error}")
    return features_by_path


def ratio_matches(desc_a, desc_b, matcher, ratio):
    raw_matches = matcher.knnMatch(desc_a, desc_b, k=2)
    good = []
    for pair in raw_matches:
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance < ratio * second.distance:
            good.append(first)
    return good


def mutual_ratio_matches(desc_a, desc_b, matcher, ratio):
    forward = ratio_matches(desc_a, desc_b, matcher, ratio)
    backward = ratio_matches(desc_b, desc_a, matcher, ratio)
    backward_pairs = {(m.trainIdx, m.queryIdx) for m in backward}
    return [m for m in forward if (m.queryIdx, m.trainIdx) in backward_pairs]


def points_coverage(points, width, height):
    if len(points) < 4:
        return 0.0
    points = np.asarray(points, dtype=np.float32)
    xs = points[:, 0]
    ys = points[:, 1]
    box_width = float(xs.max() - xs.min())
    box_height = float(ys.max() - ys.min())
    if box_width <= 0 or box_height <= 0:
        return 0.0
    return (box_width * box_height) / float(width * height)


def apply_homography(H, x, y):
    point = np.array([x, y, 1.0], dtype=np.float64)
    mapped = H @ point
    if abs(mapped[2]) < 1e-9:
        return None
    return mapped[0] / mapped[2], mapped[1] / mapped[2]


def transform_from_homography(H, width_a, height_a, width_b, height_b):
    center_a_x = width_a / 2.0
    center_a_y = height_a / 2.0
    center_b_x = width_b / 2.0
    center_b_y = height_b / 2.0

    mapped_center = apply_homography(H, center_a_x, center_a_y)
    if mapped_center is None:
        return None

    mapped_x, mapped_y = mapped_center
    dx_px = center_b_x - mapped_x
    dy_px = center_b_y - mapped_y
    move_px = math.hypot(dx_px, dy_px)

    step = 10.0
    p0 = apply_homography(H, center_a_x, center_a_y)
    px = apply_homography(H, center_a_x + step, center_a_y)
    py = apply_homography(H, center_a_x, center_a_y + step)
    if p0 is None or px is None or py is None:
        return None

    scale_x = math.dist(p0, px) / step
    scale_y = math.dist(p0, py) / step
    if scale_x <= 0 or scale_y <= 0:
        return None

    scale = math.sqrt(scale_x * scale_y)
    rotation_deg = math.degrees(math.atan2(px[1] - p0[1], px[0] - p0[0]))
    return dx_px, dy_px, move_px, scale, rotation_deg


def mean_reprojection_error(H, src_points, dst_points, mask):
    mask = mask.ravel().astype(bool)
    if mask.sum() == 0:
        return None

    src = src_points.reshape(-1, 2)[mask]
    dst = dst_points.reshape(-1, 2)[mask]
    errors = []
    for point_a, point_b in zip(src, dst):
        mapped = apply_homography(H, point_a[0], point_a[1])
        if mapped is not None:
            errors.append(math.dist(mapped, point_b))
    if not errors:
        return None
    return float(sum(errors) / len(errors))


def calculate_match_confidence(inliers, inlier_ratio, coverage_a, coverage_b, reprojection_error):
    ratio_score = max(0.0, min((inlier_ratio - args.min_inlier_ratio) / (1.0 - args.min_inlier_ratio), 1.0))
    inlier_score = min(inliers / 80.0, 1.0)
    coverage_score = min(min(coverage_a, coverage_b) / 0.20, 1.0)
    if reprojection_error is None:
        reprojection_score = 0.0
    else:
        reprojection_score = max(0.0, min(1.0 - (reprojection_error / args.max_reprojection_error), 1.0))
    confidence = 0.35 * ratio_score + 0.25 * inlier_score + 0.20 * coverage_score + 0.20 * reprojection_score
    return confidence, ratio_score


def match_query_to_anchor(query, anchor):
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    kp_a = query["keypoints"]
    desc_a = query["descriptors"]
    scale_a = query["scale"]
    kp_b = anchor["keypoints"]
    desc_b = anchor["descriptors"]
    scale_b = anchor["scale"]

    if desc_a is None or desc_b is None:
        return None
    if len(kp_a) < 8 or len(kp_b) < 8:
        return None

    good_matches = mutual_ratio_matches(desc_a, desc_b, matcher, args.ratio)
    if len(good_matches) < args.min_good:
        return None

    src_points = []
    dst_points = []
    for match in good_matches:
        point_a = kp_a[match.queryIdx].pt
        point_b = kp_b[match.trainIdx].pt
        src_points.append([point_a[0] / scale_a, point_a[1] / scale_a])
        dst_points.append([point_b[0] / scale_b, point_b[1] / scale_b])

    src_np = np.float32(src_points).reshape(-1, 1, 2)
    dst_np = np.float32(dst_points).reshape(-1, 1, 2)

    method = cv2.USAC_MAGSAC if hasattr(cv2, "USAC_MAGSAC") else cv2.RANSAC
    H, mask = cv2.findHomography(src_np, dst_np, method, args.ransac)
    if H is None or mask is None:
        return None

    mask_bool = mask.ravel().astype(bool)
    inliers = int(mask_bool.sum())
    if inliers < args.min_inliers:
        return None

    inlier_ratio = inliers / len(good_matches)
    if inlier_ratio < args.min_inlier_ratio:
        return None

    inlier_points_a = [src_points[i] for i, keep in enumerate(mask_bool) if keep]
    inlier_points_b = [dst_points[i] for i, keep in enumerate(mask_bool) if keep]
    coverage_a = points_coverage(inlier_points_a, query["width"], query["height"])
    coverage_b = points_coverage(inlier_points_b, anchor["width"], anchor["height"])
    if coverage_a < args.min_coverage or coverage_b < args.min_coverage:
        return None

    reprojection_error = mean_reprojection_error(H, src_np, dst_np, mask)
    if reprojection_error is None or reprojection_error > args.max_reprojection_error:
        return None

    transform = transform_from_homography(H, query["width"], query["height"], anchor["width"], anchor["height"])
    if transform is None:
        return None

    dx_px, dy_px, move_px, scale_a_to_b, rotation_deg = transform
    confidence, ratio_score = calculate_match_confidence(inliers, inlier_ratio, coverage_a, coverage_b, reprojection_error)
    if confidence < args.min_confidence:
        return None

    average_distance = float(sum(match.distance for i, match in enumerate(good_matches) if mask_bool[i]) / inliers)

    return {
        "good_matches": len(good_matches),
        "inliers": inliers,
        "inlier_ratio": inlier_ratio,
        "average_distance": average_distance,
        "mean_reprojection_error": reprojection_error,
        "coverage_a": coverage_a,
        "coverage_b": coverage_b,
        "confidence": confidence,
        "ratio_score": ratio_score,
        "dx_px": dx_px,
        "dy_px": dy_px,
        "move_px": move_px,
        "scale_a_to_b": scale_a_to_b,
        "rotation_deg": rotation_deg,
        "homography": H.tolist(),
    }


def estimate_altitude_from_scale(query_width, anchor_width, anchor_alt, scale_a_to_b):
    if scale_a_to_b <= 0:
        return None
    query_fov = math.radians(args.query_fov)
    anchor_fov = math.radians(args.anchor_fov)
    return scale_a_to_b * anchor_alt * math.tan(anchor_fov / 2.0) * query_width / (math.tan(query_fov / 2.0) * anchor_width)


def weighted_average_estimates(estimates):
    weights = np.asarray([max(e["confidence"], 1e-6) for e in estimates], dtype=np.float64)
    weights = weights / weights.sum()

    east = float(sum(w * e["estimated_east"] for w, e in zip(weights, estimates)))
    north = float(sum(w * e["estimated_north"] for w, e in zip(weights, estimates)))

    alt_values = [(w, e["estimated_alt"]) for w, e in zip(weights, estimates) if e["estimated_alt"] is not None]
    if alt_values:
        alt = float(sum(w * value for w, value in alt_values) / sum(w for w, _ in alt_values))
    else:
        alt = None

    confidence = float(sum(w * e["confidence"] for w, e in zip(weights, estimates)))
    return east, north, alt, confidence


def make_estimate(anchor, anchor_features, match_result, pixel_to_meter_matrix, origin_lat, origin_lon):
    dxdy = np.asarray([match_result["dx_px"], match_result["dy_px"]], dtype=np.float64)
    delta_east, delta_north = dxdy @ pixel_to_meter_matrix

    estimated_east = anchor["east"] - float(delta_east)
    estimated_north = anchor["north"] - float(delta_north)

    estimated_alt = estimate_altitude_from_scale(
        query_features["width"],
        anchor_features["width"],
        anchor["alt"],
        match_result["scale_a_to_b"],
    )

    estimated_lat, estimated_lon = local_meters_to_latlon(estimated_east, estimated_north, origin_lat, origin_lon)

    return {
        "anchor_path": anchor["path"],
        "anchor_lat": anchor["lat"],
        "anchor_lon": anchor["lon"],
        "anchor_alt": anchor["alt"],
        "estimated_lat": estimated_lat,
        "estimated_lon": estimated_lon,
        "estimated_alt": estimated_alt,
        "estimated_east": estimated_east,
        "estimated_north": estimated_north,
        **match_result,
    }


def verify_candidate(anchor, anchor_features):
    if anchor_features is None:
        return None
    result = match_query_to_anchor(query_features, anchor_features)
    if result is None:
        return None
    return anchor["path"], result


# BFS-only runtime
# This script intentionally DOES NOT do global search.
# Use estimate_image_position.py for the first/global lock.
# Then pass the best anchor path/id here for local tracking.

if args.last_known_id is None and args.last_known_path is None:
    raise ValueError(
        "BFS-only mode requires --last-known-id or --last-known-path. "
        "Run estimate_image_position.py first to get the initial/global anchor."
    )

print("BFS-only estimator settings:")
print(f"  DB: {DB_PATH}")
print(f"  Query image: {QUERY_PATH}")
print(f"  Last known id: {args.last_known_id}")
print(f"  Last known path: {args.last_known_path}")
print(f"  BFS depth: {args.bfs_depth}")
print(f"  Neighbor limit: {args.bfs_neighbor_limit}")
print(f"  Max candidates: {args.bfs_max_candidates}")
print(f"  Workers: {args.workers}")
print(f"  Max SIFT features: {args.max_features}")
print()

with connect() as conn:
    anchors = load_anchor_positions(conn)

    if len(anchors) == 0:
        raise RuntimeError("No anchor GPS positions found in edge_transforms.")

    origin = next(iter(anchors.values()))
    origin_lat = origin["lat"]
    origin_lon = origin["lon"]

    for anchor in anchors.values():
        east, north = latlon_to_local_meters(
            anchor["lat"],
            anchor["lon"],
            origin_lat,
            origin_lon,
        )
        anchor["east"] = east
        anchor["north"] = north

    calibration = fit_pixel_to_meter_model(conn)
    pixel_to_meter_matrix = calibration["matrix"]

    local_candidates, local_info = bfs_local_candidates(conn, anchors)

print("Calibration:")
print(f"  rows: {calibration['rows']}")
print(f"  mean error: {calibration['mean_error_m']:.2f} m")
print(f"  median error: {calibration['median_error_m']:.2f} m")
print()

print(f"Anchors in DB: {len(anchors)}")

if local_info is None:
    raise RuntimeError(
        "Could not resolve last-known image. Provide an images.id with --last-known-id "
        "or an exact DB path / suffix with --last-known-path."
    )

print(
    f"Local BFS from image id {local_info['start_id']} "
    f"({local_info['start_path']})"
)
print(
    f"  visited graph nodes: {local_info['visited_nodes']} | "
    f"anchor candidates: {local_info['candidates']}"
)
print()

if not local_candidates:
    raise RuntimeError(
        "BFS found no local anchor candidates. Increase --bfs-depth / --bfs-neighbor-limit "
        "or check that the last-known image is present in images/image_matches."
    )

sift = (
    cv2.SIFT_create(nfeatures=args.max_features)
    if args.max_features > 0
    else cv2.SIFT_create()
)

query_features = extract_features(QUERY_PATH, sift, args.feature_max_size)

all_matches = []
checked_paths = set()


def run_sift_stage(stage_name, candidates):
    new_candidates = [
        anchor for anchor in candidates
        if anchor["path"] not in checked_paths
    ]

    if not new_candidates:
        print(f"SIFT verifying {stage_name}: 0 new candidates")
        return []

    print(f"SIFT verifying {stage_name}: {len(new_candidates)} new candidates")

    with connect() as conn:
        cached_features = load_cached_anchor_features(
            conn,
            [anchor["path"] for anchor in new_candidates],
        )

    estimated_items = []

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {}

        for anchor in new_candidates:
            checked_paths.add(anchor["path"])
            features = cached_features.get(anchor["path"])

            if features is None:
                image_path = resolve_db_image_path(anchor["path"])

                if not image_path.exists():
                    continue

                try:
                    features = extract_features(image_path, sift, args.feature_max_size)
                except Exception as error:
                    print(f"Skipping candidate {anchor['path']}: {error}")
                    continue

            future = executor.submit(verify_candidate, anchor, features)
            futures[future] = (anchor, features)

        for future in as_completed(futures):
            anchor, features = futures[future]
            result = future.result()

            if result is None:
                continue

            _, match_result = result
            estimate = make_estimate(
                anchor,
                features,
                match_result,
                pixel_to_meter_matrix,
                origin_lat,
                origin_lon,
            )
            estimated_items.append(estimate)

    estimated_items.sort(key=lambda item: item["confidence"], reverse=True)
    return estimated_items


estimated_items = run_sift_stage("local BFS candidates", local_candidates)
all_matches.extend(estimated_items)
all_matches.sort(key=lambda item: item["confidence"], reverse=True)

print(f"  usable local matches: {len(estimated_items)}")
print(f"  checked anchors: {len(checked_paths)} / {len(anchors)}")

for item in all_matches[:8]:
    print(
        f"    {item['anchor_path']} | "
        f"conf={item['confidence']:.2f} | "
        f"inliers={item['inliers']} | "
        f"lat={item['estimated_lat']:.7f} | "
        f"lon={item['estimated_lon']:.7f}"
    )

print()

if len(all_matches) == 0:
    raise RuntimeError(
        "No usable LOCAL/BFS matches found. This script does not fall back to global. "
        "Either the previous anchor is wrong/stale, the graph neighborhood is too small, "
        "or thresholds are too strict. Try increasing --bfs-depth / --bfs-max-candidates "
        "or lowering --min-good / --min-inliers."
    )

best_matches = all_matches[: min(15, len(all_matches))]
east, north, alt, final_confidence = weighted_average_estimates(best_matches)
lat, lon = local_meters_to_latlon(east, north, origin_lat, origin_lon)

result_payload = {
    "mode": "bfs_only",
    "query_image": str(QUERY_PATH),
    "estimated_lat": lat,
    "estimated_lon": lon,
    "estimated_alt": alt,
    "confidence": final_confidence,
    "used_matches": len(best_matches),
    "total_matches": len(all_matches),
    "checked_anchors": len(checked_paths),
    "anchors_in_db": len(anchors),
    "last_known_id": args.last_known_id,
    "last_known_path": args.last_known_path,
    "resolved_last_known_id": local_info["start_id"],
    "resolved_last_known_path": local_info["start_path"],
    "bfs_depth": args.bfs_depth,
    "bfs_neighbor_limit": args.bfs_neighbor_limit,
    "bfs_max_candidates": args.bfs_max_candidates,
    "bfs_visited_nodes": local_info["visited_nodes"],
    "bfs_candidates": local_info["candidates"],
    "calibration": {
        "rows": calibration["rows"],
        "mean_error_m": calibration["mean_error_m"],
        "median_error_m": calibration["median_error_m"],
        "matrix": calibration["matrix"].tolist(),
    },
    "top_matches": best_matches,
}

print("Final estimate:")
print(f"  lat: {lat:.8f}")
print(f"  lon: {lon:.8f}")
print(f"  alt: {alt:.2f} m" if alt is not None else "  alt: unknown")
print(f"  confidence: {final_confidence:.3f}")
print(f"  used matches: {len(best_matches)} / {len(all_matches)}")
print(f"  checked anchors: {len(checked_paths)} / {len(anchors)}")

if args.output:
    output_path = Path(args.output)
    output_path.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")
    print(f"Saved JSON: {output_path}")

if args.save:
    with connect() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS estimated_positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_image TEXT NOT NULL,
                estimated_lat REAL NOT NULL,
                estimated_lon REAL NOT NULL,
                estimated_alt REAL,
                confidence REAL NOT NULL,
                used_matches INTEGER NOT NULL,
                total_matches INTEGER NOT NULL,
                checked_anchors INTEGER NOT NULL,
                details_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        cur.execute(
            """
            INSERT INTO estimated_positions (
                query_image,
                estimated_lat,
                estimated_lon,
                estimated_alt,
                confidence,
                used_matches,
                total_matches,
                checked_anchors,
                details_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(QUERY_PATH),
                lat,
                lon,
                alt,
                final_confidence,
                len(best_matches),
                len(all_matches),
                len(checked_paths),
                json.dumps(result_payload),
            ),
        )

        conn.commit()

    print("Saved estimate to table: estimated_positions")
