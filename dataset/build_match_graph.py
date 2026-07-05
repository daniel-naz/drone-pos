import argparse
import json
import math
import sqlite3
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".jpen", ".png"}

parser = argparse.ArgumentParser(
    description="Build a fast image match graph using cached OpenCV SIFT features."
)

parser.add_argument("-i", "--input", required=True, help="Input image folder. Can contain nested folders.")
parser.add_argument("-db", "--database", default="graph.db", help="Output SQLite database path.")

# Match quality thresholds.
parser.add_argument("--min-good", type=int, default=40)
parser.add_argument("--min-inliers", type=int, default=25)
parser.add_argument("--min-inlier-ratio", type=float, default=0.40)
parser.add_argument("--min-coverage", type=float, default=0.08)
parser.add_argument("--min-confidence", type=float, default=0.60)
parser.add_argument("--max-reprojection-error", type=float, default=6.0)
parser.add_argument("--ratio", type=float, default=0.72, help="Lowe ratio test. Lower is stricter.")
parser.add_argument("--ransac", type=float, default=3.0)

# Feature/cache settings.
parser.add_argument("--max-size", type=int, default=1200, help="Resize largest image side before SIFT. Lower is faster.")
parser.add_argument("--max-features", type=int, default=2500, help="Maximum SIFT features per image. 0 means unlimited.")
parser.add_argument("--overwrite", action="store_true", help="Delete existing database before creating a new one.")
parser.add_argument("--clear-matches", action="store_true", help="Keep cached features, delete old matches.")
parser.add_argument("--no-store-match-points", action="store_true", help="Do not store inlier point pairs JSON.")

# Candidate generation. This is the main speedup.
parser.add_argument(
    "--same-folder-window",
    type=int,
    default=30,
    help="Compare each frame only with the next N frames in the same folder. 0 disables same-folder candidates.",
)
parser.add_argument(
    "--cross-folder-top-k",
    type=int,
    default=10,
    help="For each image, compare only the K most similar images from other folders using a cheap thumbnail prefilter. 0 disables cross-folder candidates.",
)
parser.add_argument("--thumb-size", type=int, default=32, help="Thumbnail size used for cross-folder candidate search.")
parser.add_argument(
    "--min-thumb-similarity",
    type=float,
    default=0.0,
    help="Minimum thumbnail cosine similarity for cross-folder candidates. 0 keeps top-k regardless.",
)
parser.add_argument(
    "--candidate-limit",
    type=int,
    default=0,
    help="Debug: only process the first N candidate pairs. 0 means no limit.",
)
parser.add_argument(
    "--workers",
    type=int,
    default=1,
    help="Thread count for pair matching. Try 4-8. SQLite writing stays on the main thread.",
)
parser.add_argument(
    "--matcher",
    choices=["flann", "bf"],
    default="flann",
    help="FLANN is faster for SIFT. BF is exact but slower.",
)

args = parser.parse_args()

INPUT = Path(args.input)
DB_PATH = Path(args.database)

if not INPUT.exists():
    raise FileNotFoundError(f"Input folder does not exist: {INPUT}")
if not INPUT.is_dir():
    raise ValueError(f"Input must be a folder: {INPUT}")
if DB_PATH.exists() and args.overwrite:
    DB_PATH.unlink()

thread_data = threading.local()


def parse_frame_id(path: Path):
    try:
        return int(path.stem)
    except ValueError:
        return None


def image_sort_key(path: Path):
    frame_id = parse_frame_id(path)
    if frame_id is None:
        frame_id = path.stem
    return (str(path.parent), frame_id)


def get_image_files(folder: Path):
    return sorted(
        (
            file
            for file in folder.rglob("*")
            if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=image_sort_key,
    )


def load_gray_image(path: Path, max_size: int):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")

    height, width = image.shape[:2]
    scale = 1.0

    if max_size > 0:
        largest_side = max(width, height)
        if largest_side > max_size:
            scale = max_size / largest_side
            new_width = round(width * scale)
            new_height = round(height * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return image, width, height, scale


def load_thumbnail_vector(path: Path, thumb_size: int):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None

    image = cv2.resize(image, (thumb_size, thumb_size), interpolation=cv2.INTER_AREA)
    vector = image.astype(np.float32).reshape(-1)
    vector -= float(vector.mean())

    norm = float(np.linalg.norm(vector))
    if norm <= 1e-6:
        return None

    return vector / norm


def create_database(db_path: Path):
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()

    cursor.execute("PRAGMA foreign_keys = ON")
    cursor.execute("PRAGMA journal_mode = WAL")
    cursor.execute("PRAGMA synchronous = NORMAL")
    cursor.execute("PRAGMA temp_store = MEMORY")

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL UNIQUE,
            file_name TEXT NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            file_size INTEGER NOT NULL,
            modified_ns INTEGER NOT NULL,
            feature_max_size INTEGER NOT NULL,
            feature_scale REAL NOT NULL,
            keypoints INTEGER NOT NULL
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS image_features (
            image_id INTEGER PRIMARY KEY,
            keypoints_json TEXT NOT NULL,
            descriptors_blob BLOB,
            descriptor_rows INTEGER NOT NULL,
            descriptor_cols INTEGER NOT NULL,
            descriptor_dtype TEXT,
            FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS image_matches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_a_id INTEGER NOT NULL,
            image_b_id INTEGER NOT NULL,
            good_matches INTEGER NOT NULL,
            inliers INTEGER NOT NULL,
            inlier_ratio REAL NOT NULL,
            average_distance REAL NOT NULL,
            mean_reprojection_error REAL NOT NULL DEFAULT 0,
            coverage_a REAL NOT NULL,
            coverage_b REAL NOT NULL,
            confidence REAL NOT NULL,
            ratio_score REAL NOT NULL DEFAULT 0,
            inlier_score REAL NOT NULL DEFAULT 0,
            coverage_score REAL NOT NULL DEFAULT 0,
            reprojection_score REAL NOT NULL DEFAULT 0,
            homography_json TEXT,
            match_points_json TEXT,
            FOREIGN KEY (image_a_id) REFERENCES images(id) ON DELETE CASCADE,
            FOREIGN KEY (image_b_id) REFERENCES images(id) ON DELETE CASCADE,
            UNIQUE(image_a_id, image_b_id)
        )
        """
    )

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_images_path ON images(path)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_matches_pair ON image_matches(image_a_id, image_b_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_matches_confidence ON image_matches(confidence)")

    connection.commit()
    return connection


def serialize_keypoints(keypoints):
    data = [
        [
            float(kp.pt[0]),
            float(kp.pt[1]),
            float(kp.size),
            float(kp.angle),
            float(kp.response),
            int(kp.octave),
            int(kp.class_id),
        ]
        for kp in keypoints
    ]
    return json.dumps(data, separators=(",", ":"))


def deserialize_keypoints(keypoints_json: str):
    data = json.loads(keypoints_json)
    return [
        cv2.KeyPoint(
            float(item[0]),
            float(item[1]),
            float(item[2]),
            float(item[3]),
            float(item[4]),
            int(item[5]),
            int(item[6]),
        )
        for item in data
    ]


def serialize_descriptors(descriptors):
    if descriptors is None:
        return None, 0, 0, None
    descriptors = np.asarray(descriptors, dtype=np.float32)
    rows, cols = descriptors.shape
    return sqlite3.Binary(descriptors.tobytes()), rows, cols, str(descriptors.dtype)


def deserialize_descriptors(blob, rows: int, cols: int, dtype: str):
    if blob is None or rows == 0 or cols == 0 or dtype is None:
        return None
    return np.frombuffer(blob, dtype=np.dtype(dtype)).reshape((rows, cols)).copy()


def find_image_row(connection, path: Path):
    return connection.execute("SELECT * FROM images WHERE path = ?", (str(path),)).fetchone()


def load_cached_features(connection, image_id: int):
    row = connection.execute("SELECT * FROM image_features WHERE image_id = ?", (image_id,)).fetchone()
    if row is None:
        return None
    return (
        deserialize_keypoints(row["keypoints_json"]),
        deserialize_descriptors(row["descriptors_blob"], row["descriptor_rows"], row["descriptor_cols"], row["descriptor_dtype"]),
    )


def delete_matches_for_image(connection, image_id: int):
    connection.execute("DELETE FROM image_matches WHERE image_a_id = ? OR image_b_id = ?", (image_id, image_id))


def save_image_and_features(
    connection,
    image_path: Path,
    width: int,
    height: int,
    file_size: int,
    modified_ns: int,
    feature_max_size: int,
    feature_scale: float,
    keypoints,
    descriptors,
    existing_image_id=None,
):
    cursor = connection.cursor()

    if existing_image_id is None:
        cursor.execute(
            """
            INSERT INTO images (
                path, file_name, width, height, file_size, modified_ns,
                feature_max_size, feature_scale, keypoints
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(image_path),
                image_path.name,
                width,
                height,
                file_size,
                modified_ns,
                feature_max_size,
                feature_scale,
                len(keypoints),
            ),
        )
        image_id = cursor.lastrowid
    else:
        image_id = existing_image_id
        cursor.execute(
            """
            UPDATE images
            SET file_name = ?, width = ?, height = ?, file_size = ?, modified_ns = ?,
                feature_max_size = ?, feature_scale = ?, keypoints = ?
            WHERE id = ?
            """,
            (
                image_path.name,
                width,
                height,
                file_size,
                modified_ns,
                feature_max_size,
                feature_scale,
                len(keypoints),
                image_id,
            ),
        )
        cursor.execute("DELETE FROM image_features WHERE image_id = ?", (image_id,))

    keypoints_json = serialize_keypoints(keypoints)
    descriptors_blob, rows, cols, dtype = serialize_descriptors(descriptors)

    cursor.execute(
        """
        INSERT INTO image_features (
            image_id, keypoints_json, descriptors_blob,
            descriptor_rows, descriptor_cols, descriptor_dtype
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (image_id, keypoints_json, descriptors_blob, rows, cols, dtype),
    )

    return image_id


def get_or_compute_image_features(connection, sift, image_path: Path, max_size: int):
    stat = image_path.stat()
    file_size = int(stat.st_size)
    modified_ns = int(stat.st_mtime_ns)

    row = find_image_row(connection, image_path)

    if row is not None:
        cache_is_valid = (
            int(row["file_size"]) == file_size
            and int(row["modified_ns"]) == modified_ns
            and int(row["feature_max_size"]) == max_size
        )
        if cache_is_valid:
            cached = load_cached_features(connection, row["id"])
            if cached is not None:
                keypoints, descriptors = cached
                return {
                    "id": row["id"],
                    "path": image_path,
                    "width": row["width"],
                    "height": row["height"],
                    "scale": row["feature_scale"],
                    "keypoints": keypoints,
                    "descriptors": descriptors,
                    "from_cache": True,
                }

        delete_matches_for_image(connection, row["id"])
        existing_image_id = row["id"]
    else:
        existing_image_id = None

    gray, width, height, scale = load_gray_image(image_path, max_size)
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    image_id = save_image_and_features(
        connection,
        image_path,
        width,
        height,
        file_size,
        modified_ns,
        max_size,
        scale,
        keypoints,
        descriptors,
        existing_image_id=existing_image_id,
    )

    return {
        "id": image_id,
        "path": image_path,
        "width": width,
        "height": height,
        "scale": scale,
        "keypoints": keypoints,
        "descriptors": descriptors,
        "from_cache": False,
    }


def load_existing_match_pairs(connection):
    return {
        (int(row[0]), int(row[1]))
        for row in connection.execute("SELECT image_a_id, image_b_id FROM image_matches")
    }


def insert_match(connection, image_a_id: int, image_b_id: int, result):
    if image_a_id > image_b_id:
        image_a_id, image_b_id = image_b_id, image_a_id

    homography_json = json.dumps(result["homography"].tolist()) if result["homography"] is not None else None
    match_points_json = None if args.no_store_match_points else json.dumps(result["match_points"], separators=(",", ":"))

    connection.execute(
        """
        INSERT OR REPLACE INTO image_matches (
            image_a_id, image_b_id, good_matches, inliers, inlier_ratio,
            average_distance, mean_reprojection_error, coverage_a, coverage_b,
            confidence, ratio_score, inlier_score, coverage_score, reprojection_score,
            homography_json, match_points_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            image_a_id,
            image_b_id,
            result["good_matches"],
            result["inliers"],
            result["inlier_ratio"],
            result["average_distance"],
            result["mean_reprojection_error"],
            result["coverage_a"],
            result["coverage_b"],
            result["confidence"],
            result["ratio_score"],
            result["inlier_score"],
            result["coverage_score"],
            result["reprojection_score"],
            homography_json,
            match_points_json,
        ),
    )


def create_matcher():
    if args.matcher == "flann":
        index_params = dict(algorithm=1, trees=5)  # KD-tree for SIFT float descriptors.
        search_params = dict(checks=50)
        return cv2.FlannBasedMatcher(index_params, search_params)

    return cv2.BFMatcher(cv2.NORM_L2)


def get_thread_matcher():
    matcher = getattr(thread_data, "matcher", None)
    if matcher is None:
        matcher = create_matcher()
        thread_data.matcher = matcher
    return matcher


def ratio_matches(desc_a, desc_b, matcher, ratio: float):
    if desc_a is None or desc_b is None or len(desc_a) < 2 or len(desc_b) < 2:
        return []

    raw_matches = matcher.knnMatch(desc_a, desc_b, k=2)
    good_matches = []

    for match_pair in raw_matches:
        if len(match_pair) < 2:
            continue
        first, second = match_pair
        if first.distance < ratio * second.distance:
            good_matches.append(first)

    return good_matches


def mutual_ratio_matches(desc_a, desc_b, matcher, ratio: float):
    forward = ratio_matches(desc_a, desc_b, matcher, ratio)
    backward = ratio_matches(desc_b, desc_a, matcher, ratio)
    backward_pairs = {(match.trainIdx, match.queryIdx) for match in backward}
    return [match for match in forward if (match.queryIdx, match.trainIdx) in backward_pairs]


def points_coverage(points, width: int, height: int) -> float:
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


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def score_between(value: float, bad: float, good: float) -> float:
    if good == bad:
        return 1.0 if value >= good else 0.0
    return clamp01((value - bad) / (good - bad))


def score_lower_is_better(value: float, good: float, bad: float) -> float:
    if bad == good:
        return 1.0 if value <= good else 0.0
    return clamp01((bad - value) / (bad - good))


def calculate_reprojection_error(src_points_np, dst_points_np, homography, mask_bool):
    inlier_src = src_points_np[mask_bool].reshape(-1, 1, 2)
    inlier_dst = dst_points_np[mask_bool].reshape(-1, 2)
    if len(inlier_src) == 0:
        return float("inf")
    projected = cv2.perspectiveTransform(inlier_src, homography).reshape(-1, 2)
    errors = np.linalg.norm(projected - inlier_dst, axis=1)
    return float(errors.mean())


def calculate_confidence(inliers, inlier_ratio, coverage_a, coverage_b, mean_reprojection_error):
    min_coverage = min(coverage_a, coverage_b)
    ratio_score = score_between(inlier_ratio, bad=0.25, good=0.75)
    inlier_score = score_between(math.log1p(inliers), bad=math.log1p(10), good=math.log1p(80))
    coverage_score = score_between(min_coverage, bad=0.03, good=0.20)
    reprojection_score = score_lower_is_better(mean_reprojection_error, good=1.5, bad=8.0)
    confidence = 0.35 * ratio_score + 0.20 * inlier_score + 0.25 * coverage_score + 0.20 * reprojection_score
    return {
        "confidence": clamp01(confidence),
        "ratio_score": ratio_score,
        "inlier_score": inlier_score,
        "coverage_score": coverage_score,
        "reprojection_score": reprojection_score,
    }


def match_images(image_a, image_b):
    kp_a = image_a["keypoints"]
    desc_a = image_a["descriptors"]
    scale_a = image_a["scale"]
    kp_b = image_b["keypoints"]
    desc_b = image_b["descriptors"]
    scale_b = image_b["scale"]

    if desc_a is None or desc_b is None or len(kp_a) < 8 or len(kp_b) < 8:
        return None

    matcher = get_thread_matcher()
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

    src_points_np = np.float32(src_points).reshape(-1, 1, 2)
    dst_points_np = np.float32(dst_points).reshape(-1, 1, 2)

    method = cv2.USAC_MAGSAC if hasattr(cv2, "USAC_MAGSAC") else cv2.RANSAC
    homography, mask = cv2.findHomography(src_points_np, dst_points_np, method, args.ransac)

    if homography is None or mask is None:
        return None

    mask = mask.ravel().astype(bool)
    inlier_matches = [match for match, is_inlier in zip(good_matches, mask) if is_inlier]
    inliers = len(inlier_matches)

    if inliers < args.min_inliers:
        return None

    inlier_ratio = inliers / len(good_matches)
    if inlier_ratio < args.min_inlier_ratio:
        return None

    mean_reprojection_error = calculate_reprojection_error(src_points_np, dst_points_np, homography, mask)
    if mean_reprojection_error > args.max_reprojection_error:
        return None

    inlier_points_a = []
    inlier_points_b = []

    for match in inlier_matches:
        point_a = kp_a[match.queryIdx].pt
        point_b = kp_b[match.trainIdx].pt
        inlier_points_a.append([point_a[0] / scale_a, point_a[1] / scale_a])
        inlier_points_b.append([point_b[0] / scale_b, point_b[1] / scale_b])

    coverage_a = points_coverage(inlier_points_a, image_a["width"], image_a["height"])
    coverage_b = points_coverage(inlier_points_b, image_b["width"], image_b["height"])

    if coverage_a < args.min_coverage or coverage_b < args.min_coverage:
        return None

    average_distance = sum(match.distance for match in inlier_matches) / inliers
    confidence_data = calculate_confidence(inliers, inlier_ratio, coverage_a, coverage_b, mean_reprojection_error)
    confidence = confidence_data["confidence"]

    if confidence < args.min_confidence:
        return None

    match_points = []
    if not args.no_store_match_points:
        for match in inlier_matches:
            point_a = kp_a[match.queryIdx].pt
            point_b = kp_b[match.trainIdx].pt
            match_points.append(
                {
                    "a": [point_a[0] / scale_a, point_a[1] / scale_a],
                    "b": [point_b[0] / scale_b, point_b[1] / scale_b],
                    "distance": float(match.distance),
                }
            )

    return {
        "good_matches": len(good_matches),
        "inliers": inliers,
        "inlier_ratio": inlier_ratio,
        "average_distance": average_distance,
        "mean_reprojection_error": mean_reprojection_error,
        "coverage_a": coverage_a,
        "coverage_b": coverage_b,
        "confidence": confidence,
        "ratio_score": confidence_data["ratio_score"],
        "inlier_score": confidence_data["inlier_score"],
        "coverage_score": confidence_data["coverage_score"],
        "reprojection_score": confidence_data["reprojection_score"],
        "homography": homography,
        "match_points": match_points,
    }


def build_same_folder_candidates(images, window: int):
    candidates = set()
    if window <= 0:
        return candidates

    groups = defaultdict(list)
    for index, image in enumerate(images):
        groups[str(image["path"].parent)].append(index)

    for indexes in groups.values():
        indexes.sort(key=lambda idx: image_sort_key(images[idx]["path"]))

        for pos, left_index in enumerate(indexes):
            max_pos = min(len(indexes), pos + window + 1)
            for right_pos in range(pos + 1, max_pos):
                a = left_index
                b = indexes[right_pos]
                if a > b:
                    a, b = b, a
                candidates.add((a, b))

    return candidates


def build_cross_folder_candidates(images, top_k: int, thumb_size: int, min_similarity: float):
    candidates = set()
    if top_k <= 0:
        return candidates

    print("Building thumbnail candidates for cross-folder matches...")

    vectors = []
    valid_indexes = []
    parents = []

    for index, image in enumerate(images):
        vector = load_thumbnail_vector(image["path"], thumb_size)
        if vector is None:
            continue
        vectors.append(vector)
        valid_indexes.append(index)
        parents.append(str(image["path"].parent))

    if len(vectors) < 2:
        return candidates

    matrix = np.vstack(vectors).astype(np.float32)
    parents = np.array(parents)
    valid_indexes = np.array(valid_indexes, dtype=np.int32)

    chunk_size = 512

    for start in range(0, len(matrix), chunk_size):
        end = min(len(matrix), start + chunk_size)
        sims = matrix[start:end] @ matrix.T

        for local_row, row_index in enumerate(range(start, end)):
            sims[local_row, row_index] = -np.inf
            sims[local_row, parents == parents[row_index]] = -np.inf

            if top_k >= len(matrix):
                candidate_positions = np.argsort(-sims[local_row])
            else:
                candidate_positions = np.argpartition(-sims[local_row], top_k)[:top_k]
                candidate_positions = candidate_positions[np.argsort(-sims[local_row][candidate_positions])]

            for other_pos in candidate_positions:
                similarity = float(sims[local_row, other_pos])
                if not np.isfinite(similarity):
                    continue
                if similarity < min_similarity:
                    continue

                a = int(valid_indexes[row_index])
                b = int(valid_indexes[other_pos])
                if a == b:
                    continue
                if a > b:
                    a, b = b, a
                candidates.add((a, b))

        print(f"  Thumbnail candidate progress: {end}/{len(matrix)}")

    return candidates


def pair_id_tuple(images, pair):
    a_index, b_index = pair
    a_id = int(images[a_index]["id"])
    b_id = int(images[b_index]["id"])
    if a_id > b_id:
        a_id, b_id = b_id, a_id
    return a_id, b_id


def match_pair(pair):
    a_index, b_index = pair
    return pair, match_images(images[a_index], images[b_index])


image_files = get_image_files(INPUT)
print(f"Found {len(image_files)} images")
if len(image_files) < 2:
    raise RuntimeError("Need at least 2 images to build a match graph.")

connection = create_database(DB_PATH)

if args.clear_matches:
    print("Clearing existing matches, keeping cached image features...")
    connection.execute("DELETE FROM image_matches")
    connection.commit()

sift_feature_count = 0 if args.max_features <= 0 else args.max_features
sift = cv2.SIFT_create(nfeatures=sift_feature_count)

images = []
cache_hits = 0
cache_misses = 0

print("Loading or extracting SIFT features...")
for index, image_path in enumerate(image_files, start=1):
    try:
        image_data = get_or_compute_image_features(connection, sift, image_path, args.max_size)
        images.append(image_data)
        if image_data["from_cache"]:
            cache_hits += 1
            source = "cache"
        else:
            cache_misses += 1
            source = "computed"

        print(f"[{index}/{len(image_files)}] {image_path} | keypoints: {len(image_data['keypoints'])} | {source}")
        if index % 25 == 0:
            connection.commit()
    except Exception as error:
        print(f"Skipping image: {image_path} | {error}")

connection.commit()

print()
print("Feature cache summary:")
print(f"  cache hits: {cache_hits}")
print(f"  computed:   {cache_misses}")

print()
print("Building candidate pairs...")
same_candidates = build_same_folder_candidates(images, args.same_folder_window)
cross_candidates = build_cross_folder_candidates(
    images,
    args.cross_folder_top_k,
    args.thumb_size,
    args.min_thumb_similarity,
)

candidate_pairs = sorted(same_candidates | cross_candidates)

if args.candidate_limit > 0:
    candidate_pairs = candidate_pairs[: args.candidate_limit]

existing_pairs = set()
if not args.clear_matches:
    existing_pairs = load_existing_match_pairs(connection)
    candidate_pairs = [pair for pair in candidate_pairs if pair_id_tuple(images, pair) not in existing_pairs]

print(f"Same-folder candidates:  {len(same_candidates)}")
print(f"Cross-folder candidates: {len(cross_candidates)}")
print(f"Total candidates:       {len(candidate_pairs)}")
print(f"Old brute-force pairs:  {len(images) * (len(images) - 1) // 2}")
print()
print("Matching candidate pairs...")
print(f"matcher={args.matcher}")
print(f"workers={args.workers}")
print(f"max_size={args.max_size}")
print(f"max_features={args.max_features}")
print(f"same_folder_window={args.same_folder_window}")
print(f"cross_folder_top_k={args.cross_folder_top_k}")
print(f"store_match_points={not args.no_store_match_points}")

checked_pairs = 0
saved_edges = 0

if args.workers <= 1:
    for pair in candidate_pairs:
        checked_pairs += 1
        result = match_images(images[pair[0]], images[pair[1]])

        if result is not None:
            insert_match(connection, images[pair[0]]["id"], images[pair[1]]["id"], result)
            saved_edges += 1
            print(
                f"Strong match: {images[pair[0]]['path']} <-> {images[pair[1]]['path']} | "
                f"good={result['good_matches']} | inliers={result['inliers']} | "
                f"ratio={result['inlier_ratio']:.2f} | "
                f"coverage=({result['coverage_a']:.2f}, {result['coverage_b']:.2f}) | "
                f"reproj={result['mean_reprojection_error']:.2f}px | "
                f"confidence={result['confidence']:.2f}"
            )

        if checked_pairs % 250 == 0:
            connection.commit()
            print(f"Checked {checked_pairs}/{len(candidate_pairs)} candidates | edges={saved_edges}")
else:
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(match_pair, pair) for pair in candidate_pairs]

        for future in as_completed(futures):
            pair, result = future.result()
            checked_pairs += 1

            if result is not None:
                insert_match(connection, images[pair[0]]["id"], images[pair[1]]["id"], result)
                saved_edges += 1
                print(
                    f"Strong match: {images[pair[0]]['path']} <-> {images[pair[1]]['path']} | "
                    f"good={result['good_matches']} | inliers={result['inliers']} | "
                    f"ratio={result['inlier_ratio']:.2f} | "
                    f"coverage=({result['coverage_a']:.2f}, {result['coverage_b']:.2f}) | "
                    f"reproj={result['mean_reprojection_error']:.2f}px | "
                    f"confidence={result['confidence']:.2f}"
                )

            if checked_pairs % 250 == 0:
                connection.commit()
                print(f"Checked {checked_pairs}/{len(candidate_pairs)} candidates | edges={saved_edges}")

connection.commit()
connection.close()

print()
print("Done.")
print(f"Database: {DB_PATH}")
print(f"Images stored/loaded: {len(images)}")
print(f"Feature cache hits: {cache_hits}")
print(f"Feature cache computed: {cache_misses}")
print(f"Candidate pairs checked: {checked_pairs}")
print(f"Strong matches saved: {saved_edges}")
