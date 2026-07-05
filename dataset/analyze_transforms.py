import argparse
import json
import math
import re
import sqlite3
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser(description="Analyze relative image transforms using DJI SRT telemetry.")

parser.add_argument("-db", "--database", default="graph.db", help="Path to graph.db")
parser.add_argument("-tp", "--telemetry-path", required=True, help="Folder containing DJI .SRT files")
parser.add_argument("--base-fov", type=float, default=60.0, help="Base camera FOV in degrees")
parser.add_argument("--use-dzoom", action="store_true", help="Adjust FOV using dzoom_ratio if available")

args = parser.parse_args()

DB_PATH = Path(args.database)
TELEMETRY_PATH = Path(args.telemetry_path)

EARTH_RADIUS = 6378137.0


def effective_fov(base_fov_deg: float, dzoom_ratio):
    if dzoom_ratio is None:
        return base_fov_deg

    zoom = dzoom_ratio / 10000.0

    if zoom <= 0:
        return base_fov_deg

    base_rad = math.radians(base_fov_deg)
    effective_rad = 2 * math.atan(math.tan(base_rad / 2) / zoom)

    return math.degrees(effective_rad)


def parse_float(pattern: str, text: str):
    match = re.search(pattern, text)

    if not match:
        return None

    return float(match.group(1))


def parse_int(pattern: str, text: str):
    match = re.search(pattern, text)

    if not match:
        return None

    return int(match.group(1))


def parse_dji_srt_file(srt_path: Path, base_fov: float):
    """
    Returns:
    {
        frame_id: {
            "video": "DJI_0006",
            "frame": 4345,
            "lat": 32.104819,
            "lon": 35.212160,
            "rel_alt": 120.0,
            "abs_alt": 805.041,
            "focal_len": 240,
            "dzoom_ratio": 10000,
            "fov": 60.0,
        }
    }
    """

    video_name = srt_path.stem
    text = srt_path.read_text(encoding="utf-8", errors="ignore")

    blocks = re.split(r"\n\s*\n", text)

    frames = {}

    for block in blocks:
        srt_count = parse_int(r"SrtCnt\s*:\s*(\d+)", block)

        if srt_count is None:
            continue

        lat = parse_float(r"\[latitude:\s*([-+]?\d+(?:\.\d+)?)\]", block)
        lon = parse_float(r"\[longitude:\s*([-+]?\d+(?:\.\d+)?)\]", block)
        rel_alt = parse_float(r"\[rel_alt:\s*([-+]?\d+(?:\.\d+)?)", block)
        abs_alt = parse_float(r"abs_alt:\s*([-+]?\d+(?:\.\d+)?)\]", block)

        focal_len = parse_int(r"\[focal_len\s*:\s*(\d+)\]", block)
        dzoom_ratio = parse_int(r"\[dzoom_ratio:\s*(\d+)", block)

        timestamp_match = re.search(
            r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)",
            block,
        )

        timestamp = timestamp_match.group(1) if timestamp_match else None

        if lat is None or lon is None or rel_alt is None:
            continue

        if args.use_dzoom:
            fov = effective_fov(base_fov, dzoom_ratio)
        else:
            fov = base_fov

        frames[srt_count] = {
            "video": video_name,
            "frame": srt_count,
            "timestamp": timestamp,
            "lat": lat,
            "lon": lon,
            "rel_alt": rel_alt,
            "abs_alt": abs_alt,
            "focal_len": focal_len,
            "dzoom_ratio": dzoom_ratio,
            "fov": fov,
        }

    return frames


def load_telemetry_folder(folder: Path, base_fov: float):
    if not folder.exists():
        raise FileNotFoundError(f"Telemetry folder does not exist: {folder}")

    if not folder.is_dir():
        raise ValueError(f"Telemetry path must be a folder: {folder}")

    telemetry = {}

    srt_files = sorted(
        file for file in folder.iterdir()
        if file.is_file() and file.suffix.lower() == ".srt"
    )

    print(f"Found {len(srt_files)} SRT files")

    for srt_file in srt_files:
        frames = parse_dji_srt_file(srt_file, base_fov)

        for frame_id, data in frames.items():
            telemetry[(srt_file.stem, frame_id)] = data

        print(f"{srt_file.name}: {len(frames)} telemetry frames")

    return telemetry


def image_key_from_path(path_text: str):
    path = Path(path_text.replace("\\", "/"))

    video = path.parent.name

    try:
        frame = int(path.stem)
    except ValueError:
        raise ValueError(f"Image filename must be numeric frame id: {path}")

    return video, frame


def latlon_to_local_meters(lat, lon, origin_lat, origin_lon):
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)

    origin_lat_rad = math.radians(origin_lat)
    origin_lon_rad = math.radians(origin_lon)

    east = (lon_rad - origin_lon_rad) * math.cos(origin_lat_rad) * EARTH_RADIUS
    north = (lat_rad - origin_lat_rad) * EARTH_RADIUS

    return east, north


def apply_homography(H, x, y):
    point = np.array([x, y, 1.0], dtype=np.float64)
    mapped = H @ point

    if abs(mapped[2]) < 1e-9:
        return None

    return mapped[0] / mapped[2], mapped[1] / mapped[2]


def homography_center_shift_and_scale(H, width_a, height_a, width_b, height_b):
    cx_a = width_a / 2
    cy_a = height_a / 2

    cx_b = width_b / 2
    cy_b = height_b / 2

    mapped_center = apply_homography(H, cx_a, cy_a)

    if mapped_center is None:
        return None

    mapped_x, mapped_y = mapped_center

    dx_px = cx_b - mapped_x
    dy_px = cy_b - mapped_y

    step = 10.0

    p0 = apply_homography(H, cx_a, cy_a)
    px = apply_homography(H, cx_a + step, cy_a)
    py = apply_homography(H, cx_a, cy_a + step)

    if p0 is None or px is None or py is None:
        return None

    scale_x = math.dist(p0, px) / step
    scale_y = math.dist(p0, py) / step
    scale = math.sqrt(scale_x * scale_y)

    rotation_rad = math.atan2(px[1] - p0[1], px[0] - p0[0])
    rotation_deg = math.degrees(rotation_rad)

    return dx_px, dy_px, scale, rotation_deg


def estimate_altitude_from_scale(alt_a, fov_a, fov_b, scale_a_to_b):
    if scale_a_to_b <= 0:
        return None

    fov_a_rad = math.radians(fov_a)
    fov_b_rad = math.radians(fov_b)

    return alt_a * math.tan(fov_a_rad / 2) / (
        scale_a_to_b * math.tan(fov_b_rad / 2)
    )


def column_exists(cursor, table_name, column_name):
    cursor.execute(f'PRAGMA table_info("{table_name}")')
    return any(row[1] == column_name for row in cursor.fetchall())


telemetry = load_telemetry_folder(TELEMETRY_PATH, args.base_fov)

if len(telemetry) == 0:
    raise RuntimeError("No telemetry loaded from SRT files.")

origin = next(iter(telemetry.values()))
origin_lat = origin["lat"]
origin_lon = origin["lon"]

connection = sqlite3.connect(DB_PATH)
cursor = connection.cursor()

has_mean_reprojection_error = column_exists(cursor, "image_matches", "mean_reprojection_error")
has_reprojection_error = column_exists(cursor, "image_matches", "reprojection_error")

if has_mean_reprojection_error:
    reprojection_select = "m.mean_reprojection_error"
elif has_reprojection_error:
    reprojection_select = "m.reprojection_error"
else:
    reprojection_select = "NULL"

cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS edge_transforms (
        match_id INTEGER PRIMARY KEY,

        image_a_path TEXT NOT NULL,
        image_b_path TEXT NOT NULL,

        video_a TEXT NOT NULL,
        frame_a INTEGER NOT NULL,
        video_b TEXT NOT NULL,
        frame_b INTEGER NOT NULL,

        dx_px REAL NOT NULL,
        dy_px REAL NOT NULL,
        scale_a_to_b REAL NOT NULL,
        rotation_deg REAL NOT NULL,

        lat_a REAL NOT NULL,
        lon_a REAL NOT NULL,
        alt_a REAL NOT NULL,

        lat_b REAL NOT NULL,
        lon_b REAL NOT NULL,
        alt_b REAL NOT NULL,

        gps_east_delta REAL NOT NULL,
        gps_north_delta REAL NOT NULL,
        gps_alt_delta REAL NOT NULL,

        meters_per_pixel REAL NOT NULL,

        estimated_alt_b REAL,
        alt_error REAL,

        fov_a REAL,
        fov_b REAL,
        dzoom_a INTEGER,
        dzoom_b INTEGER,

        confidence REAL,
        inliers INTEGER,
        inlier_ratio REAL,
        mean_reprojection_error REAL
    )
    """
)

cursor.execute("DELETE FROM edge_transforms")

cursor.execute(
    f"""
    SELECT
        m.id,

        a.path,
        b.path,

        a.width,
        a.height,
        b.width,
        b.height,

        m.homography_json,
        m.confidence,
        m.inliers,
        m.inlier_ratio,

        {reprojection_select} AS mean_reprojection_error

    FROM image_matches m
    JOIN images a ON a.id = m.image_a_id
    JOIN images b ON b.id = m.image_b_id
    WHERE m.homography_json IS NOT NULL
    """
)

rows = cursor.fetchall()

saved = 0
skipped_missing_telemetry = 0
skipped_bad_transform = 0
skipped_other = 0

for row in rows:
    (
        match_id,
        image_a_path,
        image_b_path,
        width_a,
        height_a,
        width_b,
        height_b,
        homography_json,
        confidence,
        inliers,
        inlier_ratio,
        mean_reprojection_error,
    ) = row

    try:
        video_a, frame_a = image_key_from_path(image_a_path)
        video_b, frame_b = image_key_from_path(image_b_path)

        key_a = (video_a, frame_a)
        key_b = (video_b, frame_b)

        if key_a not in telemetry or key_b not in telemetry:
            skipped_missing_telemetry += 1
            continue

        gps_a = telemetry[key_a]
        gps_b = telemetry[key_b]

        H = np.array(json.loads(homography_json), dtype=np.float64)

        transform = homography_center_shift_and_scale(
            H,
            width_a,
            height_a,
            width_b,
            height_b,
        )

        if transform is None:
            skipped_bad_transform += 1
            continue

        dx_px, dy_px, scale_a_to_b, rotation_deg = transform

        east_a, north_a = latlon_to_local_meters(
            gps_a["lat"],
            gps_a["lon"],
            origin_lat,
            origin_lon,
        )

        east_b, north_b = latlon_to_local_meters(
            gps_b["lat"],
            gps_b["lon"],
            origin_lat,
            origin_lon,
        )

        gps_east_delta = east_b - east_a
        gps_north_delta = north_b - north_a
        gps_alt_delta = gps_b["rel_alt"] - gps_a["rel_alt"]

        pixel_distance = math.hypot(dx_px, dy_px)
        meter_distance = math.hypot(gps_east_delta, gps_north_delta)

        if pixel_distance <= 1e-6:
            skipped_bad_transform += 1
            continue

        meters_per_pixel = meter_distance / pixel_distance

        estimated_alt_b = estimate_altitude_from_scale(
            gps_a["rel_alt"],
            gps_a["fov"],
            gps_b["fov"],
            scale_a_to_b,
        )

        alt_error = None

        if estimated_alt_b is not None:
            alt_error = estimated_alt_b - gps_b["rel_alt"]

        cursor.execute(
            """
            INSERT OR REPLACE INTO edge_transforms (
                match_id,

                image_a_path,
                image_b_path,

                video_a,
                frame_a,
                video_b,
                frame_b,

                dx_px,
                dy_px,
                scale_a_to_b,
                rotation_deg,

                lat_a,
                lon_a,
                alt_a,

                lat_b,
                lon_b,
                alt_b,

                gps_east_delta,
                gps_north_delta,
                gps_alt_delta,

                meters_per_pixel,

                estimated_alt_b,
                alt_error,

                fov_a,
                fov_b,
                dzoom_a,
                dzoom_b,

                confidence,
                inliers,
                inlier_ratio,
                mean_reprojection_error
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                match_id,

                image_a_path,
                image_b_path,

                video_a,
                frame_a,
                video_b,
                frame_b,

                dx_px,
                dy_px,
                scale_a_to_b,
                rotation_deg,

                gps_a["lat"],
                gps_a["lon"],
                gps_a["rel_alt"],

                gps_b["lat"],
                gps_b["lon"],
                gps_b["rel_alt"],

                gps_east_delta,
                gps_north_delta,
                gps_alt_delta,

                meters_per_pixel,

                estimated_alt_b,
                alt_error,

                gps_a["fov"],
                gps_b["fov"],
                gps_a["dzoom_ratio"],
                gps_b["dzoom_ratio"],

                confidence,
                inliers,
                inlier_ratio,
                mean_reprojection_error,
            ),
        )

        saved += 1

    except Exception as error:
        print(f"Skipping match {match_id}: {error}")
        skipped_other += 1

connection.commit()

print()
print("Done.")
print(f"Saved edge transforms: {saved}")
print(f"Skipped missing telemetry: {skipped_missing_telemetry}")
print(f"Skipped bad transform: {skipped_bad_transform}")
print(f"Skipped other: {skipped_other}")

cursor.execute(
    """
    SELECT
        match_id,
        video_a,
        frame_a,
        video_b,
        frame_b,
        dx_px,
        dy_px,
        scale_a_to_b,
        gps_east_delta,
        gps_north_delta,
        gps_alt_delta,
        meters_per_pixel,
        confidence
    FROM edge_transforms
    ORDER BY confidence DESC
    LIMIT 20
    """
)

print()
print("Top transforms:")

for row in cursor.fetchall():
    print(row)

connection.close()