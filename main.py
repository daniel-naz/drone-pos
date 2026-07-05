import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np
from pyproj import Geod


GEOD = Geod(ellps="WGS84")


def read_known_images(csv_file: str):
    csv_path = Path(csv_file)
    base_dir = csv_path.parent
    rows = []

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for i, row in enumerate(reader):
            name = row.get("name") or row.get("image") or row.get("filename") or row.get("path")
            if not name:
                raise ValueError("CSV must contain a name/image/filename/path column.")

            path = Path(name)
            if not path.is_absolute():
                path = base_dir / path

            rows.append({
                "index": i,
                "name": name,
                "path": path,
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "alt": float(row["alt"]),
                "heading": float(row["heading"]) if row.get("heading") not in (None, "") else None,
            })

    if not rows:
        raise ValueError("CSV contains no images.")

    return rows


def load_image(path: Path, max_size: int | None):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    if max_size is not None:
        h, w = img.shape[:2]
        scale = min(1.0, max_size / max(h, w))

        if scale < 1.0:
            img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray


def detect_sift(gray):
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)

    if des is None or len(kp) < 6:
        raise RuntimeError("Not enough SIFT features.")

    return kp, des


def get_features(item, cache, max_size):
    key = str(item["path"])

    if key in cache:
        return cache[key]

    color, gray = load_image(item["path"], max_size)
    kp, des = detect_sift(gray)

    cache[key] = {
        "color": color,
        "gray": gray,
        "kp": kp,
        "des": des,
        "width": gray.shape[1],
        "height": gray.shape[0],
    }

    return cache[key]


def match_sift(ref_features, new_features, ratio):
    matcher = cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=5),
        dict(checks=80)
    )

    knn = matcher.knnMatch(ref_features["des"], new_features["des"], k=2)

    good = []
    for pair in knn:
        if len(pair) != 2:
            continue

        m, n = pair

        if m.distance < ratio * n.distance:
            good.append(m)

    if len(good) < 6:
        raise RuntimeError(f"Not enough good matches: {len(good)}")

    pts_ref = np.float32([ref_features["kp"][m.queryIdx].pt for m in good])
    pts_new = np.float32([new_features["kp"][m.trainIdx].pt for m in good])

    return good, pts_ref, pts_new


def estimate_transform(pts_ref, pts_new, ransac):
    # Partial affine = rotation + translation + uniform scale.
    # Good for rough visual-scale altitude estimate.
    M, inliers = cv2.estimateAffinePartial2D(
        pts_ref,
        pts_new,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac,
        maxIters=5000,
        confidence=0.995,
        refineIters=20,
    )

    if M is None or inliers is None:
        raise RuntimeError("Could not estimate transform.")

    return M, inliers


def scale_rotation(M):
    a = M[0, 0]
    c = M[1, 0]

    scale = math.sqrt(a * a + c * c)
    rotation_deg = math.degrees(math.atan2(c, a))

    return float(scale), float(rotation_deg)


def transform_point(M, x, y):
    return (
        float(M[0, 0] * x + M[0, 1] * y + M[0, 2]),
        float(M[1, 0] * x + M[1, 1] * y + M[1, 2]),
    )


def meters_per_pixel(width, height, altitude_m, hfov_deg, vfov_deg):
    hfov = math.radians(hfov_deg)
    vfov = math.radians(vfov_deg)

    ground_width = 2 * altitude_m * math.tan(hfov / 2)
    ground_height = 2 * altitude_m * math.tan(vfov / 2)

    return ground_width / width, ground_height / height


def image_shift_to_world(dx_px, dy_px, mpp_x, mpp_y, heading_deg):
    image_shift_right_m = dx_px * mpp_x
    image_shift_down_m = dy_px * mpp_y

    # Apparent image motion is opposite to camera right/left movement.
    camera_right_m = -image_shift_right_m

    # Approximation for downward-looking camera.
    camera_forward_m = image_shift_down_m

    heading = math.radians(heading_deg)

    north_m = (
        camera_forward_m * math.cos(heading)
        - camera_right_m * math.sin(heading)
    )

    east_m = (
        camera_forward_m * math.sin(heading)
        + camera_right_m * math.cos(heading)
    )

    return east_m, north_m


def move_lat_lon(lat, lon, east_m, north_m):
    dist = math.hypot(east_m, north_m)
    if dist < 1e-9:
        return lat, lon

    azimuth = math.degrees(math.atan2(east_m, north_m))
    azimuth = (azimuth + 360) % 360

    lon2, lat2, _ = GEOD.fwd(lon, lat, azimuth, dist)
    return lat2, lon2


def gps_distance_m(lat1, lon1, lat2, lon2):
    _, _, dist = GEOD.inv(lon1, lat1, lon2, lat2)
    return abs(dist)


def weighted_median(values, weights):
    pairs = sorted(zip(values, weights), key=lambda x: x[0])
    total = sum(weights)

    if total <= 0:
        return None

    acc = 0
    for value, weight in pairs:
        acc += weight
        if acc >= total / 2:
            return value

    return pairs[-1][0]


def estimate_from_reference(ref, ref_features, new_features, args, matches_dir):
    good, pts_ref, pts_new = match_sift(ref_features, new_features, args.ratio)
    M, inliers = estimate_transform(pts_ref, pts_new, args.ransac)

    inlier_count = int(inliers.sum())
    inlier_ratio = inlier_count / len(good)

    scale, rotation_deg = scale_rotation(M)

    if scale <= 0:
        raise RuntimeError("Invalid scale.")

    estimated_alt = ref["alt"] / scale

    ref_cx = ref_features["width"] / 2
    ref_cy = ref_features["height"] / 2

    new_cx = new_features["width"] / 2
    new_cy = new_features["height"] / 2

    mapped_cx, mapped_cy = transform_point(M, ref_cx, ref_cy)

    dx_px = mapped_cx - new_cx
    dy_px = mapped_cy - new_cy

    mpp_x, mpp_y = meters_per_pixel(
        new_features["width"],
        new_features["height"],
        estimated_alt,
        args.hfov,
        args.vfov
    )

    heading = ref["heading"] if ref["heading"] is not None else args.heading

    east_m, north_m = image_shift_to_world(
        dx_px,
        dy_px,
        mpp_x,
        mpp_y,
        heading
    )

    est_lat, est_lon = move_lat_lon(ref["lat"], ref["lon"], east_m, north_m)

    # Quality score: many inliers + high inlier ratio is better.
    # Penalize insane altitude/scale estimates.
    quality = inlier_count * inlier_ratio

    if estimated_alt < args.min_alt or estimated_alt > args.max_alt:
        quality *= 0.1

    result = {
        "ref_index": ref["index"],
        "ref_name": ref["name"],
        "estimated_lat": est_lat,
        "estimated_lon": est_lon,
        "estimated_alt": estimated_alt,
        "movement_east_m": east_m,
        "movement_north_m": north_m,
        "scale": scale,
        "rotation_deg": rotation_deg,
        "tx_px": float(M[0, 2]),
        "ty_px": float(M[1, 2]),
        "center_dx_px": dx_px,
        "center_dy_px": dy_px,
        "matches": len(good),
        "inliers": inlier_count,
        "inlier_ratio": inlier_ratio,
        "quality": quality,
        "heading_used": heading,
        "match_image": "",
    }

    if args.save_matches:
        draw_mask = inliers.ravel().astype(int).tolist()

        match_img = cv2.drawMatches(
            ref_features["color"],
            ref_features["kp"],
            new_features["color"],
            new_features["kp"],
            good,
            None,
            matchesMask=draw_mask,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        out_path = matches_dir / f"match_ref_{ref['index']:03d}.jpg"
        cv2.imwrite(str(out_path), match_img)
        result["match_image"] = str(out_path)

    return result


def write_results_csv(results, output_path):
    fields = [
        "ref_index",
        "ref_name",
        "estimated_lat",
        "estimated_lon",
        "estimated_alt",
        "movement_east_m",
        "movement_north_m",
        "scale",
        "rotation_deg",
        "tx_px",
        "ty_px",
        "center_dx_px",
        "center_dy_px",
        "matches",
        "inliers",
        "inlier_ratio",
        "quality",
        "heading_used",
        "match_image",
        "status",
        "error",
    ]

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for r in results:
            writer.writerow(r)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("known_csv", help="CSV with known images: name,lat,lon,alt,heading")
    parser.add_argument("new_image", help="Single new image with unknown lat/lon/alt")

    parser.add_argument("--heading", type=float, default=0.0, help="Fallback heading/yaw. 0=north, 90=east")
    parser.add_argument("--hfov", type=float, default=73.7)
    parser.add_argument("--vfov", type=float, default=53.1)

    parser.add_argument("--ratio", type=float, default=0.75)
    parser.add_argument("--ransac", type=float, default=4.0)
    parser.add_argument("--max-size", type=int, default=1600)

    parser.add_argument("--top", type=int, default=5, help="How many best references to combine")
    parser.add_argument("--min-inliers", type=int, default=12)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.15)
    parser.add_argument("--min-alt", type=float, default=1.0)
    parser.add_argument("--max-alt", type=float, default=200.0)

    parser.add_argument("--out", default="new_image_estimates.csv")
    parser.add_argument("--matches-dir", default="matches_new")
    parser.add_argument("--save-matches", action="store_true")

    args = parser.parse_args()

    known = read_known_images(args.known_csv)

    new_item = {
        "index": -1,
        "name": Path(args.new_image).name,
        "path": Path(args.new_image),
    }

    matches_dir = Path(args.matches_dir)
    matches_dir.mkdir(parents=True, exist_ok=True)

    cache = {}
    new_features = get_features(new_item, cache, args.max_size)

    results = []

    for ref in known:
        try:
            ref_features = get_features(ref, cache, args.max_size)
            result = estimate_from_reference(ref, ref_features, new_features, args, matches_dir)

            if (
                result["inliers"] < args.min_inliers
                or result["inlier_ratio"] < args.min_inlier_ratio
            ):
                result["quality"] *= 0.2

            result["status"] = "ok"
            result["error"] = ""

            print(
                f"[OK] {ref['name']} | "
                f"inliers={result['inliers']} ratio={result['inlier_ratio']:.2f} "
                f"quality={result['quality']:.2f} | "
                f"est=({result['estimated_lat']:.8f}, {result['estimated_lon']:.8f}, {result['estimated_alt']:.2f}m)"
            )

            results.append(result)

        except Exception as e:
            print(f"[FAIL] {ref['name']} | {e}")
            results.append({
                "ref_index": ref["index"],
                "ref_name": ref["name"],
                "status": "fail",
                "error": str(e),
            })

    ok = [r for r in results if r.get("status") == "ok" and r.get("quality", 0) > 0]
    ok = sorted(ok, key=lambda r: r["quality"], reverse=True)

    write_results_csv(results, args.out)

    print("\n=== Final estimate ===")

    if not ok:
        print("No valid reference image matched the new image.")
        print(f"Saved raw results to: {args.out}")
        return

    selected = ok[:args.top]

    weights = [r["quality"] for r in selected]

    final_lat = weighted_median([r["estimated_lat"] for r in selected], weights)
    final_lon = weighted_median([r["estimated_lon"] for r in selected], weights)
    final_alt = weighted_median([r["estimated_alt"] for r in selected], weights)

    print(f"Used top {len(selected)} reference matches.")
    print(f"Estimated new image latitude:  {final_lat:.8f}")
    print(f"Estimated new image longitude: {final_lon:.8f}")
    print(f"Estimated new image altitude:  {final_alt:.3f} m")
    print(f"Best reference image: {selected[0]['ref_name']}")
    print(f"Best quality: {selected[0]['quality']:.2f}")
    print(f"Saved all estimates to: {args.out}")

    if args.save_matches:
        print(f"Saved match images to: {matches_dir}")


if __name__ == "__main__":
    main()