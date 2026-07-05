import csv
import re
import subprocess
from pathlib import Path


INPUT_CSV = Path("unprocessed/unprocessed.csv")
VIDEOS_DIR = Path("unprocessed/videos")
SRT_DIR = Path("unprocessed/SRT")
PREPROCESS_CSV = Path("preprocess.csv")

FORMAT = (
    "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/"
    "best[height<=1080][ext=mp4]/"
    "best[height<=1080]"
)


def safe_filename(name: str) -> str:
    name = name.strip()
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    name = re.sub(r"\s+", "_", name)
    return name[:120] or "video"


def find_downloaded_video(name: str) -> Path | None:
    candidates = [
        VIDEOS_DIR / f"{name}.mp4",
        VIDEOS_DIR / f"{name}.mkv",
        VIDEOS_DIR / f"{name}.webm",
    ]

    for path in candidates:
        if path.exists() and path.stat().st_size > 0:
            return path

    matches = list(VIDEOS_DIR.glob(f"{name}.*"))
    matches = [p for p in matches if p.is_file() and p.stat().st_size > 0]

    if not matches:
        return None

    return matches[0]


def normalize_srt_path(srt_value: str) -> str:
    srt_value = (srt_value or "").strip()

    if not srt_value:
        return ""

    srt_path = Path(srt_value)

    # If CSV says DJI_0006, turn it into unprocessed/SRT/DJI_0006
    if not srt_path.is_absolute() and len(srt_path.parts) == 1:
        srt_path = SRT_DIR / srt_path

    return str(srt_path).replace("\\", "/")


def main():
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

    successful_rows = []

    with open(INPUT_CSV, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            raw_name = row["name"]
            link = row["link"].strip()
            fps = str(row.get("fps", "")).strip()
            srt = str(row.get("SRT", "")).strip()

            name = safe_filename(raw_name)
            output_template = str(VIDEOS_DIR / f"{name}.%(ext)s")

            cmd = [
                "py", "-m", "yt_dlp",
                "--no-playlist",
                "--force-overwrites",
                "--no-continue",
                "-f", FORMAT,
                "--merge-output-format", "mp4",
                "-o", output_template,
                link,
            ]

            print(f"\nDownloading {name}")
            print(link)

            result = subprocess.run(cmd)

            if result.returncode != 0:
                print(f"[FAIL] Download failed: {name}")
                continue

            downloaded_video = find_downloaded_video(name)

            if downloaded_video is None:
                print(f"[FAIL] Download command finished, but video file was not found: {name}")
                continue

            video_path_for_csv = str(downloaded_video).replace("\\", "/")
            srt_path_for_csv = normalize_srt_path(srt)

            successful_rows.append({
                "video_name": video_path_for_csv,
                "fps": fps,
                "SRT": srt_path_for_csv,
            })

            print(f"[OK] {video_path_for_csv}")

    with open(PREPROCESS_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_name", "fps", "SRT"])
        writer.writeheader()
        writer.writerows(successful_rows)

    print("\nDone.")
    print(f"Successful downloads: {len(successful_rows)}")
    print(f"Created: {PREPROCESS_CSV}")


if __name__ == "__main__":
    main()