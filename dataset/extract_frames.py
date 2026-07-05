import argparse
from pathlib import Path
import cv2

parser = argparse.ArgumentParser(description="Script to process videos into jpeg images.")

parser.add_argument(
    "-i",
    "--input",
    required=True,
    help="Path to an input video file or folder containing videos",
)
parser.add_argument(
    "-f",
    "--fpm",
    type=int,
    required=True,
    help="Number of frames to extract each minute",
)
parser.add_argument(
    "-o",
    "--output",
    required=True,
    help="Path to the output folder",
)

args = parser.parse_args()

INPUT = Path(args.input)
FPM = args.fpm
OUTPUT = Path(args.output)

if FPM <= 0:
    raise ValueError("FPM must be greater than 0")


def extract_video_frames(video_path: Path, output_root: Path, fpm: int):
    video_output_dir = output_root / video_path.stem
    video_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting frames from: {video_path}")
    print(f"Output folder: {video_output_dir}")

    video = cv2.VideoCapture(str(video_path))

    if not video.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    video_fps = video.get(cv2.CAP_PROP_FPS)

    print("fps =", video_fps)

    if video_fps <= 0:
        raise RuntimeError(f"Could not read video FPS: {video_path}")

    # Count-based frame interval
    frames_per_minute = video_fps * 60
    frame_interval = round(frames_per_minute / fpm)

    if frame_interval <= 0:
        frame_interval = 1

    print("frame interval =", frame_interval)

    frame_index = 0
    saved_count = 0

    while True:
        success, frame = video.read()

        if not success:
            break

        if frame_index % frame_interval == 0:
            source_frame_index = frame_index + 1
            output_path = video_output_dir / f"{source_frame_index}.jpeg"

            cv2.imwrite(
                str(output_path),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )

            saved_count += 1

        frame_index += 1

    video.release()

    print(f"Done. Saved {saved_count} frames from {video_path.name}")


OUTPUT.mkdir(parents=True, exist_ok=True)

if INPUT.is_file():
    if INPUT.suffix.lower() != ".mp4":
        raise ValueError(f"Input file must be .mp4: {INPUT}")

    videos = [INPUT]

elif INPUT.is_dir():
    videos = sorted(
        file
        for file in INPUT.iterdir()
        if file.is_file() and file.suffix.lower() == ".mp4"
    )

    if len(videos) == 0:
        raise ValueError(f"No .mp4 files found in folder: {INPUT}")

else:
    raise FileNotFoundError(f"Input path does not exist: {INPUT}")

print()
print(f"Videos found: {len(videos)}")

for video_path in videos:
    print(f"\nExtracting video #{videos.index(video_path)}")
    extract_video_frames(video_path, OUTPUT, FPM)

print()
print(f"Finished extracting all videos into: {OUTPUT}")