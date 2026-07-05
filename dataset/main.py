import argparse
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser(
    description="Script to create a full drone navigation dataset"
)

# SHARED ARGS
parser.add_argument(
    "-ff",
    "--frame-folder",
    required=False,
    help="Path to the frames folder",
    default="dataset/frames",
)

# FRAME EXTRACTION ARGS
parser.add_argument(
    "-cis",
    "--create-images-set",
    action="store_true",
    help="Recreate the image frames from the input videos.",
)

parser.add_argument(
    "-i",
    "--input",
    required=False,
    help="Path to the input video file or folder",
    default="unprocessed/videos",
)

parser.add_argument(
    "-f",
    "--fpm",
    type=int,
    required=False,
    help="Number of frames to extract each minute",
    default=60,
)

# REMOVE SIMILAR ARGS
parser.add_argument(
    "-rs",
    "--remove-similar",
    action="store_true",
    help="Remove closely similar images from the frames folder.",
)

parser.add_argument(
    "-p",
    "--percent",
    type=int,
    required=False,
    help="Similarity percentage from 0 to 100.",
    default=95,
)

# MAKE MATCHES ARGS
parser.add_argument(
    "-mm",
    "--make-matches",
    action="store_true",
    help="Build image match graph database from the frames folder.",
)

parser.add_argument(
    "-db",
    "--database",
    required=False,
    help="Path to the graph database file.",
    default="graph.db",
)

parser.add_argument(
    "--same-folder-window",
    type=int,
    required=False,
    help="How many nearby frames to compare inside the same video folder.",
    default=30,
)

parser.add_argument(
    "--cross-folder-top-k",
    type=int,
    required=False,
    help="How many likely cross-video matches to check per image.",
    default=50,
)

parser.add_argument(
    "--max-size",
    type=int,
    required=False,
    help="Resize largest image side before SIFT.",
    default=1000,
)

parser.add_argument(
    "--max-features",
    type=int,
    required=False,
    help="Maximum SIFT features per image.",
    default=2000,
)

parser.add_argument(
    "--workers",
    type=int,
    required=False,
    help="Number of parallel workers for matching.",
    default=6,
)

parser.add_argument(
    "--clear-matches",
    action="store_true",
    help="Clear existing matches but keep cached SIFT features.",
)

parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Delete and recreate the graph database.",
)

parser.add_argument(
    "--store-match-points",
    action="store_true",
    help="Store full SIFT inlier match points. Bigger DB.",
)

args = parser.parse_args()

BASE_DIR = Path(__file__).resolve().parent

frame_folder = Path(args.frame_folder)

extract_script = BASE_DIR / "extract_frames.py"
remove_similar_script = BASE_DIR / "remove_similar.py"
match_script = BASE_DIR / "build_match_graph_fast.py"


# HANDLE FRAME EXTRACTION
if args.create_images_set:
    if not extract_script.exists():
        raise FileNotFoundError(f"Could not find script: {extract_script}")

    print("Extracting frames with:")
    print(f"\tInput path: {args.input}")
    print(f"\tFrame folder: {frame_folder}")
    print(f"\tFPM: {args.fpm}")

    subprocess.run(
        [
            sys.executable,
            str(extract_script),
            "-i",
            args.input,
            "-f",
            str(args.fpm),
            "-o",
            str(frame_folder),
        ],
        check=True,
    )


# HANDLE REMOVE SIMILAR
if args.remove_similar:
    if not remove_similar_script.exists():
        raise FileNotFoundError(f"Could not find script: {remove_similar_script}")

    if not frame_folder.exists():
        raise FileNotFoundError(f"Frame folder does not exist: {frame_folder}")

    print("Removing similar images with:")
    print(f"\tFrame folder: {frame_folder}")
    print(f"\tSimilarity percent: {args.percent}")

    subprocess.run(
        [
            sys.executable,
            str(remove_similar_script),
            "-i",
            str(frame_folder),
            "-p",
            str(args.percent),
        ],
        check=True,
    )


# HANDLE MAKE MATCHES
if args.make_matches:
    if not match_script.exists():
        raise FileNotFoundError(f"Could not find script: {match_script}")

    if not frame_folder.exists():
        raise FileNotFoundError(f"Frame folder does not exist: {frame_folder}")

    print("Making image match graph with:")
    print(f"\tFrame folder: {frame_folder}")
    print(f"\tDatabase: {args.database}")
    print(f"\tSame-folder window: {args.same_folder_window}")
    print(f"\tCross-folder top K: {args.cross_folder_top_k}")
    print(f"\tMax size: {args.max_size}")
    print(f"\tMax features: {args.max_features}")
    print(f"\tWorkers: {args.workers}")

    command = [
        sys.executable,
        str(match_script),
        "-i",
        str(frame_folder),
        "-db",
        args.database,
        "--same-folder-window",
        str(args.same_folder_window),
        "--cross-folder-top-k",
        str(args.cross_folder_top_k),
        "--max-size",
        str(args.max_size),
        "--max-features",
        str(args.max_features),
        "--workers",
        str(args.workers),
    ]

    if args.clear_matches:
        command.append("--clear-matches")

    if args.overwrite:
        command.append("--overwrite")

    if not args.store_match_points:
        command.append("--no-store-match-points")

    subprocess.run(command, check=True)


if not args.create_images_set and not args.remove_similar and not args.make_matches:
    print("Nothing to do.")
    print("Use -cis to extract frames.")
    print("Use -rs to remove similar images.")
    print("Use -mm to make image matches.")