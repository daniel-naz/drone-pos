import argparse
from pathlib import Path
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".jpen", ".png"}


parser = argparse.ArgumentParser(
    description="Remove closely similar images from a folder."
)

parser.add_argument(
    "-i",
    "--input",
    required=True,
    help="Input folder. Can contain nested folders.",
)

parser.add_argument(
    "-p",
    "--percent",
    type=float,
    required=True,
    help="Similarity percentage from 0 to 100. Example: 95 removes very similar images.",
)

args = parser.parse_args()

INPUT = Path(args.input)
PERCENT = args.percent

if not INPUT.exists():
    raise FileNotFoundError(f"Input folder does not exist: {INPUT}")

if not INPUT.is_dir():
    raise ValueError(f"Input must be a folder: {INPUT}")

if PERCENT < 0 or PERCENT > 100:
    raise ValueError("Percent must be between 0 and 100")


def get_image_files(folder: Path):
    return sorted(
        file
        for file in folder.rglob("*")
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    )


def dhash(image_path: Path, hash_size: int = 8) -> int:
    """
    Creates a simple perceptual hash.
    Similar images should have similar hashes.
    """
    with Image.open(image_path) as img:
        img = img.convert("L")
        img = img.resize((hash_size + 1, hash_size))

        pixels = list(img.getdata())

    result = 0

    for row in range(hash_size):
        for col in range(hash_size):
            left = pixels[row * (hash_size + 1) + col]
            right = pixels[row * (hash_size + 1) + col + 1]

            result <<= 1

            if left > right:
                result |= 1

    return result


def similarity_percent(hash_a: int, hash_b: int, bits: int = 64) -> float:
    different_bits = (hash_a ^ hash_b).bit_count()
    same_bits = bits - different_bits

    return (same_bits / bits) * 100


images = get_image_files(INPUT)

print(f"Found {len(images)} images")
print(f"Similarity threshold: {PERCENT}%")
print()

kept_images = []
deleted_count = 0
skipped_count = 0

for image_path in images:
    try:
        current_hash = dhash(image_path)
    except Exception as error:
        print(f"Skipping unreadable image: {image_path} | {error}")
        skipped_count += 1
        continue

    duplicate_of = None
    duplicate_similarity = 0

    for kept_path, kept_hash in kept_images:
        similarity = similarity_percent(current_hash, kept_hash)

        if similarity >= PERCENT:
            duplicate_of = kept_path
            duplicate_similarity = similarity
            break

    if duplicate_of is not None:
        print(f"Deleting: {image_path}")
        print(f"  Similar to: {duplicate_of}")
        print(f"  Similarity: {duplicate_similarity:.2f}%")

        image_path.unlink()
        deleted_count += 1
    else:
        kept_images.append((image_path, current_hash))

print()
print("Done.")
print(f"Kept: {len(kept_images)}")
print(f"Deleted: {deleted_count}")
print(f"Skipped: {skipped_count}")