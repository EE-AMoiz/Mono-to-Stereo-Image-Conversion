#!/usr/bin/env python3
"""
Batch-crop & resize every image in *src* to 512 × 512 and save them
sequentially (00000.png, 00001.png, …) in *dst*.

Supports JPG/JPEG, PNG, HEIC/HEIF.

Examples
--------
$ python rename_resize.py                              # defaults
$ python rename_resize.py --src ./orig_images          # custom source
$ python rename_resize.py --src ./imgs --dst ./out512  # both paths

python rename_resize.py --src '/home/abdulmoiz/Projects/ml-depth-pro/data/Dataset' --dst '/home/abdulmoiz/Projects/ml-depth-pro/data/Dataset_out512'
"""

from pathlib import Path
from PIL import Image
# import pillow_heif           # uncomment if you need HEIC/HEIF
from tqdm import tqdm
import argparse
import sys

# ----------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Crop centre-square, resize to 512×512, and save sequentially.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    "--src", type=Path, default=Path("./test2017"),
    help="Folder containing the original images")

parser.add_argument(
    "--dst", type=Path, default=None,
    help=("Output folder.  If omitted, a sibling of --src named "
          "'resized_512' will be created/used."))

args = parser.parse_args()

SRC: Path = args.src
DST: Path = args.dst or (SRC.parent / "resized_512")

# ----------------------------------------------------------------------
# Sanity checks & setup
# ----------------------------------------------------------------------
if not SRC.exists() or not SRC.is_dir():
    sys.exit(f"❌ Source folder not found: {SRC}")

DST.mkdir(parents=True, exist_ok=True)

ACCEPT = {".jpg", ".jpeg", ".png", ".heic", ".heif"}

def crop_center_square(im: Image.Image) -> Image.Image:
    """Take the largest centred square from an image."""
    w, h = im.size
    side = min(w, h)
    left = (w - side) // 2
    upper = (h - side) // 2
    return im.crop((left, upper, left + side, upper + side))

# ----------------------------------------------------------------------
# Gather files once so numbering is stable
# ----------------------------------------------------------------------
files = sorted(p for p in SRC.iterdir() if p.suffix.lower() in ACCEPT)

if not files:
    sys.exit(f"⚠️  No supported images found in {SRC}")

for idx, path in enumerate(tqdm(files, desc="Processing")):
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")           # unify colour mode
            img = crop_center_square(img)
            img = img.resize((512, 512), Image.LANCZOS)

            out_name = f"{idx:05d}.png"
            img.save(DST / out_name, quality=95)
    except Exception as e:
        print(f"⚠️  {path.name} skipped ➜ {e}")

print(f"✅ Done!  Saved {len(files):,} images to {DST.resolve()}")

