"""
#STEP1:
#-----
python remove_suffix.py #set the folder first

#STEP2:
#-----
python align_replace_edgesafe.py \
  --img1_dir ./Dataset_512_2d-3d/masked_image2 \
  --img2_dir ./Dataset_512_2d-3d/my_output \
  --out_dir  ./Dataset_512_2d-3d/refined_output \
  --align translation \
  --black_thresh 10 \
  --erode 3 \
  --feather 4

python align_replace_edgesafe.py \
  --img1_dir ./masked_image \
  --img2_dir ./model_output \
  --out_dir  ./refined_R_small\
  --align translation \
  --black_thresh 10 \
  --erode 3 \
  --feather 5

#STEP3:
#-----
python align_side_by_side.py ./img1/1.png ./refined_output/1.png vr_sbs.png

python align_side_by_side.py ./img1 ./refined_output ./LR_merge
"""
# align_replace_edgesafe.py
import os, glob, argparse
import numpy as np
import cv2
from tqdm import tqdm

EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# ---------------------------- I/O helpers ----------------------------

def imread_color(p):
    img = cv2.imread(p, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(p)
    return img

def imread_gray(p):
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(p)
    return img

def find_pairs(img1_dir, img2_dir):
    """
    Pair files by filename. If img2/<name> is missing, tries <name>_inpaint.<ext>.
    Returns list of (img1_path, img2_path).
    """
    files = []
    for ext in EXTS:
        files += glob.glob(os.path.join(img1_dir, f"*{ext}"))
    files.sort()

    pairs = []
    for p1 in files:
        name = os.path.basename(p1)
        stem, ext = os.path.splitext(name)

        p2 = os.path.join(img2_dir, name)
        if not os.path.exists(p2):
            alt = os.path.join(img2_dir, f"{stem}_inpaint{ext}")
            if os.path.exists(alt):
                p2 = alt
            else:
                continue
        pairs.append((p1, p2))
    return pairs

# ------------------------ alignment (ECC) ----------------------------

def ecc_register(img1, img2, valid_mask=None, mode="translation", iters=200, eps=1e-5):
    """
    Estimate warp that maps img1 -> img2 using ECC.
    valid_mask: HxW uint8 (0/255). Only trusted regions of img1 should be 255.
    mode: 'translation' (2 DoF) or 'affine' (6 DoF).
    Returns (warpMatrix 2x3, motionType, ok_flag).
    """
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if img1.ndim == 3 else img1
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if img2.ndim == 3 else img2
    g1 = g1.astype(np.float32) / 255.0
    g2 = g2.astype(np.float32) / 255.0

    mask = (valid_mask > 0).astype(np.uint8) if valid_mask is not None else None

    if mode == "affine":
        warp_mode = cv2.MOTION_AFFINE
    else:
        warp_mode = cv2.MOTION_TRANSLATION

    W = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, eps)
    try:
        _, W = cv2.findTransformECC(
            templateImage=g2,
            inputImage=g1,
            warpMatrix=W,
            motionType=warp_mode,
            criteria=criteria,
            inputMask=mask,
            gaussFiltSize=5
        )
        return W, warp_mode, True
    except cv2.error:
        return np.eye(2, 3, dtype=np.float32), warp_mode, False

def warp_affine_img(img, W, dsize, border=cv2.BORDER_REPLICATE):
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    return cv2.warpAffine(img, W, dsize, flags=flags, borderMode=border)

def warp_affine_mask(mask, W, dsize):
    # NEAREST + constant 0 keeps mask binary and avoids gray halos
    flags = cv2.WARP_INVERSE_MAP | cv2.INTER_NEAREST
    return cv2.warpAffine(mask, W, dsize, flags=flags,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)

# ---------------------- masks & compositing --------------------------

def non_black_mask(img, thresh=10):
    """
    Binary mask (0/255) where img is NOT black.
    thresh is on luma (safer than sum of BGR).
    """
    if img.ndim == 3:
        y = (0.114*img[...,0] + 0.587*img[...,1] + 0.299*img[...,2]).astype(np.float32)
    else:
        y = img.astype(np.float32)
    return (y > thresh).astype(np.uint8) * 255

def refine_mask(mask_warped, erode_px=3, feather_px=4):
    """
    1) Erode a little to remove the contaminated rim.
    2) Optional soft edge (feather) for seamless transition.
    Returns alpha float HxWx1 in [0,1].
    """
    core = mask_warped
    if erode_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*erode_px+1, 2*erode_px+1))
        core = cv2.erode(core, k)

    if feather_px <= 0:
        return (core > 127).astype(np.float32)[..., None]

    # Build a soft transition between core (1) and outside (0)
    kf = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*feather_px+1, 2*feather_px+1))
    dil = cv2.dilate(core, kf)
    ring = (dil > 0).astype(np.uint8) * 255

    dist_in  = cv2.distanceTransform(core,       cv2.DIST_L2, 3)
    dist_out = cv2.distanceTransform(255 - ring, cv2.DIST_L2, 3)
    alpha = dist_in / (dist_in + dist_out + 1e-6)

    alpha[(core == 0) & (ring == 0)] = 0.0
    alpha[(core == 255)] = 1.0
    return alpha[..., None].astype(np.float32)

def compose_edgesafe(img1, img2, W, warp_mode,
                     black_thresh=10, erode_px=3, feather_px=4):
    """
    img2 is base. Replace only where img1 (after alignment) is non-black.
    Edge-safe via: mask-from-original -> warp mask (NEAREST) -> erode -> feather.
    """
    H, Wd = img2.shape[:2]

    # 1) mask from original img1 (non-black = trusted content)
    m_nb = non_black_mask(img1, thresh=black_thresh)  # 0/255

    # 2) warp img1 and mask into img2 frame
    img1_warp = warp_affine_img(img1, W, (Wd, H), border=cv2.BORDER_REPLICATE)
    m_warp    = warp_affine_mask(m_nb, W, (Wd, H))

    # 3) refine mask and build alpha
    alpha = refine_mask(m_warp, erode_px=erode_px, feather_px=feather_px)  # HxWx1 float

    # 4) composite (alpha=1 -> take img1_warp; alpha=0 -> keep img2)
    out = (img2.astype(np.float32) * (1.0 - alpha) +
           img1_warp.astype(np.float32) * alpha).astype(np.uint8)
    return out

# --------------------------- pipeline -------------------------------

def process_one(img1_path, img2_path, out_dir, align_mode="translation",
                black_thresh=10, erode_px=3, feather_px=4, resize_to="img2"):
    """
    img1: holey image
    img2: inpainted image (base)
    """
    img1 = imread_color(img1_path)
    img2 = imread_color(img2_path)

    # Unify sizes (default: resize img1 to img2 size)
    if resize_to == "img2":
        H, W = img2.shape[:2]
        if img1.shape[:2] != (H, W):
            img1 = cv2.resize(img1, (W, H), interpolation=cv2.INTER_CUBIC)
    elif resize_to == "img1":
        H, W = img1.shape[:2]
        if img2.shape[:2] != (H, W):
            img2 = cv2.resize(img2, (W, H), interpolation=cv2.INTER_CUBIC)
    else:
        raise ValueError("resize_to must be 'img2' or 'img1'")

    # Build valid mask from img1 (non-black -> 255, holes -> 0)
    valid_mask = non_black_mask(img1, thresh=black_thresh)

    # ECC alignment (img1 -> img2) using only valid areas
    Wmat, wm, ok = ecc_register(img1, img2, valid_mask=valid_mask, mode=align_mode)
    # If ECC fails, Wmat is identity (no alignment)

    # Edge-safe compose
    out = compose_edgesafe(img1, img2, Wmat, wm,
                           black_thresh=black_thresh,
                           erode_px=erode_px, feather_px=feather_px)

    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, os.path.basename(img1_path)), out)

# ---------------------------- CLI -----------------------------------

def main():
    ap = argparse.ArgumentParser(description="Align-and-replace (edge-safe): img2 base, paste non-black from img1.")
    ap.add_argument("--img1_dir", required=True, help="Folder of holey images (img1).")
    ap.add_argument("--img2_dir", required=True, help="Folder of inpaint images (img2).")
    ap.add_argument("--out_dir",  required=True, help="Output folder.")
    ap.add_argument("--align", choices=["translation","affine"], default="translation",
                    help="ECC alignment model (start with translation).")
    ap.add_argument("--black_thresh", type=int, default=10,
                    help="Luma threshold: <= thresh treated as black hole.")
    ap.add_argument("--erode", type=int, default=3,
                    help="Erode pixels in warped mask to drop contaminated rim (px).")
    ap.add_argument("--feather", type=int, default=4,
                    help="Feather width (px) to soften edge; 0 disables.")
    ap.add_argument("--resize_to", choices=["img2","img1"], default="img2",
                    help="Resize which image so sizes match (default: resize img1 to img2).")
    args = ap.parse_args()

    pairs = find_pairs(args.img1_dir, args.img2_dir)
    if not pairs:
        print("No matching filenames found. Ensure img1 and img2 share names.")
        return

    for p1, p2 in tqdm(pairs, desc="Compositing", unit="img"):
        try:
            process_one(
                p1, p2, args.out_dir,
                align_mode=args.align,
                black_thresh=args.black_thresh,
                erode_px=args.erode,
                feather_px=args.feather,
                resize_to=args.resize_to
            )
        except Exception as e:
            print(f"[WARN] Skipped {p1}: {e}")

    print(f"Done. Saved {len(pairs)} images to: {args.out_dir}")

if __name__ == "__main__":
    main()
