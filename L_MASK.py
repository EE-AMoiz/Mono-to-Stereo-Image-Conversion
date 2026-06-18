import cv2
import os
import numpy as np
import time
from tqdm import tqdm
import argparse
import multiprocessing
from functools import partial
import gc

VALID_EXTS = ('.png', '.jpg', '.jpeg')


def _list_clean_images(folder):
    names = []
    for f in os.listdir(folder):
        fl = f.lower()
        if not fl.endswith(VALID_EXTS):
            continue
        if f.startswith("._"):
            continue
        names.append(f)
    return names


def _stem(name):
    return os.path.splitext(name)[0]


def forward_warp_x(src_img, src_disp, scale_factor, direction_sign, shift_mult=1.0,
                   valid_src=None, write_disp=False):
    """
    Forward-warp along x using disparity, with optional gating by valid_src.

    shift_pixels = shift_mult * (disp / scale_factor)

    direction_sign:
      +1 => shift RIGHT  (new_x = x + offset)
      -1 => shift LEFT   (new_x = x - offset)

    valid_src:
      uint8 HxW 0/255 map. If provided, only pixels with 255 are warped.
    """
    h, w = src_img.shape[:2]
    dst_img = np.zeros_like(src_img)
    dst_valid = np.zeros((h, w), dtype=np.uint8)
    dst_disp = np.zeros((h, w), dtype=src_disp.dtype) if write_disp else None

    ys = np.arange(h, dtype=np.int32)

    sf = float(scale_factor)
    if sf <= 0:
        raise ValueError("scale_factor must be > 0")

    shift_mult = float(shift_mult)

    # back-to-front order helps occlusions
    x_range = range(w - 1, -1, -1) if direction_sign > 0 else range(w)

    for x in x_range:
        if valid_src is not None:
            src_ok = (valid_src[ys, x] == 255)
        else:
            src_ok = np.ones(h, dtype=bool)

        disp_values = (src_disp[ys, x].astype(np.float32) / sf) * shift_mult
        floor_offset = np.floor(disp_values).astype(np.int32)
        ceil_offset = np.ceil(disp_values).astype(np.int32)

        # floor landing
        new_x_floor = x + direction_sign * floor_offset
        okf = (new_x_floor >= 0) & (new_x_floor < w) & src_ok
        if np.any(okf):
            y = ys[okf]
            nx = new_x_floor[okf]
            dst_img[y, nx] = src_img[y, x]
            dst_valid[y, nx] = 255
            if write_disp:
                dst_disp[y, nx] = src_disp[y, x]

        # ceil landing
        new_x_ceil = x + direction_sign * ceil_offset
        okc = (new_x_ceil >= 0) & (new_x_ceil < w) & src_ok
        if np.any(okc):
            y = ys[okc]
            nx = new_x_ceil[okc]
            dst_img[y, nx] = src_img[y, x]
            dst_valid[y, nx] = 255
            if write_disp:
                dst_disp[y, nx] = src_disp[y, x]

    return dst_img, dst_valid, dst_disp


class TwoPassLeftView:
    """
    Two-pass forward-warp that creates LEFT-view style disocclusions (holes on left of objects),
    with explicit translation control.

    MIRROR OF TwoPassRightView:

    Pass 1: shift LEFT  by 1.0 * disp/scale_factor
    Pass 2: shift RIGHT by translation * disp_tmp/scale_factor

    IMPORTANT FIX:
      Pass 2 only warps pixels that are valid from Pass 1 (prevents “black pixels” from holes
      being moved into valid areas and not covered by the mask).
    """

    def __init__(self, img_bgr, disp, scale_factor=6.0, translation=2.0, mask_dilate=0):
        self.img = img_bgr
        if disp.ndim == 3:
            disp = cv2.cvtColor(disp, cv2.COLOR_BGR2GRAY)
        self.disp = disp

        self.scale_factor = float(scale_factor)
        self.translation = float(translation)
        self.mask_dilate = int(mask_dilate)

        self.tmp_img = None
        self.tmp_valid = None
        self.tmp_disp = None

        self.left_view = None
        self.left_valid = None
        self.mask = None
        self.masked_left_view = None

    def run(self):
        # ---- Pass 1: LEFT shift (reference, fixed at 1.0) ----
        self.tmp_img, self.tmp_valid, self.tmp_disp = forward_warp_x(
            src_img=self.img,
            src_disp=self.disp,
            scale_factor=self.scale_factor,
            direction_sign=-1,   # MIRROR
            shift_mult=1.0,
            valid_src=None,
            write_disp=True
        )

        # ---- Pass 2: RIGHT shift (controllable translation) ----
        self.left_view, self.left_valid, _ = forward_warp_x(
            src_img=self.tmp_img,
            src_disp=self.tmp_disp,
            scale_factor=self.scale_factor,
            direction_sign=+1,   # MIRROR
            shift_mult=self.translation,
            valid_src=self.tmp_valid,     # CRITICAL FIX (same as yours)
            write_disp=False
        )

        # Holes come ONLY from landing map (not pixel color)
        self.mask = (self.left_valid == 0).astype(np.uint8) * 255

        # Optional: slightly expand mask to cover 1px borders/dots
        if self.mask_dilate > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            self.mask = cv2.dilate(self.mask, k, iterations=self.mask_dilate)

        # Force perfect alignment: black exactly where mask says hole
        self.masked_left_view = self.left_view.copy()
        self.masked_left_view[self.mask == 255] = 0

    def write_all(self, out_root, frame_idx):
        os.makedirs(os.path.join(out_root, "orig"), exist_ok=True)
        os.makedirs(os.path.join(out_root, "masked_image"), exist_ok=True)
        os.makedirs(os.path.join(out_root, "mask"), exist_ok=True)

        cv2.imwrite(os.path.join(out_root, "orig", f"{frame_idx:05d}.png"), self.img)
        cv2.imwrite(os.path.join(out_root, "masked_image", f"{frame_idx:05d}.png"), self.masked_left_view)
        cv2.imwrite(os.path.join(out_root, "mask", f"{frame_idx:05d}_mask.png"), self.mask)


def _process_one(pair, save_path, scale_factor, translation, mask_dilate):
    try:
        cv2.setNumThreads(0)
    except Exception:
        pass

    img_path, disp_path, idx = pair
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    disp = cv2.imread(disp_path, cv2.IMREAD_UNCHANGED)

    if img is None or disp is None:
        print(f"[WARN] Could not read:\n  img: {img_path}\n  disp: {disp_path}")
        return False

    try:
        proc = TwoPassLeftView(
            img_bgr=img,
            disp=disp,
            scale_factor=scale_factor,
            translation=translation,
            mask_dilate=mask_dilate
        )
        proc.run()
        proc.write_all(save_path, idx)
    finally:
        del img, disp, proc
        gc.collect()

    return True


def process_images(orig_folder, disp_folder, save_path, workers=None, start_index=0,
                   scale_factor=6.0, translation=2.0, mask_dilate=0):

    if workers is None:
        workers = max(1, multiprocessing.cpu_count() - 1)

    orig_names = _list_clean_images(orig_folder)
    disp_names = _list_clean_images(disp_folder)

    orig_map = {_stem(f): os.path.join(orig_folder, f) for f in orig_names}
    disp_map = {_stem(f): os.path.join(disp_folder, f) for f in disp_names}

    keys = sorted(set(orig_map.keys()) & set(disp_map.keys()))
    total = len(keys)
    if total == 0:
        print("No matching filename stems.")
        return

    print(f"Pairs: {total}")
    print(f"scale_factor={scale_factor}  translation={translation}  mask_dilate={mask_dilate}")
    print(f"NOTE: net shift ~ (translation-1) * disp/scale_factor (rightward if translation>1).")

    def pair_iter():
        for i, k in enumerate(keys):
            yield (orig_map[k], disp_map[k], start_index + i)

    func = partial(_process_one, save_path=save_path,
                   scale_factor=scale_factor, translation=translation, mask_dilate=mask_dilate)

    pool = multiprocessing.Pool(processes=workers, maxtasksperchild=200)
    pbar = tqdm(total=total, desc="Processing", unit="img")
    t0 = time.time()

    try:
        for _ok in pool.imap_unordered(func, pair_iter(), chunksize=8):
            pbar.update(1)
            if pbar.n % 20 == 0:
                elapsed = time.time() - t0
                pbar.set_postfix({"FPS": f"{pbar.n / max(1e-6, elapsed):.1f}"})
    finally:
        pool.close()
        pool.join()
        pbar.close()


if __name__ == "__main__":
    try:
        cv2.setNumThreads(0)
    except Exception:
        pass

    parser = argparse.ArgumentParser("Two-pass warp LEFT-view synthesis with translation control.")
    parser.add_argument("--orig", required=True, help="Folder with original images")
    parser.add_argument("--disp", required=True, help="Folder with disparity maps")
    parser.add_argument("--output", required=True, help="Output folder")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)

    # Your controls:
    parser.add_argument("--scale-factor", type=float, default=6.0,
                        help="Shift divisor: shift = disp/scale_factor. Bigger => smaller shifts.")
    parser.add_argument("--translation", type=float, default=2.0,
                        help="Controls final right shift strength (left view). Use >1.0 for net right shift.")
    parser.add_argument("--mask-dilate", type=int, default=0,
                        help="Dilate mask N iterations to cover tiny borders/dots (image is blackened accordingly).")

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    process_images(
        args.orig, args.disp, args.output,
        workers=args.workers,
        start_index=args.start_index,
        scale_factor=args.scale_factor,
        translation=args.translation,
        mask_dilate=args.mask_dilate
    )

