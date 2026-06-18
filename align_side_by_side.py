import cv2
import numpy as np
import sys
import os

# valid extensions to look for
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

def make_sbs(left_path, right_path, out_path, order="LR"):
    """
    Processes a single pair of images and saves the result.
    """
    # Read images
    imgL = cv2.imread(left_path, cv2.IMREAD_COLOR)
    imgR = cv2.imread(right_path, cv2.IMREAD_COLOR)

    if imgL is None:
        print(f"Warning: Could not read left image: {left_path}")
        return False
    if imgR is None:
        print(f"Warning: Could not read right image: {right_path}")
        return False

    # Resize both to the same size (use min height/width)
    h = min(imgL.shape[0], imgR.shape[0])
    w = min(imgL.shape[1], imgR.shape[1])

    imgL = cv2.resize(imgL, (w, h), interpolation=cv2.INTER_AREA)
    imgR = cv2.resize(imgR, (w, h), interpolation=cv2.INTER_AREA)

    # Concatenate horizontally
    if order.upper() == "LR":
        sbs = np.hstack((imgL, imgR))  # [Left | Right]
    else:
        sbs = np.hstack((imgR, imgL))  # [Right | Left]

    # Save result
    cv2.imwrite(out_path, sbs)
    return True

def process_folders(left_dir, right_dir, out_dir, order="LR"):
    """
    Iterates through the left directory and looks for matching files in the right directory.
    """
    # 1. Create output directory if it doesn't exist
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f"Created output directory: {out_dir}")

    # 2. Get list of files in left directory
    try:
        left_files = sorted(os.listdir(left_dir))
    except FileNotFoundError:
        print(f"Error: The directory '{left_dir}' does not exist.")
        sys.exit(1)

    processed_count = 0

    print(f"Processing images from '{left_dir}' and '{right_dir}'...")
    print("-" * 50)

    # 3. Loop through files
    for filename in left_files:
        # Check if file is an image based on extension
        ext = os.path.splitext(filename)[1].lower()
        if ext not in VALID_EXTENSIONS:
            continue

        # Construct full paths
        l_path = os.path.join(left_dir, filename)
        r_path = os.path.join(right_dir, filename) # Assumes filename is identical
        out_path = os.path.join(out_dir, filename)

        # Check if corresponding right file exists
        if not os.path.exists(r_path):
            print(f"[SKIP] Match not found for: {filename}")
            continue

        # Process the pair
        success = make_sbs(l_path, r_path, out_path, order)
        
        if success:
            print(f"[OK] Saved: {filename}")
            processed_count += 1
        else:
            print(f"[FAIL] Could not process: {filename}")

    print("-" * 50)
    print(f"Done. Processed {processed_count} pairs.")
    print(f"Output folder: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python make_sbs.py <left_folder> <right_folder> [output_folder] [LR|RL]")
        sys.exit(1)

    left_folder = sys.argv[1]
    right_folder = sys.argv[2]
    
    # Default output folder name if not provided
    out_folder = sys.argv[3] if len(sys.argv) > 3 else "left_right_merged"
    
    # Default order if not provided
    order_mode = sys.argv[4] if len(sys.argv) > 4 else "LR"

    process_folders(left_folder, right_folder, out_folder, order_mode)