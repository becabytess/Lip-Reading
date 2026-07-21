# import os
# import numpy as np
# import cv2
# from tqdm import tqdm

# # ===== CONFIG =====
# ROOT_DIR = os.path.join('raw_dataset','1','data')
# FPS = 25
# DELETE_NPY = True   # ⚠️ set True AFTER verifying everything works
# # ==================

# def convert_npy_to_mp4(npy_path):
#     try:
#         arr = np.load(npy_path)

#         # ---- Fix shape ----
#         # (1, T, H, W) -> (T, H, W)
#         if arr.ndim == 4 and arr.shape[0] == 1:
#             arr = arr.squeeze(0)

#         # (T, H, W, 1) -> (T, H, W)
#         if arr.ndim == 4 and arr.shape[-1] == 1:
#             arr = arr[..., 0]

#         if arr.ndim != 3:
#             print(f"Skipping: {npy_path}, shape={arr.shape}")
#             return

#         T, H, W = arr.shape

#         # Ensure uint8
#         if arr.dtype != np.uint8:
#             arr = arr.astype(np.uint8)

#         # Output path (same directory)
#         mp4_path = os.path.splitext(npy_path)[0] + ".mp4"

#         fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#         out = cv2.VideoWriter(mp4_path, fourcc, FPS, (W, H))

#         for i in range(T):
#             frame = arr[i]

#             # grayscale -> BGR
#             frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

#             out.write(frame)

#         out.release()

#         # ---- Optional delete ----
#         if DELETE_NPY:
#             os.remove(npy_path)

#     except Exception as e:
#         print(f"Error: {npy_path}, {e}")


# def main():
#     npy_files = []

#     # Collect all .npy files
#     for root, _, files in os.walk(ROOT_DIR):
#         for f in files:
#             if f.endswith(".npy"):
#                 npy_files.append(os.path.join(root, f))

#     print(f"Found {len(npy_files)} files")

#     # Convert
#     for path in tqdm(npy_files):
#         convert_npy_to_mp4(path)


# if __name__ == "__main__":
#     main()

