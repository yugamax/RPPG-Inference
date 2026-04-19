import os
import cv2
import numpy as np
from scipy.signal import butter, filtfilt, detrend
import mediapipe as mp
from concurrent.futures import ThreadPoolExecutor, as_completed

DATA_ROOT = "data/UBFC_DATASET/DATASET_2"
SAVE_DIR  = "processed"
os.makedirs(SAVE_DIR, exist_ok=True)

def make_face_mesh():
    face_mesh_module = getattr(mp.solutions, "face_mesh")
    return face_mesh_module.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

FOREHEAD    = [10, 67, 103, 109, 338, 297, 332, 284]
LEFT_CHEEK  = [50, 101, 118, 119, 120, 121]
RIGHT_CHEEK = [280, 330, 347, 348, 349, 350]
NOSE        = [4, 5, 195, 197, 6]
CHIN        = [152, 175, 199, 200, 208, 428]

ROI_WEIGHTS = {
    "forehead":    1.5,
    "left_cheek":  1.2,
    "right_cheek": 1.2,
    "nose":        0.8,
    "chin":        0.6,
}

def fill_roi(mask, landmark_obj, indices, h, w):
    pts = []
    for i in indices:
        p = landmark_obj.landmark[i]
        pts.append([int(p.x * w), int(p.y * h)])
    pts  = np.array(pts, np.int32)
    hull = cv2.convexHull(pts)
    cv2.fillConvexPoly(mask, hull, 255)


def get_roi_mean(frame_rgb, face_mesh_instance):
    h, w, _ = frame_rgb.shape
    result   = face_mesh_instance.process(frame_rgb)
    if not result.multi_face_landmarks:
        return None

    lm = result.multi_face_landmarks[0]

    roi_defs = [
        (FOREHEAD,    ROI_WEIGHTS["forehead"]),
        (LEFT_CHEEK,  ROI_WEIGHTS["left_cheek"]),
        (RIGHT_CHEEK, ROI_WEIGHTS["right_cheek"]),
        (NOSE,        ROI_WEIGHTS["nose"]),
        (CHIN,        ROI_WEIGHTS["chin"]),
    ]

    weighted_sum = np.zeros(3, dtype=np.float64)
    total_weight = 0.0

    for indices, weight in roi_defs:
        mask = np.zeros((h, w), dtype=np.uint8)
        fill_roi(mask, lm, indices, h, w)
        pixels = frame_rgb[mask == 255]
        if len(pixels) == 0:
            continue
        weighted_sum  += weight * pixels.mean(axis=0)
        total_weight  += weight

    if total_weight == 0:
        return None

    return (weighted_sum / total_weight).astype(np.float32)


def pos_method(rgb_signal):
    eps   = 1e-8
    C     = rgb_signal.T.astype(np.float64)

    mu    = C.mean(axis=1, keepdims=True)
    C_n   = C / (mu + eps)

    S     = np.array([[0, 1, -1],
                      [-2, 1,  1]], dtype=np.float64)
    P     = S @ C_n

    std1  = P[0].std() + eps
    std2  = P[1].std() + eps
    h     = P[0] + (std1 / std2) * P[1]

    return h.astype(np.float32)


def chrom_method(rgb_signal):
    eps = 1e-8
    R   = rgb_signal[:, 0].astype(np.float64)
    G   = rgb_signal[:, 1].astype(np.float64)
    B   = rgb_signal[:, 2].astype(np.float64)

    Xs  = 3 * R - 2 * G
    Ys  = 1.5 * R + G - 1.5 * B

    std_x = Xs.std() + eps
    std_y = Ys.std() + eps
    h     = Xs - (std_x / std_y) * Ys

    return h.astype(np.float32)


def bandpass(sig, fs, lo=0.7, hi=4.0):
    nyq    = fs / 2.0
    lo_adj = max(lo, 0.01)
    hi_adj = min(hi, nyq * 0.95)
    if lo_adj >= hi_adj:
        raise ValueError(f"Invalid bandpass range for fs={fs}: lo={lo_adj}, hi={hi_adj}")
    ret = butter(3, [lo_adj / nyq, hi_adj / nyq], btype='band')
    if isinstance(ret, tuple) and len(ret) == 2:
        b, a = ret
    else:
        b, a = ret[0], ret[1]
    return filtfilt(b, a, sig)


def normalize(sig):
    sig = detrend(sig)
    return (sig - sig.mean()) / (sig.std() + 1e-8)


def normalize_rgb(rgb_signal):
    out = rgb_signal.copy().astype(np.float32)
    for c in range(3):
        out[:, c] = normalize(out[:, c])
    return out


def parse_ground_truth(gt):
    arr = np.asarray(gt, dtype=np.float64)
    if arr.ndim == 1:
        raise ValueError("ground_truth.txt is 1D; expected at least 3 series")
    if arr.shape[0] >= 3:
        ppg, t = arr[0], arr[2]
    elif arr.shape[1] >= 3:
        ppg, t = arr[:, 0], arr[:, 2]
    else:
        raise ValueError(f"Unsupported ground-truth shape: {arr.shape}")

    ppg   = np.asarray(ppg, dtype=np.float64).ravel()
    t     = np.asarray(t,   dtype=np.float64).ravel()
    valid = np.isfinite(ppg) & np.isfinite(t)
    ppg, t = ppg[valid], t[valid]

    if len(ppg) < 2 or len(t) < 2:
        raise ValueError("Not enough valid GT samples")

    order  = np.argsort(t)
    t, ppg = t[order], ppg[order]

    if np.allclose(t[0], t[-1]):
        raise ValueError("GT timestamps are constant")

    return ppg, t


def process_subject(subj):
    subj_path  = os.path.join(DATA_ROOT, subj)
    video_path = os.path.join(subj_path, "vid.avi")
    gt_path    = os.path.join(subj_path, "ground_truth.txt")

    if not os.path.exists(video_path) or not os.path.exists(gt_path):
        return f"[SKIP] {subj} — missing files"

    try:
        gt        = np.loadtxt(gt_path)
        ppg, t    = parse_ground_truth(gt)
    except Exception as e:
        return f"[SKIP] {subj} — bad ground truth ({e})"

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        return f"[SKIP] {subj} — bad FPS"

    face_mesh_instance = make_face_mesh()

    rgb_signal = []
    timestamps = []
    bad_frames = 0
    idx        = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mean_rgb  = get_roi_mean(frame_rgb, face_mesh_instance)

            if mean_rgb is None:
                bad_frames += 1
                mean_rgb    = rgb_signal[-1] if rgb_signal else np.zeros(3, dtype=np.float32)

            rgb_signal.append(mean_rgb)
            timestamps.append(idx / fps)
            idx += 1
    finally:
        cap.release()
        face_mesh_instance.close()

    if len(rgb_signal) < 64:
        return f"[SKIP] {subj} — too short ({len(rgb_signal)} frames)"

    rgb_signal = np.array(rgb_signal, dtype=np.float32)
    timestamps = np.array(timestamps, dtype=np.float64)

    aligned_ppg = np.interp(timestamps, t, ppg)
    try:
        aligned_ppg = bandpass(aligned_ppg, fps)
    except ValueError as e:
        return f"[SKIP] {subj} — filtering failed ({e})"
    aligned_ppg = normalize(aligned_ppg).astype(np.float32)

    try:
        pos_sig   = bandpass(pos_method(rgb_signal),   fps)
        pos_sig   = normalize(pos_sig).astype(np.float32)
    except Exception:
        pos_sig   = np.zeros(len(rgb_signal), dtype=np.float32)

    try:
        chrom_sig = bandpass(chrom_method(rgb_signal), fps)
        chrom_sig = normalize(chrom_sig).astype(np.float32)
    except Exception:
        chrom_sig = np.zeros(len(rgb_signal), dtype=np.float32)

    rgb_signal = normalize_rgb(rgb_signal)

    np.save(os.path.join(SAVE_DIR, f"{subj}_rgb.npy"),   rgb_signal)
    np.save(os.path.join(SAVE_DIR, f"{subj}_ppg.npy"),   aligned_ppg)
    np.save(os.path.join(SAVE_DIR, f"{subj}_fps.npy"),   np.array([fps], dtype=np.float32))
    np.save(os.path.join(SAVE_DIR, f"{subj}_pos.npy"),   pos_sig)
    np.save(os.path.join(SAVE_DIR, f"{subj}_chrom.npy"), chrom_sig)

    return (
        f"[OK] {subj} — {idx} frames, {bad_frames} bad "
        f"({100 * bad_frames / max(idx, 1):.1f}%)"
    )


def preprocess():
    if not os.path.isdir(DATA_ROOT):
        print(f"Dataset path not found: {DATA_ROOT}")
        return

    subjects = sorted(
        d for d in os.listdir(DATA_ROOT)
        if os.path.isdir(os.path.join(DATA_ROOT, d))
    )
    print(f"Found {len(subjects)} subjects: {subjects}\n")

    if not subjects:
        print("No subject folders found. Nothing to preprocess.")
        return

    max_workers = min(4, len(subjects))

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(process_subject, s): s for s in subjects}
        for fut in as_completed(futures):
            subj = futures[fut]
            try:
                print(fut.result())
            except Exception as e:
                print(f"[ERR] {subj} — unexpected failure ({e})")

    print("\n✅ Preprocessing complete")


if __name__ == "__main__":
    preprocess()