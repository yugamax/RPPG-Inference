import os
import tempfile
from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from scipy.signal import find_peaks

from processing import get_roi_mean, make_face_mesh, normalize, normalize_rgb


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model.pth.zip")
DEFAULT_SEQ_LEN = 256


def _get_cors_origins() -> List[str]:
    cors_origins = os.getenv("CORS_ALLOW_ORIGINS", "*").strip()
    if not cors_origins:
        return ["*"]
    return [origin.strip() for origin in cors_origins.split(",") if origin.strip()]


class DilatedBlock(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, dilation: int):
        super().__init__()
        self.conv = nn.Conv1d(ch_in, ch_out, 3, padding=dilation, dilation=dilation)
        self.bn = nn.BatchNorm1d(ch_out)
        self.act = nn.GELU()
        self.skip = nn.Conv1d(ch_in, ch_out, 1) if ch_in != ch_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x))) + self.skip(x)


class TSCAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            DilatedBlock(3, 64, 1),
            DilatedBlock(64, 64, 2),
            DilatedBlock(64, 96, 4),
            DilatedBlock(96, 96, 8),
            DilatedBlock(96, 64, 16),
        )
        self.attn = nn.MultiheadAttention(64, 4, batch_first=True)
        self.norm = nn.LayerNorm(64)
        self.head = nn.Conv1d(64, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        f = feat.permute(0, 2, 1)
        attn_out, _ = self.attn(f, f, f)
        f = self.norm(f + attn_out)
        f = f.permute(0, 2, 1)
        return self.head(f).squeeze(1)


app = FastAPI(title="rPPG Inference API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_get_cors_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_model(model_path: str, device: torch.device) -> TSCAN:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = TSCAN().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def extract_rgb_signal(video_path: str) -> tuple[np.ndarray, float, int, int]:
    cap = cv2.VideoCapture(video_path)
    mesh = make_face_mesh()

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        cap.release()
        raise ValueError("Video FPS is invalid. You can pass fps_override in the request.")

    rgb_signal = []
    bad_frames = 0
    frames = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mean_rgb = get_roi_mean(frame_rgb, mesh)

            if mean_rgb is None:
                bad_frames += 1
                mean_rgb = rgb_signal[-1] if rgb_signal else np.zeros(3, dtype=np.float32)

            rgb_signal.append(mean_rgb)
            frames += 1
    finally:
        cap.release()
        mesh.close()

    if len(rgb_signal) < 32:
        raise ValueError(f"Video too short for inference: {len(rgb_signal)} frames")

    rgb_signal_np = np.array(rgb_signal, dtype=np.float32)
    return rgb_signal_np, fps, frames, bad_frames


def predict_signal(
    model: TSCAN,
    rgb_signal: np.ndarray,
    device: torch.device,
    seq_len: int = DEFAULT_SEQ_LEN,
    stride: Optional[int] = None,
) -> np.ndarray:
    stride = stride or (seq_len // 2)
    T = rgb_signal.shape[0]

    rgb_norm = normalize_rgb(rgb_signal)
    pred = np.zeros(T, dtype=np.float32)
    count = np.zeros(T, dtype=np.float32)

    starts: List[int] = []
    if T <= seq_len:
        starts = [0]
    else:
        starts = list(range(0, T - seq_len + 1, stride))
        if starts[-1] != T - seq_len:
            starts.append(T - seq_len)

    with torch.no_grad():
        for s in starts:
            e = min(s + seq_len, T)
            clip = rgb_norm[s:e]

            if len(clip) < seq_len:
                pad_len = seq_len - len(clip)
                pad = np.repeat(clip[-1][None, :], pad_len, axis=0)
                clip = np.concatenate([clip, pad], axis=0)

            x = torch.from_numpy(clip.T.astype(np.float32)).unsqueeze(0).to(device)
            y = model(x).squeeze(0).cpu().numpy()[: e - s]
            pred[s:e] += y
            count[s:e] += 1.0

    count[count == 0] = 1.0
    pred = pred / count
    pred = normalize(pred).astype(np.float32)
    return pred


def estimate_bpm(signal: np.ndarray, fps: float) -> float:
    if len(signal) < 32 or fps <= 0:
        return float("nan")

    sig = signal.astype(np.float64)
    sig = sig - np.mean(sig)

    spectrum = np.abs(np.fft.rfft(sig))
    freqs = np.fft.rfftfreq(len(sig), d=1.0 / fps)

    band = (freqs >= 0.7) & (freqs <= 4.0)
    if not np.any(band):
        return float("nan")

    idx = np.argmax(spectrum[band])
    peak_hz = freqs[band][idx]
    return float(peak_hz * 60.0)


def estimate_heart_rate_trend(signal: np.ndarray, fps: float) -> float:
    window = max(int(8 * fps), 32)
    hop = max(int(2 * fps), 8)
    if len(signal) < window:
        return 0.0

    hr_values: List[float] = []
    for start in range(0, len(signal) - window + 1, hop):
        seg = signal[start : start + window]
        bpm = estimate_bpm(seg, fps)
        if np.isfinite(bpm):
            hr_values.append(float(bpm))

    if len(hr_values) < 2:
        return 0.0

    x = np.arange(len(hr_values), dtype=np.float64)
    y = np.asarray(hr_values, dtype=np.float64)
    slope = np.polyfit(x, y, 1)[0]
    return float(slope)


def estimate_rmssd_ms(signal: np.ndarray, fps: float) -> float:
    if len(signal) < 32 or fps <= 0:
        return float("nan")

    distance = max(int(0.3 * fps), 1)
    prominence = max(0.1 * float(np.std(signal)), 1e-6)
    peaks, _ = find_peaks(signal, distance=distance, prominence=prominence)
    if len(peaks) < 3:
        return float("nan")

    rr_ms = np.diff(peaks).astype(np.float64) / fps * 1000.0
    if len(rr_ms) < 2:
        return float("nan")

    diff_rr = np.diff(rr_ms)
    rmssd = np.sqrt(np.mean(diff_rr ** 2))
    return float(rmssd)


def estimate_respiratory_rate(signal: np.ndarray, fps: float) -> float:
    if len(signal) < 64 or fps <= 0:
        return float("nan")

    sig = signal.astype(np.float64)
    sig = sig - np.mean(sig)
    spectrum = np.abs(np.fft.rfft(sig))
    freqs = np.fft.rfftfreq(len(sig), d=1.0 / fps)

    band = (freqs >= 0.1) & (freqs <= 0.5)
    if not np.any(band):
        return float("nan")

    idx = np.argmax(spectrum[band])
    peak_hz = freqs[band][idx]
    return float(peak_hz * 60.0)


def estimate_confidence_score(
    signal_quality: float,
    heart_rate: float,
    heart_rate_trend: float,
    rmssd: float,
    respiratory_rate: float,
) -> float:
    score = float(np.clip(signal_quality, 0.0, 1.0))

    if heart_rate <= 0:
        score *= 0.65
    if rmssd <= 0:
        score *= 0.8
    if respiratory_rate <= 0:
        score *= 0.85

    # Very unstable trends usually mean motion or poor estimate stability.
    trend_penalty = min(abs(heart_rate_trend) / 20.0, 0.35)
    score *= 1.0 - trend_penalty

    # Extremely implausible values reduce confidence further.
    if heart_rate < 40 or heart_rate > 180:
        score *= 0.8
    if rmssd > 200:
        score *= 0.85
    if respiratory_rate > 40:
        score *= 0.9

    return float(np.clip(score, 0.0, 1.0))


def _interp_heart_rate(hr: float) -> str:
    if hr < 60:
        return "A bit low"
    if hr <= 100:
        return "In a typical range"
    return "A bit high"


def _interp_heart_trend(trend: float) -> str:
    if trend > 0.1:
        return "Going up, which can happen with activity or stress"
    if trend < -0.1:
        return "Going down, which often means resting or relaxing"
    return "Fairly steady"


def _interp_rmssd(rmssd: float) -> str:
    if rmssd > 50:
        return "Good recovery"
    if rmssd >= 20:
        return "Reasonable recovery"
    return "Recovery looks low"


def _interp_respiratory_rate(rr: float) -> str:
    if rr < 12:
        return "Breathing is slower than usual"
    if rr <= 20:
        return "Breathing looks normal"
    return "Breathing is faster than usual"


def _interp_signal_quality(sq: float) -> str:
    if sq > 0.8:
        return "Very reliable"
    if sq > 0.5:
        return "Moderately reliable"
    return "Unreliable - please try again"


def _plain_takeaway(status: str, heart_rate: float, respiratory_rate: float, rmssd: float) -> str:
    notes = []
    if heart_rate < 60:
        notes.append("heart rate is on the low side")
    elif heart_rate > 100:
        notes.append("heart rate is higher than usual")

    if respiratory_rate > 20:
        notes.append("breathing is faster than usual")
    elif respiratory_rate < 12:
        notes.append("breathing is slower than usual")

    if rmssd > 50:
        notes.append("recovery looks good")
    elif rmssd < 20:
        notes.append("recovery looks low")

    if not notes:
        return f"Overall, this looks {status.lower()} and fairly steady."

    joined = "; ".join(notes)
    return f"Overall, this looks {status.lower()}, but {joined}."


def _status_and_flags(heart_rate: float, rmssd: float, respiratory_rate: float) -> tuple[str, bool, int, int]:
    stressed = 0
    calm = 0

    if heart_rate > 100:
        stressed += 1
    elif heart_rate < 60:
        calm += 1

    if rmssd < 20:
        stressed += 1
    elif rmssd > 50:
        calm += 1

    if respiratory_rate > 20:
        stressed += 1
    elif respiratory_rate < 12:
        calm += 1

    mixed = calm > 0 and stressed > 0

    if mixed:
        return "Mixed", mixed, calm, stressed
    if stressed >= 2:
        return "Stressed", mixed, calm, stressed
    if calm >= 2:
        return "Calm", mixed, calm, stressed
    return "Normal", mixed, calm, stressed


def _reliability_label(signal_quality: float, confidence_score: float) -> str:
    if signal_quality >= 0.8 and confidence_score >= 0.8:
        return "High"
    if confidence_score >= 0.55:
        return "Medium"
    return "Low"


def _note_for_signals(
    mixed: bool,
    signal_quality: float,
    confidence_score: float,
    heart_rate: float,
    respiratory_rate: float,
    rmssd: float,
) -> Optional[str]:
    if confidence_score < 0.5 or signal_quality < 0.5:
        return "Low confidence: try a steadier shot with even lighting."

    if mixed:
        return "Mixed signals: breathing and recovery are not pointing the same way."

    if respiratory_rate > 20 and heart_rate < 70 and rmssd > 50:
        return "Breathing looks fast while other signals look calm; consider a retest."

    return None


def format_health_summary(
    heart_rate: float,
    heart_rate_trend: float,
    rmssd: float,
    respiratory_rate: float,
    signal_quality: float,
    confidence_score: float,
) -> str:
    hr_text = _interp_heart_rate(heart_rate)
    trend_text = _interp_heart_trend(heart_rate_trend)
    rmssd_text = _interp_rmssd(rmssd)
    resp_text = _interp_respiratory_rate(respiratory_rate)
    sq_text = _interp_signal_quality(signal_quality)
    status, mixed, _, _ = _status_and_flags(heart_rate, rmssd, respiratory_rate)
    reliability = _reliability_label(signal_quality, confidence_score)
    note = _note_for_signals(mixed, signal_quality, confidence_score, heart_rate, respiratory_rate, rmssd)
    takeaway = _plain_takeaway(status, heart_rate, respiratory_rate, rmssd)

    trend_val = f"{heart_rate_trend:+.3f}"

    summary = (
        "Health Summary:\n"
        f"Status: {status}\n\n"
        f"Reliability: {reliability}\n\n"
        f"Takeaway: {takeaway}\n\n"
    )

    if note:
        summary += f"Note: {note}\n\n"

    summary += (
        "Measurements:\n\n"
        f"* Heart Rate: {heart_rate:.2f} BPM ({hr_text})\n"
        f"* Heart Trend: {trend_val} ({trend_text})\n"
        f"* Recovery (RMSSD): {rmssd:.2f} ms ({rmssd_text})\n"
        f"* Breathing Rate: {respiratory_rate:.2f} breaths/min ({resp_text})\n"
        f"* Signal Quality: {signal_quality:.2f} ({sq_text})\n\n"
        "---"
    )

    return summary


@app.on_event("startup")
def startup_event() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)

    model = load_model(model_path, device)
    app.state.device = device
    app.state.model = model
    app.state.model_path = model_path


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(getattr(app.state, "device", "not_loaded")),
        "model_loaded": hasattr(app.state, "model"),
        "model_path": getattr(app.state, "model_path", "not_set"),
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    include_signal: bool = Query(False),
    fps_override: Optional[float] = Query(None, gt=0),
) -> dict:
    suffix = os.path.splitext(file.filename or "upload.bin")[1] or ".avi"
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        rgb_signal, fps, frame_count, bad_frames = extract_rgb_signal(tmp_path)
        if fps_override is not None:
            fps = float(fps_override)

        pred_ppg = predict_signal(app.state.model, rgb_signal, app.state.device)
        heart_rate = estimate_bpm(pred_ppg, fps)
        heart_rate_trend = estimate_heart_rate_trend(pred_ppg, fps)
        rmssd = estimate_rmssd_ms(pred_ppg, fps)
        respiratory_rate = estimate_respiratory_rate(pred_ppg, fps)
        signal_quality = float(1.0 - (bad_frames / max(frame_count, 1)))

        if not np.isfinite(heart_rate):
            heart_rate = 0.0
        if not np.isfinite(heart_rate_trend):
            heart_rate_trend = 0.0
        if not np.isfinite(rmssd):
            rmssd = 0.0
        if not np.isfinite(respiratory_rate):
            respiratory_rate = 0.0

        confidence_score = estimate_confidence_score(
            signal_quality=signal_quality,
            heart_rate=heart_rate,
            heart_rate_trend=heart_rate_trend,
            rmssd=rmssd,
            respiratory_rate=respiratory_rate,
        )
        confidence_percent = float(np.clip(confidence_score * 100.0, 0.0, 100.0))

        health_summary = format_health_summary(
            heart_rate=heart_rate,
            heart_rate_trend=heart_rate_trend,
            rmssd=rmssd,
            respiratory_rate=respiratory_rate,
            signal_quality=signal_quality,
            confidence_score=confidence_score,
        )

        response = {
            "fps": fps,
            "frames": frame_count,
            "bad_frames": bad_frames,
            "face_detection_rate": signal_quality,
            "estimated_bpm": heart_rate,
            "heart_rate": heart_rate,
            "heart_rate_trend": heart_rate_trend,
            "rmssd": rmssd,
            "respiratory_rate": respiratory_rate,
            "signal_quality": signal_quality,
            "confidence_score": confidence_score,
            "confidence_percent": confidence_percent,
            "health_summary": health_summary,
            "signal_length": int(len(pred_ppg)),
        }

        if include_signal:
            response["predicted_ppg"] = pred_ppg.tolist()

        return response
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except PermissionError:
                pass


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run("inference_api:app", host=host, port=port, reload=False)