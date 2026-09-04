"""
HeartLens v8 — Multi-User Web App (session-scoped, browser-camera based)
==============================================================================
Educational rPPG demo. Not a medical device.

ARCHITECTURE CHANGE from v7: previously the server read ONE physical
webcam (cv2.VideoCapture(0)) and everyone viewed that single feed. That
does not scale to many independent users.

v8 instead:
  - Each user's BROWSER captures ITS OWN webcam (JavaScript getUserMedia)
  - The browser periodically POSTs a small JPEG frame to /process_frame
    along with a session_id (generated once per browser tab)
  - The server keeps a separate SessionState per session_id, runs the
    same AMCF + respiration + stress pipeline on each session
    independently, and returns live stats in the response
  - Many people, each on their own device, can use the app AT THE SAME
    TIME, each seeing only their own vitals

This also still supports multiple FACES within one session's frame
(e.g. a teacher points one laptop camera at a group) via the existing
"secondary people" lightweight tracking, in addition to many separate
sessions.
"""

import cv2
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from flask import Flask, render_template, request, jsonify, send_from_directory
import threading
import time
import os
import json
import base64
import uuid
from collections import deque, Counter

app = Flask(__name__)

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
BUFFER_SECONDS = 10
FPS_ASSUMED = 10                 # client posts frames at ~10/sec
BUFFER_SIZE = BUFFER_SECONDS * FPS_ASSUMED * 2
MIN_BPM, MAX_BPM = 42, 220
CALIBRATION_SECONDS = 10
BPM_HISTORY_LEN = 12
RESP_HISTORY_LEN = 8
MAX_FACE_JUMP_RATIO = 0.35
SMOOTHING_ALPHA = 0.15
RESP_SMOOTHING_ALPHA = 0.2
MIN_BREATHS, MAX_BREATHS = 6, 30
LOG_INTERVAL = 2.0
MIN_CONFIDENCE = 1.3
ROI_NAMES = ["forehead", "cheek_l", "cheek_r"]

MAX_SECONDARY_PEOPLE = 3
SECONDARY_BUFFER_SIZE = FPS_ASSUMED * 8
TRACK_TIMEOUT = 3.0
SECONDARY_RECOMPUTE_INTERVAL = 1.0
SMILE_CHECK_INTERVAL = 0.5
SESSION_TIMEOUT = 300  # drop sessions inactive for 5 minutes

PROFILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profiles")
os.makedirs(PROFILES_DIR, exist_ok=True)
RESEARCH_CSV = os.path.join(PROFILES_DIR, "..", "research_data.csv")
RESEARCH_CSV = os.path.abspath(RESEARCH_CSV)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")


# ---------------------------------------------------------------------
# Per-session state + session registry
# ---------------------------------------------------------------------
class SessionState:
    def __init__(self):
        self.bpm = 0.0
        self.physio_stress = "Calibrating..."
        self.stress = "Calibrating..."
        self.smile_detected = False
        self.resp_rate = 0.0
        self.roi_confidence = {name: 0.0 for name in ROI_NAMES}

        self.roi_buffers = {
            name: {"r": deque(maxlen=BUFFER_SIZE), "g": deque(maxlen=BUFFER_SIZE), "b": deque(maxlen=BUFFER_SIZE)}
            for name in ROI_NAMES
        }
        self.chest_buffer = deque(maxlen=BUFFER_SIZE)
        self.bpm_history = deque(maxlen=BPM_HISTORY_LEN)
        self.resp_history = deque(maxlen=RESP_HISTORY_LEN)
        self.smoothed_bpm = None
        self.smoothed_resp = None

        self.last_spectrum = {"freqs": [], "power": []}
        self.session_log = deque(maxlen=1000)

        self.frame_times = deque(maxlen=30)
        self.start_time = time.time()
        self.last_face_center = None
        self.prev_chest_gray = None
        self.last_smile_check = 0.0
        self.last_log_time = 0.0

        self.secondary_people = {}
        self.next_person_id = 1

        self.user_name = None
        self.baseline_bpm = None
        self.baseline_sessions = 0

        self.validation_log = deque(maxlen=200)

        self.amcf_enabled = True
        self.skin_tone = None

        self.last_active = time.time()
        self.last_face_boxes = {}


sessions = {}
sessions_lock = threading.RLock()


def get_session(session_id):
    with sessions_lock:
        if session_id not in sessions:
            sessions[session_id] = SessionState()
        sessions[session_id].last_active = time.time()
        return sessions[session_id]


def cleanup_sessions():
    while True:
        time.sleep(60)
        now = time.time()
        with sessions_lock:
            stale = [sid for sid, s in sessions.items() if now - s.last_active > SESSION_TIMEOUT]
            for sid in stale:
                del sessions[sid]


threading.Thread(target=cleanup_sessions, daemon=True).start()


# ---------------------------------------------------------------------
# Signal processing helpers (unchanged core algorithms)
# ---------------------------------------------------------------------
def bandpass_filter(signal, fs, low, high, order=3):
    nyq = 0.5 * fs
    if nyq <= high or len(signal) < order * 3 + 1:
        return signal
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    try:
        return filtfilt(b, a, signal)
    except ValueError:
        return signal


def chrom_signal(r, g, b):
    r = np.array(r, dtype=np.float64)
    g = np.array(g, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    r_n = r / (np.mean(r) + 1e-8)
    g_n = g / (np.mean(g) + 1e-8)
    b_n = b / (np.mean(b) + 1e-8)
    x = 3 * r_n - 2 * g_n
    y = 1.5 * r_n + g_n - 1.5 * b_n
    std_x = np.std(x) + 1e-8
    std_y = np.std(y) + 1e-8
    alpha = std_x / std_y
    s = x - alpha * y
    return s - np.mean(s)


def fft_spectrum_with_confidence(signal, fs, min_bpm, max_bpm):
    n = len(signal)
    if n < fs * 4:
        return None, 0.0, [], []
    windowed = signal * np.hanning(n)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    power = np.abs(np.fft.rfft(windowed))
    valid = (freqs >= min_bpm / 60) & (freqs <= max_bpm / 60)
    if not np.any(valid):
        return None, 0.0, [], []
    freqs_valid = freqs[valid] * 60.0
    power_valid = power[valid]
    peak_idx = np.argmax(power_valid)
    peak_bpm = float(freqs_valid[peak_idx])
    mean_power = np.mean(power_valid) + 1e-8
    confidence = float(power_valid[peak_idx] / mean_power)
    max_p = np.max(power_valid) + 1e-8
    power_norm = (power_valid / max_p).tolist()
    return peak_bpm, confidence, freqs_valid.tolist(), power_norm


def estimate_stress(signal, fs):
    min_distance = max(1, int(fs * 0.4))
    peaks, _ = find_peaks(signal, distance=min_distance)
    if len(peaks) < 5:
        return "Calibrating..."
    intervals_ms = np.diff(peaks) / fs * 1000.0
    diffs = np.diff(intervals_ms)
    if len(diffs) < 2:
        return "Calibrating..."
    rmssd = np.sqrt(np.mean(diffs ** 2))
    if rmssd > 60:
        return "Calm"
    elif rmssd > 30:
        return "Normal"
    else:
        return "Elevated"


def get_roi(frame, face_rect, name):
    x, y, w, h = face_rect
    if name == "forehead":
        rx, ry, rw, rh = x + int(w * 0.25), y + int(h * 0.05), int(w * 0.5), int(h * 0.20)
    elif name == "cheek_l":
        rx, ry, rw, rh = x + int(w * 0.12), y + int(h * 0.55), int(w * 0.25), int(h * 0.20)
    else:
        rx, ry, rw, rh = x + int(w * 0.63), y + int(h * 0.55), int(w * 0.25), int(h * 0.20)
    return frame[ry:ry + rh, rx:rx + rw], (rx, ry, rw, rh)


def get_chest_roi(frame, face_rect):
    x, y, w, h = face_rect
    frame_h, frame_w = frame.shape[:2]
    cy = min(y + h + int(h * 0.15), frame_h - 1)
    ch = min(int(h * 0.9), frame_h - cy)
    cx = max(0, x - int(w * 0.1))
    cw = min(int(w * 1.2), frame_w - cx)
    if ch <= 0 or cw <= 0:
        return None, (cx, cy, 0, 0)
    return frame[cy:cy + ch, cx:cx + cw], (cx, cy, cw, ch)


def load_profile(name):
    path = os.path.join(PROFILES_DIR, f"{name}.json")
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {"sessions": []}


def save_profile(name, profile):
    path = os.path.join(PROFILES_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(profile, f, indent=2)


# ---------------------------------------------------------------------
# Core per-frame processing (runs once per POSTed frame, per session)
# ---------------------------------------------------------------------
def process_one_frame(s: SessionState, frame):
    now = time.time()
    s.frame_times.append(now)

    if len(s.frame_times) >= 2:
        span = s.frame_times[-1] - s.frame_times[0]
        fs_estimate = (len(s.frame_times) - 1) / span if span > 0 else FPS_ASSUMED
    else:
        fs_estimate = FPS_ASSUMED
    fs_estimate = max(3.0, min(fs_estimate, 30.0))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = list(face_cascade.detectMultiScale(gray, 1.3, 5))

    result = {
        "face_found": False, "bpm": s.bpm, "resp_rate": s.resp_rate,
        "stress": s.stress, "physio_stress": s.physio_stress,
        "smile_detected": s.smile_detected, "roi_confidence": s.roi_confidence,
        "roi_boxes": {}, "face_box": None, "chest_box": None,
        "secondary_people": [], "calibrating": True,
    }

    if len(faces) == 0:
        return result

    faces.sort(key=lambda f: f[2] * f[3], reverse=True)
    face_rect = faces[0]
    other_faces = faces[1:1 + MAX_SECONDARY_PEOPLE]
    x, y, w, h = face_rect
    center = (x + w / 2, y + h / 2)
    result["face_found"] = True
    result["face_box"] = [int(x), int(y), int(w), int(h)]

    motion_ok = True
    if s.last_face_center is not None:
        dx = abs(center[0] - s.last_face_center[0])
        dy = abs(center[1] - s.last_face_center[1])
        if dx > w * MAX_FACE_JUMP_RATIO or dy > h * MAX_FACE_JUMP_RATIO:
            motion_ok = False
    s.last_face_center = center

    roi_boxes = {}
    for name in ROI_NAMES:
        roi_img, box = get_roi(frame, face_rect, name)
        roi_boxes[name] = box
        if roi_img.size > 0 and motion_ok:
            s.roi_buffers[name]["b"].append(float(np.mean(roi_img[:, :, 0])))
            s.roi_buffers[name]["g"].append(float(np.mean(roi_img[:, :, 1])))
            s.roi_buffers[name]["r"].append(float(np.mean(roi_img[:, :, 2])))
    result["roi_boxes"] = {k: [int(v) for v in box] for k, box in roi_boxes.items()}

    chest_roi, (cx, cy, cw, ch) = get_chest_roi(frame, face_rect)
    result["chest_box"] = [int(cx), int(cy), int(cw), int(ch)] if ch > 0 else None
    if chest_roi is not None and chest_roi.size > 0:
        chest_gray = cv2.cvtColor(chest_roi, cv2.COLOR_BGR2GRAY)
        chest_gray = cv2.resize(chest_gray, (64, 64))
        if s.prev_chest_gray is not None and motion_ok:
            flow = cv2.calcOpticalFlowFarneback(s.prev_chest_gray, chest_gray, None, 0.5, 2, 15, 3, 5, 1.2, 0)
            s.chest_buffer.append(float(np.mean(flow[..., 1])))
        s.prev_chest_gray = chest_gray

    if now - s.last_smile_check > SMILE_CHECK_INTERVAL:
        s.last_smile_check = now
        mouth_y = y + int(h * 0.6)
        mouth_region = gray[mouth_y:y + h, x:x + w]
        smiles = smile_cascade.detectMultiScale(mouth_region, 1.7, 20) if mouth_region.size > 0 else []
        s.smile_detected = len(smiles) > 0

    elapsed = now - s.start_time
    buf_len = len(s.roi_buffers["forehead"]["g"])
    have_enough = buf_len > fs_estimate * 4 and elapsed > CALIBRATION_SECONDS
    result["calibrating"] = not have_enough

    if have_enough:
        candidates = {}
        forehead_filtered = None
        fusion_rois = ROI_NAMES if s.amcf_enabled else ["forehead"]
        for name in ROI_NAMES:
            buf = s.roi_buffers[name]
            sig = chrom_signal(buf["r"], buf["g"], buf["b"])
            filtered = bandpass_filter(sig, fs_estimate, 0.7, 4.0)
            bpm, conf, freqs_bpm, power_norm = fft_spectrum_with_confidence(filtered, fs_estimate, MIN_BPM, MAX_BPM)
            s.roi_confidence[name] = round(min(conf, 10.0) / 10.0 * 100, 0)
            if name in fusion_rois and bpm is not None and conf >= MIN_CONFIDENCE:
                candidates[name] = (bpm, conf)
            if name == "forehead":
                forehead_filtered = filtered
                s.last_spectrum = {"freqs": freqs_bpm, "power": power_norm}

        if candidates:
            total_conf = sum(c for _, c in candidates.values())
            fused_bpm = sum(b * c for b, c in candidates.values()) / total_conf
            s.bpm_history.append(fused_bpm)
            median_bpm = float(np.median(s.bpm_history))
            s.smoothed_bpm = (median_bpm if s.smoothed_bpm is None else
                               SMOOTHING_ALPHA * median_bpm + (1 - SMOOTHING_ALPHA) * s.smoothed_bpm)
            s.bpm = round(s.smoothed_bpm, 1)

        if forehead_filtered is not None:
            s.physio_stress = estimate_stress(forehead_filtered, fs_estimate)

        levels = {"Calm": 0, "Normal": 1, "Elevated": 2}
        if s.physio_stress in levels:
            lvl = levels[s.physio_stress]
            if s.smile_detected and lvl > 0:
                lvl -= 1
            s.stress = {0: "Calm", 1: "Normal", 2: "Elevated"}[lvl]
        else:
            s.stress = s.physio_stress

        if len(s.chest_buffer) > fs_estimate * 6:
            resp_sig = np.array(s.chest_buffer) - np.mean(s.chest_buffer)
            resp_filtered = bandpass_filter(resp_sig, fs_estimate, MIN_BREATHS / 60, MAX_BREATHS / 60)
            resp_bpm, _, _, _ = fft_spectrum_with_confidence(resp_filtered, fs_estimate, MIN_BREATHS, MAX_BREATHS)
            if resp_bpm:
                s.resp_history.append(resp_bpm)
                median_resp = float(np.median(s.resp_history))
                s.smoothed_resp = (median_resp if s.smoothed_resp is None else
                                    RESP_SMOOTHING_ALPHA * median_resp + (1 - RESP_SMOOTHING_ALPHA) * s.smoothed_resp)
                s.resp_rate = round(s.smoothed_resp, 1)

        if now - s.last_log_time > LOG_INTERVAL:
            s.session_log.append({"t": round(elapsed, 1), "bpm": s.bpm, "resp": s.resp_rate, "stress": s.stress})
            s.last_log_time = now
    else:
        remaining = max(0, int(CALIBRATION_SECONDS - elapsed))
        s.physio_stress = f"Calibrating... {remaining}s"
        s.stress = s.physio_stress

    # secondary people (extra faces in this session's own frame)
    stale = [pid for pid, info in s.secondary_people.items() if now - info["last_seen"] > TRACK_TIMEOUT]
    for pid in stale:
        del s.secondary_people[pid]

    for of in other_faces:
        ox, oy, ow, oh = of
        ocenter = (ox + ow / 2, oy + oh / 2)
        match_id, best_dist = None, None
        for pid, info in s.secondary_people.items():
            d = ((ocenter[0] - info["centroid"][0]) ** 2 + (ocenter[1] - info["centroid"][1]) ** 2) ** 0.5
            if best_dist is None or d < best_dist:
                best_dist, match_id = d, pid
        if match_id is None or best_dist > ow * 0.8:
            if len(s.secondary_people) >= MAX_SECONDARY_PEOPLE:
                continue
            match_id = s.next_person_id
            s.next_person_id += 1
            s.secondary_people[match_id] = {
                "centroid": ocenter, "last_seen": now,
                "r": deque(maxlen=SECONDARY_BUFFER_SIZE), "g": deque(maxlen=SECONDARY_BUFFER_SIZE), "b": deque(maxlen=SECONDARY_BUFFER_SIZE),
                "bpm": 0.0, "last_compute": 0.0, "bbox": (ox, oy, ow, oh),
            }
        info = s.secondary_people[match_id]
        info["centroid"] = ocenter
        info["last_seen"] = now
        info["bbox"] = (ox, oy, ow, oh)
        roi_img, _ = get_roi(frame, of, "forehead")
        if roi_img.size > 0:
            info["r"].append(float(np.mean(roi_img[:, :, 2])))
            info["g"].append(float(np.mean(roi_img[:, :, 1])))
            info["b"].append(float(np.mean(roi_img[:, :, 0])))
        if now - info["last_compute"] > SECONDARY_RECOMPUTE_INTERVAL and len(info["g"]) > fs_estimate * 4:
            sig = chrom_signal(info["r"], info["g"], info["b"])
            filtered = bandpass_filter(sig, fs_estimate, 0.7, 4.0)
            bpm, conf, _, _ = fft_spectrum_with_confidence(filtered, fs_estimate, MIN_BPM, MAX_BPM)
            if bpm and conf >= MIN_CONFIDENCE:
                info["bpm"] = round(bpm, 1)
            info["last_compute"] = now

    result["secondary_people"] = [{"id": pid, "bpm": info["bpm"], "bbox": [int(v) for v in info["bbox"]]} for pid, info in s.secondary_people.items()]
    result["bpm"] = s.bpm
    result["resp_rate"] = s.resp_rate
    result["stress"] = s.stress
    result["physio_stress"] = s.physio_stress
    result["smile_detected"] = s.smile_detected
    result["roi_confidence"] = s.roi_confidence
    return result


# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/sw.js")
def service_worker():
    # Served at root (not /static/sw.js) so its default scope covers the
    # whole app ('/'), not just /static/ — required for Chrome to treat
    # this as an installable PWA.
    resp = send_from_directory("static", "sw.js", mimetype="application/javascript")
    resp.headers["Service-Worker-Allowed"] = "/"
    return resp


@app.route("/process_frame", methods=["POST"])
def process_frame():
    data = request.get_json(force=True, silent=True) or {}
    session_id = data.get("session_id")
    image_b64 = data.get("image")
    if not session_id or not image_b64:
        return jsonify({"error": "missing session_id or image"}), 400

    try:
        if "," in image_b64:
            image_b64 = image_b64.split(",", 1)[1]
        img_bytes = base64.b64decode(image_b64)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "bad image"}), 400
    except Exception:
        return jsonify({"error": "decode failed"}), 400

    s = get_session(session_id)
    result = process_one_frame(s, frame)

    delta = None
    if s.baseline_bpm is not None and s.bpm:
        delta = round(s.bpm - s.baseline_bpm, 1)
    result["baseline_bpm"] = s.baseline_bpm
    result["baseline_delta"] = delta
    result["baseline_sessions"] = s.baseline_sessions
    result["active_sessions"] = len(sessions)
    return jsonify(result)


@app.route("/spectrum")
def spectrum():
    session_id = request.args.get("session_id")
    s = get_session(session_id) if session_id else None
    if not s:
        return jsonify({"freqs": [], "power": []})
    return jsonify(s.last_spectrum)


@app.route("/set_user", methods=["POST"])
def set_user():
    data = request.get_json(force=True, silent=True) or {}
    session_id = data.get("session_id")
    name = str(data.get("name", "")).strip()[:40]
    if not session_id or not name:
        return jsonify({"ok": False, "error": "missing session_id or name"}), 400
    profile = load_profile(name)
    sessions_list = profile.get("sessions", [])
    baseline = round(sum(s["avg_bpm"] for s in sessions_list) / len(sessions_list), 1) if sessions_list else None
    s = get_session(session_id)
    s.user_name = name
    s.baseline_bpm = baseline
    s.baseline_sessions = len(sessions_list)
    return jsonify({"ok": True, "baseline_bpm": baseline, "sessions": len(sessions_list)})


@app.route("/save_session", methods=["POST"])
def save_session():
    data = request.get_json(force=True, silent=True) or {}
    session_id = data.get("session_id")
    s = get_session(session_id) if session_id else None
    if not s or not s.user_name:
        return jsonify({"ok": False, "error": "no user set"}), 400
    bpms = [e["bpm"] for e in s.session_log if e["bpm"]]
    if not bpms:
        return jsonify({"ok": False, "error": "not enough data"}), 400
    avg_bpm = round(sum(bpms) / len(bpms), 1)
    profile = load_profile(s.user_name)
    profile.setdefault("sessions", []).append({"avg_bpm": avg_bpm, "date": time.strftime("%Y-%m-%d %H:%M")})
    save_profile(s.user_name, profile)
    new_baseline = round(sum(x["avg_bpm"] for x in profile["sessions"]) / len(profile["sessions"]), 1)
    s.baseline_bpm = new_baseline
    s.baseline_sessions = len(profile["sessions"])
    return jsonify({"ok": True, "baseline_bpm": new_baseline, "sessions": len(profile["sessions"])})


@app.route("/validate", methods=["POST"])
def validate():
    data = request.get_json(force=True, silent=True) or {}
    session_id = data.get("session_id")
    s = get_session(session_id) if session_id else None
    if not s:
        return jsonify({"ok": False, "error": "no session"}), 400
    try:
        ref = float(data.get("reference_bpm"))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "invalid reference_bpm"}), 400
    if s.bpm:
        elapsed = round(time.time() - s.start_time, 1)
        s.validation_log.append({"t": elapsed, "ref": ref, "est": s.bpm, "error": round(abs(ref - s.bpm), 1)})
    return jsonify({"ok": True})


@app.route("/validation_stats")
def validation_stats():
    session_id = request.args.get("session_id")
    s = get_session(session_id) if session_id else None
    if not s or not s.validation_log:
        return jsonify({"n": 0, "mae": None, "correlation": None})
    log = list(s.validation_log)
    errors = [e["error"] for e in log]
    mae = round(sum(errors) / len(errors), 2)
    correlation = None
    if len(log) >= 3:
        refs = np.array([e["ref"] for e in log])
        ests = np.array([e["est"] for e in log])
        if np.std(refs) > 0 and np.std(ests) > 0:
            correlation = round(float(np.corrcoef(refs, ests)[0, 1]), 3)
    return jsonify({"n": len(log), "mae": mae, "correlation": correlation})


@app.route("/set_mode", methods=["POST"])
def set_mode():
    data = request.get_json(force=True, silent=True) or {}
    session_id = data.get("session_id")
    s = get_session(session_id) if session_id else None
    if not s:
        return jsonify({"ok": False}), 400
    s.amcf_enabled = bool(data.get("amcf_enabled", True))
    return jsonify({"ok": True, "amcf_enabled": s.amcf_enabled})


@app.route("/set_skin_tone", methods=["POST"])
def set_skin_tone():
    data = request.get_json(force=True, silent=True) or {}
    session_id = data.get("session_id")
    s = get_session(session_id) if session_id else None
    if not s:
        return jsonify({"ok": False}), 400
    s.skin_tone = str(data.get("skin_tone", "")).strip()[:40] or None
    return jsonify({"ok": True, "skin_tone": s.skin_tone})


@app.route("/reset_run", methods=["POST"])
def reset_run():
    data = request.get_json(force=True, silent=True) or {}
    session_id = data.get("session_id")
    s = get_session(session_id) if session_id else None
    if not s:
        return jsonify({"ok": False}), 400
    for name in ROI_NAMES:
        s.roi_buffers[name]["r"].clear()
        s.roi_buffers[name]["g"].clear()
        s.roi_buffers[name]["b"].clear()
    s.chest_buffer.clear()
    s.bpm_history.clear()
    s.resp_history.clear()
    s.smoothed_bpm = None
    s.smoothed_resp = None
    s.bpm = 0.0
    s.resp_rate = 0.0
    s.validation_log.clear()
    s.start_time = time.time()
    s.physio_stress = "Calibrating..."
    s.stress = "Calibrating..."
    return jsonify({"ok": True})


@app.route("/save_research_entry", methods=["POST"])
def save_research_entry():
    import csv
    data = request.get_json(force=True, silent=True) or {}
    session_id = data.get("session_id")
    s = get_session(session_id) if session_id else None
    if not s:
        return jsonify({"ok": False, "error": "no session"}), 400
    if not s.skin_tone:
        return jsonify({"ok": False, "error": "Set skin tone before saving a research entry"}), 400
    log = list(s.validation_log)
    if not log:
        return jsonify({"ok": False, "error": "Log at least 2-3 reference readings first"}), 400
    errors = [e["error"] for e in log]
    mae = round(sum(errors) / len(errors), 2)
    is_new = not os.path.exists(RESEARCH_CSV)
    with open(RESEARCH_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["timestamp", "name", "skin_tone", "amcf_enabled", "mae", "n_readings", "avg_bpm"])
        writer.writerow([time.strftime("%Y-%m-%d %H:%M"), s.user_name or "anonymous", s.skin_tone, s.amcf_enabled, mae, len(log), s.bpm])
    return jsonify({"ok": True, "mae": mae, "n": len(log)})


@app.route("/research_summary")
def research_summary():
    import csv
    if not os.path.exists(RESEARCH_CSV):
        return jsonify({"rows": [], "groups": []})
    rows = []
    with open(RESEARCH_CSV, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    groups = {}
    for r in rows:
        key = (r["skin_tone"], r["amcf_enabled"])
        groups.setdefault(key, []).append(float(r["mae"]))
    summary = [{"skin_tone": t, "amcf_enabled": a, "avg_mae": round(sum(m) / len(m), 2), "n_entries": len(m)}
               for (t, a), m in sorted(groups.items())]
    return jsonify({"rows": rows, "groups": summary})


@app.route("/research")
def research_page():
    return render_template("research.html")


@app.route("/report")
def report():
    session_id = request.args.get("session_id")
    s = get_session(session_id) if session_id else None
    if not s:
        return "No session found. Open the dashboard first.", 404

    log = list(s.session_log)
    session_seconds = int(time.time() - s.start_time)
    bpms = [e["bpm"] for e in log if e["bpm"]]
    resps = [e["resp"] for e in log if e["resp"]]
    avg_bpm = round(sum(bpms) / len(bpms), 1) if bpms else 0
    avg_resp = round(sum(resps) / len(resps), 1) if resps else 0
    max_bpm = round(max(bpms), 1) if bpms else 0
    min_bpm = round(min(bpms), 1) if bpms else 0
    stress_counts = dict(Counter(e["stress"] for e in log if "Calibrating" not in e["stress"]))
    validation_log = list(s.validation_log)
    val_mae = round(sum(e["error"] for e in validation_log) / len(validation_log), 2) if validation_log else None

    return render_template(
        "report.html", log=log, avg_bpm=avg_bpm, avg_resp=avg_resp, max_bpm=max_bpm,
        min_bpm=min_bpm, stress_counts=stress_counts, session_seconds=session_seconds,
        user_name=s.user_name, baseline_bpm=s.baseline_bpm, val_mae=val_mae, validation_n=len(validation_log),
    )


@app.route("/active_sessions")
def active_sessions():
    with sessions_lock:
        return jsonify({"count": len(sessions)})


if __name__ == "__main__":
    def open_browser():
        time.sleep(1.5)
        import webbrowser
        webbrowser.open("http://127.0.0.1:5000")

    threading.Thread(target=open_browser, daemon=True).start()
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)
