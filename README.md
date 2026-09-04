# HeartLens 💓
### Contactless Heart Rate & Stress Monitor — using only a webcam

## 1. Problem Statement

Continuous health monitoring today depends on wearable hardware —
smartwatches, fitness bands, pulse oximeters. But billions of people
don't own one, can't afford one, or simply aren't wearing it in the
moment they need it (a sudden anxiety spike before an exam, an elderly
person at home, a patient in a low-resource clinic).

**Nearly everyone, however, has access to a camera** — on a laptop or
phone. This project answers the question:

> *Can a plain RGB camera, with no special hardware, estimate your
> heart rate and stress level accurately enough to be useful?*

The answer is yes, using a well-established biomedical signal
processing technique called **remote Photoplethysmography (rPPG)**.

## 2. Why this matters (real-world relevance)

- **Telemedicine**: doctors could get a rough vitals check during a
  video consultation without any extra device.
- **Mental health / exam stress screening**: colleges could offer a
  quick, non-invasive stress check-in for students.
- **Accessibility**: works for anyone with a webcam — no cost barrier.
- **Elderly care**: passive monitoring through a home camera, no need
  to remember to wear a device.

## 3. How it works (pipeline)

**This project's own contribution: AMCF (Adaptive Multi-ROI Confidence Fusion).**
Most webcam heart-rate demos track a single region (usually the forehead) —
fragile if hair shadows it, the head tilts, or lighting is uneven on one side.
AMCF instead extracts independent pulse signals from **three regions**
(forehead, left cheek, right cheek), scores each region's signal quality,
and fuses them by confidence — so a noisy region contributes little while
a clean region dominates the final reading.

```
Webcam frame
   │
   ▼
Face detection (Haar Cascade)
   │
   ├──► Forehead ROI ──┐
   ├──► Left Cheek ROI ─┤   (3 independent regions)
   └──► Right Cheek ROI ┘
   │
   ▼
Per-region: CHROM signal → bandpass filter (0.7–4 Hz) → FFT
   │
   ▼
Per-region confidence = peak power / mean power in band
   (a clean periodic pulse → sharp spectral peak → high confidence;
    noise/occlusion → flat spectrum → low confidence)
   │
   ▼
AMCF fusion: weighted average of candidate BPMs by confidence
   │
   ▼
Smoothing (median + exponential) → stable displayed BPM
   │
   ▼
Forehead signal → peak detection → HRV (RMSSD) → Stress level
Chest region → optical flow → Respiration rate
   │
   ▼
Live dashboard: video + FFT spectrum + per-region confidence bars +
breathing coach (triggered on "Elevated" stress) + session report
```

**The science**: every heartbeat sends a pulse of blood through facial
capillaries. Hemoglobin absorbs green light strongly, so skin pixels
subtly brighten and dim with each heartbeat — invisible to the human
eye, but visible in the frequency domain via FFT. CHROM (De Haan &
Jeanne, 2013) combines R, G, B channels in a way that's more robust to
lighting changes than tracking green alone; AMCF adds a multi-region
confidence-fusion layer on top of that.

## 3.1 v5 additions

- **Multi-Person Monitoring**: the largest face in frame gets the full
  AMCF pipeline; up to 3 additional detected faces get lightweight
  single-ROI CHROM tracking (BPM only), shown as a live group list —
  enables classroom/group-scale screening.
- **Accuracy Validation Mode**: log a reference BPM from a phone health
  app or pulse oximeter at any time; the system computes live Mean
  Absolute Error and correlation against its own estimate — measured
  accuracy, not just a claim.
- **Personalized Baseline Learning**: sessions are saved per user name
  to a local JSON profile (`profiles/<name>.json`); future sessions
  compare against that person's own historical average instead of one
  fixed generic threshold.
- **Multimodal Stress Fusion**: physiological stress (HRV/RMSSD) is
  fused with a smile-detection cue (Haar cascade) using a simple rule
  ("smiling detected → step stress level down by one, unless already
  Calm"). **Honesty note**: this is a lightweight heuristic fusion, not
  a trained multimodal classifier — a stronger version would train a
  small model (e.g. on FER2013) to fuse learned facial-emotion
  probabilities with the physiological signal instead of one rule.

## 3.2 Research Mode — Skin-Tone Robustness Study (this project's real differentiator)

**The documented gap**: published rPPG research shows accuracy degrades
significantly on darker skin tones — melanin absorbs light, reducing the
signal-to-noise ratio of the pulse signal. Studies report CHROM-based
methods going from ~5 BPM error (light skin) to ~14 BPM error (dark skin)
(Nowara et al., npj Digital Medicine 2025; Dasari et al.). Most public
rPPG datasets (UBFC-rPPG, PURE, COHFACE) are skewed toward lighter skin
tones, so this bias is under-tested for Indian users specifically.

**What this project adds**: a built-in Research Mode that lets you:
1. Tag a volunteer's Fitzpatrick skin-tone category (self-reported, 3 buckets)
2. Toggle AMCF fusion ON/OFF to compare multi-ROI fusion vs single-ROI
3. Log reference BPM readings (phone pulse app / oximeter) via Accuracy
   Validation Mode
4. Save each run to `research_data.csv` (timestamp, name, skin tone,
   AMCF on/off, MAE, N readings)
5. View aggregated results at `/research` — a bar chart of Mean Absolute
   Error by skin tone, comparing AMCF ON vs OFF

**How to run the study**: recruit 5-8 volunteers spanning light/medium/dark
skin tones, run each volunteer twice (AMCF ON, then Reset & Recalibrate,
then AMCF OFF), logging 3-5 reference readings each time. This produces
real, citable numbers: e.g. "AMCF reduced MAE by X% for darker skin tones
compared to single-ROI, across N volunteers" — a genuine empirical
finding you can defend in a viva, not just a claim.

**Honesty note**: this is a small-scale student study, not a
peer-reviewed clinical trial — say so explicitly in your report. The
value is that it's *real measured data* addressing a *documented gap*,
which is a stronger, more defensible position than claiming a brand-new
algorithm.

## 3.3 v8: True Multi-User Architecture

Previous versions read ONE physical webcam on the server. v8 moves camera
capture to the BROWSER (each user's own device), with the server keeping
independent per-session state — so many people, each on their own laptop
or phone, can use HeartLens **at the same time**, each seeing only their
own vitals. A session ID is generated per browser tab and sent with every
frame; the server maintains a separate processing pipeline per session.

**⚠️ Critical browser security constraint**: browsers only allow camera
access (`getUserMedia`) on a "secure context" — that means `https://` or
`http://localhost` / `http://127.0.0.1`. It will **NOT** work over a
plain LAN IP like `http://10.24.50.67:5000` from another device — the
browser silently blocks camera permission.

**For a single machine (you)**: works immediately at `http://127.0.0.1:5000` or `http://localhost:5000` — no issue.

**For multiple people on different laptops over WiFi** (e.g. classroom
demo), two options:
1. **Quick demo workaround (Chrome only)**: on each OTHER device, visit
   `chrome://flags/#unsafely-treat-insecure-origin-as-secure`, add
   `http://<your-PC's-LAN-IP>:5000`, enable, relaunch Chrome. This tells
   Chrome to trust that specific address for camera access. Fine for a
   classroom demo, not for real deployment.
2. **Proper fix (for real deployment)**: run the app behind HTTPS — e.g.
   `ngrok http 5000` gives a temporary `https://` URL tunneled to your
   local server — the simplest way to let remote people test it for
   real over HTTPS without deploying anywhere.

Mention this constraint explicitly in your viva if asked "can many
people use this from home right now?" — the honest answer is: on the
same machine, yes, immediately; from different devices, yes, but needs
either the Chrome flag (for a local demo) or HTTPS (for real deployment).

## 4. Tech Stack

- **Python** — OpenCV (face detection, video I/O), NumPy/SciPy (signal
  processing: FFT, Butterworth bandpass filter, peak detection)
- **Flask** — backend server, MJPEG video streaming, JSON stats API
- **HTML/CSS/JS** — live dashboard, polls `/stats` every second

## 5. Setup (run this in VS Code)

```bash
cd heartlens
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
python app.py
```

Then open **http://localhost:5000** in your browser. Allow camera
access, sit still and well-lit for ~10 seconds — the BPM reading
stabilizes as the signal buffer fills.

## 6. Project structure

```
heartlens/
├── app.py              # Flask app + rPPG signal processing
├── requirements.txt
├── templates/
│   └── index.html      # Dashboard UI
└── static/
    └── style.css
```

## 7. Talking points for viva / interviews

- **This project's own contribution: AMCF.** Be ready to explain: "CHROM
  gives us a pulse signal per region; AMCF is our fusion layer that
  scores each region's signal quality (confidence = peak power / mean
  power — a peakiness ratio) and combines the three regions'  estimates
  weighted by that confidence, instead of trusting one fragile ROI."
- Explain **why green channel / CHROM** (hemoglobin absorption spectrum,
  and why combining R/G/B beats green alone under lighting changes).
- Explain **why bandpass filter 0.7–4 Hz** (maps to 42–240 BPM — filters
  out lighting flicker and motion noise).
- Explain **confidence metric**: a clean periodic signal produces a
  sharp, narrow spectral peak (high peak/mean ratio); noise or an
  occluded/shadowed region produces a flat spectrum (low ratio) — so
  it's a legitimate, explainable signal-quality proxy, not a black box.
- Be upfront about **limitations**: CHROM and the underlying rPPG
  concept are established published research (cite De Haan & Jeanne,
  2013); this project's original contribution is the multi-region
  confidence-fusion pipeline (AMCF) built on top of it, not a new base
  algorithm. Say what you'd add for true novelty: validate AMCF's
  accuracy improvement against ECG ground truth on a public dataset
  (e.g. UBFC-rPPG) and compare fused vs. single-ROI error rates.

## 8. Future scope (great for "future work" slide)

- Multi-face support (monitor a room, e.g. a classroom stress check)
- Mobile app version using phone camera
- Log historical trends per user (with consent) for a personal health
  dashboard
- Combine with respiration rate (chest movement tracking) for a fuller
  vitals picture

---
**Disclaimer**: Educational/engineering demonstration only. Not a
certified medical device. Do not use for actual diagnosis.
