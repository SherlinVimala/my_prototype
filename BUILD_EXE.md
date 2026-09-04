# Turning HeartLens into a Standalone Windows App (.exe)

This packages the Flask app + Python + all libraries into a single
double-clickable `.exe` — no VS Code, no terminal, no `venv activate`
needed. Anyone with a webcam can just double-click and use it.

**Note**: this still runs the server on the SAME machine that has the
webcam (see the note about multi-user in chat) — it doesn't change that,
it just removes the need for VS Code/terminal to launch it.

## Steps

1. Open VS Code terminal in the `heartlens` folder, activate venv:
   ```
   .\venv\Scripts\activate
   ```

2. Install PyInstaller:
   ```
   pip install pyinstaller
   ```

3. Build the exe (run this exact command — it bundles the templates
   and static folders which Flask needs):
   ```
   pyinstaller --onefile --add-data "templates;templates" --add-data "static;static" --name HeartLens app.py
   ```

4. Wait 1-2 minutes. The finished app appears at:
   ```
   dist\HeartLens.exe
   ```

5. Double-click `dist\HeartLens.exe` to test. Windows Defender/SmartScreen
   may warn "unknown publisher" the first time — click "More info" →
   "Run anyway" (this is normal for unsigned personal-project exe files).

6. It should auto-open your browser to the dashboard. First run will be
   a bit slow (unpacking), later runs are faster.

## For submission

- You can zip `dist\HeartLens.exe` alone and share it — the person
  running it needs a webcam but does NOT need Python installed.
- Mention in your report: "Packaged as a standalone Windows executable
  using PyInstaller for easy demonstration without a development
  environment."

## Limitations to mention honestly (future scope, not now)

- This is a desktop app tied to one machine's webcam — not a
  cloud-hosted or mobile app.
- A true mobile app would need the phone's own camera accessed via the
  browser (JavaScript `getUserMedia`) instead of Python's
  `cv2.VideoCapture(0)`, which only works with a camera physically
  attached to the machine running the code — that's a bigger redesign
  for future work, not this submission.
