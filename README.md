# TrafficGuard — Traffic Violation Detector

Flask dashboard for uploading a traffic video, detecting number plates with YOLO, reading plate text with EasyOCR, and checking exact owner records using MongoDB.

## Required files

Copy your existing `owner_database.py` and `license_plate_detector.pt` into this project folder before running it. For helmet violation detection, also provide a compatible trained `helmet_detector.pt` model. Do not commit `.env`, personal owner records, or generated videos/crops.

## Run locally

```bash
python -m venv venv
# Windows PowerShell:
venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000.

Set the environment variables needed by `owner_database.py` (for example, your MongoDB URI) in your local `.env` file if that module uses python-dotenv. Never commit secrets.

## Render

- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn --workers 1 --timeout 600 app:app`
- Add the required MongoDB environment variables in Render's Environment settings.
- Ensure `license_plate_detector.pt` is available to the service. If it exceeds GitHub's file-size limit, store it in an appropriate model-artifact location and download it during build/start.
- This synchronous demo processes one uploaded video per worker. Use short clips on limited instances; long videos may exceed hosting limits. Render's local filesystem may be ephemeral, so generated videos may disappear on restart/redeploy unless persistent storage is configured.

The app has a 100 MB upload limit and temporarily removes the original uploaded file after processing. The annotated output is kept at `output/plate_recognition.mp4` while the instance filesystem persists.

## Important

This is an academic prototype. OCR and owner lookup results may be wrong or incomplete; manually verify evidence before issuing any challan.


## Helmet violation detection

The pipeline supports an optional `helmet_detector.pt` YOLO model. It must be trained to detect a class named exactly one of: `no_helmet`, `without_helmet`, `without helmet`, `no helmet`, `helmetless`, `nohelmet`, or `withouthelmet`. If the file is missing, the app continues with number-plate recognition only and logs a warning. A generic number-plate model cannot detect helmet violations. Review detections manually; they are potential violations, not automatic legal findings.
