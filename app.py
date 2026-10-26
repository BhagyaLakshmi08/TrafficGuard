import os
import uuid
from pathlib import Path

from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from werkzeug.utils import secure_filename

from owner_database import find_owner

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change-this-secret-before-production")
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB upload limit

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "output"
CROP_DIR = BASE_DIR / "plate_crops"
OUTPUT_VIDEO = OUTPUT_DIR / "plate_recognition.mp4"
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "mpeg", "mpg"}

for folder in (UPLOAD_DIR, OUTPUT_DIR, CROP_DIR):
    folder.mkdir(parents=True, exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def dashboard():
    return render_template(
        "index.html",
        video_exists=OUTPUT_VIDEO.exists(),
        lookup_done=False,
        owner=None
    )


@app.route("/upload", methods=["POST"])
def upload_video():
    uploaded = request.files.get("video")

    if not uploaded or not uploaded.filename:
        flash("Choose a traffic video before uploading.", "error")
        return redirect(url_for("dashboard"))

    if not allowed_file(uploaded.filename):
        flash("Unsupported file type. Upload MP4, AVI, MOV, MKV, MPEG or MPG.", "error")
        return redirect(url_for("dashboard"))

    # Use a generated filename to avoid path traversal and filename collisions.
    safe_original_name = secure_filename(uploaded.filename)
    suffix = Path(safe_original_name).suffix.lower()
    input_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    uploaded.save(input_path)

    try:
        model_path = BASE_DIR / "license_plate_detector.pt"
        if not model_path.exists():
            flash("The model file license_plate_detector.pt is missing from the deployed app.", "error")
            return redirect(url_for("dashboard"))

        # Import only when needed; the pipeline loads YOLO and EasyOCR models.
        from complete_pipeline import main as process_video

        success = process_video(
            video_path=input_path,
            output_video=OUTPUT_VIDEO,
            crop_dir=CROP_DIR
        )

        if success and OUTPUT_VIDEO.exists():
            flash(f"Processing completed for {safe_original_name}. Scroll down to view the annotated video.", "success")
        else:
            flash("Video processing did not complete. Check the Render logs for the pipeline error.", "error")

    except Exception:
        app.logger.exception("Video processing failed")
        flash("Video processing failed. Check the Render logs for details.", "error")
    finally:
        # The source upload is temporary; keep only the processed output.
        try:
            input_path.unlink(missing_ok=True)
        except OSError:
            app.logger.warning("Could not remove temporary upload: %s", input_path)

    return redirect(url_for("dashboard"))


@app.route("/lookup", methods=["GET"])
def lookup():
    plate = request.args.get("plate", "").strip()
    owner = None
    lookup_done = False

    if plate:
        lookup_done = True
        try:
            owner = find_owner(plate)
        except Exception:
            app.logger.exception("Vehicle lookup failed")
            return (
                "Vehicle lookup failed. Check your MongoDB connection "
                "and owner_database.py configuration.",
                500
            )

    return render_template(
        "index.html",
        video_exists=OUTPUT_VIDEO.exists(),
        lookup_done=lookup_done,
        owner=owner
    )


@app.route("/video")
def processed_video():
    if not OUTPUT_VIDEO.exists():
        return "Processed video is not available yet.", 404
    return send_file(OUTPUT_VIDEO, mimetype="video/mp4", conditional=True)


@app.errorhandler(413)
def too_large(_error):
    flash("The uploaded video exceeds the 100 MB limit.", "error")
    return redirect(url_for("dashboard"))


if __name__ == "__main__":
    # Render provides PORT; local development falls back to 5000.
    port = int(os.environ.get("PORT", "5000"))
    app.run(debug=False, host="0.0.0.0", port=port)
