# app.py
# Flask web server for MedAI

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from model_engine import MedicalAITool
import os

app = Flask(__name__)

# Upload folder for reports/images
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Initialize shared AI engine
ai = MedicalAITool()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict_vitals", methods=["POST"])
def predict_vitals():
    """
    ICU Vitals endpoint.
    Expects JSON with numeric fields and returns XGBoost-based risk.
    """
    try:
        payload = request.get_json(force=True)
        data = {k: float(v) for k, v in payload.items()}
        result = ai.analyze_vitals(data)
        return jsonify({"status": "success", "data": result})
    except Exception as e:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": f"Vitals prediction failed: {str(e)}",
                }
            ),
            400,
        )


@app.route("/analyze_file", methods=["POST"])
def analyze_file():
    """
    File upload endpoint.
    Uses Gemini to analyze radiology/ECG/reports based on scan_type.
    """
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file part in request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"status": "error", "message": "No file selected"}), 400

    scan_type = request.form.get("scan_type", "normal")

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        result = ai.analyze_file(filepath, scan_type)
        return jsonify({"status": "success", "data": result})
    except Exception as e:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": f"File analysis failed: {str(e)}",
                }
            ),
            500,
        )


if __name__ == "__main__":
    # For development; in production use gunicorn/uwsgi
    app.run(debug=True, port=5000)
