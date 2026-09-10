import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

from app.evaluation.evaluator import EvaluationAgent


# ---------------------------------------------------------
# Flask Application
# ---------------------------------------------------------

app = Flask(__name__)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

ALLOWED_EXTENSIONS = {
    "pdf",
    "jpg",
    "jpeg",
    "png"
}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB


# Create upload directory automatically
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ---------------------------------------------------------
# Evaluation Agent
# ---------------------------------------------------------

evaluation_agent = EvaluationAgent()


# ---------------------------------------------------------
# Helper Function
# ---------------------------------------------------------

def allowed_file(filename):
    """
    Check whether the uploaded file has a supported extension.
    """

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


# ---------------------------------------------------------
# Home Page
# ---------------------------------------------------------

@app.route("/")
def home():
    """
    Display the Evaluation Agent interface.
    """

    return render_template("index.html")


# ---------------------------------------------------------
# Evaluation API
# ---------------------------------------------------------

@app.route("/evaluate", methods=["POST"])
def evaluate():
    """
    Receive question paper, answer script and optional rubrics.
    Then send them to the Evaluation Agent.
    """

    try:

        # -------------------------------------------------
        # Get form information
        # -------------------------------------------------

        subject = request.form.get("subject", "").strip()
        student_name = request.form.get("student_name", "").strip()
        roll_number = request.form.get("roll_number", "").strip()

        # -------------------------------------------------
        # Validate subject
        # -------------------------------------------------

        if not subject:
            return jsonify({
                "success": False,
                "error": "Subject is required."
            }), 400

        # -------------------------------------------------
        # Get uploaded files
        # -------------------------------------------------

        question_paper = request.files.get("question_paper")
        answer_script = request.files.get("answer_script")
        rubrics = request.files.get("rubrics")

        # -------------------------------------------------
        # Validate Question Paper
        # -------------------------------------------------

        if not question_paper:
            return jsonify({
                "success": False,
                "error": "Question paper is required."
            }), 400

        if not allowed_file(question_paper.filename):
            return jsonify({
                "success": False,
                "error": "Unsupported question paper format."
            }), 400

        # -------------------------------------------------
        # Validate Answer Script
        # -------------------------------------------------

        if not answer_script:
            return jsonify({
                "success": False,
                "error": "Answer script is required."
            }), 400

        if not allowed_file(answer_script.filename):
            return jsonify({
                "success": False,
                "error": "Unsupported answer script format."
            }), 400

        # -------------------------------------------------
        # Save Question Paper
        # -------------------------------------------------

        qp_filename = secure_filename(question_paper.filename)

        qp_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            qp_filename
        )

        question_paper.save(qp_path)

        # -------------------------------------------------
        # Save Answer Script
        # -------------------------------------------------

        answer_filename = secure_filename(answer_script.filename)

        answer_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            answer_filename
        )

        answer_script.save(answer_path)

        # -------------------------------------------------
        # Save Optional Rubrics
        # -------------------------------------------------

        rubric_path = None

        if rubrics and rubrics.filename:

            if not allowed_file(rubrics.filename):
                return jsonify({
                    "success": False,
                    "error": "Unsupported rubrics format."
                }), 400

            rubric_filename = secure_filename(
                rubrics.filename
            )

            rubric_path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                rubric_filename
            )

            rubrics.save(rubric_path)

        # -------------------------------------------------
        # Prepare Evaluation Request
        # -------------------------------------------------

        evaluation_request = {
            "subject": subject,
            "student_name": student_name,
            "roll_number": roll_number,

            "question_paper": qp_path,

            "answer_script": answer_path,

            "rubrics": rubric_path
        }

        # -------------------------------------------------
        # Send to Evaluation Agent
        # -------------------------------------------------

        result = evaluation_agent.evaluate(
            evaluation_request
        )

        # -------------------------------------------------
        # Return Result
        # -------------------------------------------------

        return jsonify({
            "success": True,
            "result": result
        })

    except Exception as error:

        print("Evaluation Error:", error)

        return jsonify({
            "success": False,
            "error": str(error)
        }), 500


# ---------------------------------------------------------
# Health Check
# ---------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "running",
        "service": "Smart Education Evaluation Agent"
    })
