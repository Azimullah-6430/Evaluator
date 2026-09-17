"""
LearnSphere AI - Smart Education Evaluation Application

Main Flask application.

Features:
- Teacher/student evaluation API
- Handwritten answer-script evaluation
- Question-paper based maximum-mark detection
- Optional rubrics
- Strict plagiarism detection
- Health check
- JSON error responses
"""

from __future__ import annotations

import logging
import os
import traceback
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from app.evaluation.evaluator import EvaluationAgent
from app.plagiarism import PlagiarismDetector


# ============================================================
# ENVIRONMENT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(override=True)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
)


# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

UPLOAD_FOLDER = BASE_DIR / "uploads"

UPLOAD_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)

app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024


ALLOWED_EXTENSIONS = {
    "pdf",
    "jpg",
    "jpeg",
    "png",
}


# ============================================================
# AGENTS
# ============================================================

evaluation_agent = EvaluationAgent()

plagiarism_detector = PlagiarismDetector()


# ============================================================
# HELPERS
# ============================================================

def allowed_file(filename: str) -> bool:
    """
    Check whether an uploaded filename has a supported extension.
    """

    if not filename:
        return False

    if "." not in filename:
        return False

    extension = filename.rsplit(".", 1)[1].lower()

    return extension in ALLOWED_EXTENSIONS


def save_uploaded_file(
    uploaded_file,
    prefix: str,
) -> str:
    """
    Safely save an uploaded file and return its absolute path.
    """

    if uploaded_file is None:
        raise ValueError(
            "No file was supplied."
        )

    if not uploaded_file.filename:
        raise ValueError(
            "Uploaded file has no filename."
        )

    filename = secure_filename(
        uploaded_file.filename
    )

    if not filename:
        raise ValueError(
            "Invalid filename."
        )

    if not allowed_file(filename):
        raise ValueError(
            f"Unsupported file format: {filename}"
        )

    extension = Path(filename).suffix.lower()

    # Add a prefix so question paper and answer script filenames
    # cannot accidentally overwrite one another.
    safe_name = (
        f"{prefix}_{filename}"
    )

    destination = (
        UPLOAD_FOLDER / safe_name
    )

    uploaded_file.save(
        str(destination)
    )

    return str(destination)


# ============================================================
# HOME
# ============================================================

@app.route("/", methods=["GET"])
def home():
    """
    Render the main frontend.
    """

    return render_template(
        "index.html"
    )


# ============================================================
# EVALUATION
# ============================================================

@app.route(
    "/evaluate",
    methods=["POST"],
)
def evaluate():
    """
    Evaluate a student's handwritten answer script and then perform
    strict complete-script plagiarism detection.
    """

    try:

        # --------------------------------------------------------
        # 1. Read student information
        # --------------------------------------------------------

        subject = (
            request.form.get(
                "subject",
                "",
            )
            .strip()
        )

        student_name = (
            request.form.get(
                "student_name",
                "",
            )
            .strip()
        )

        roll_number = (
            request.form.get(
                "roll_number",
                "",
            )
            .strip()
        )

        # --------------------------------------------------------
        # 2. Validate required student information
        # --------------------------------------------------------

        if not subject:
            return jsonify({
                "success": False,
                "error": "Subject is required.",
            }), 400

        # student_name and roll_number are optional but used for plagiarism detection

        # --------------------------------------------------------
        # 3. Read uploaded files
        # --------------------------------------------------------

        question_paper = request.files.get(
            "question_paper"
        )

        answer_script = request.files.get(
            "answer_script"
        )

        rubrics = request.files.get(
            "rubrics"
        )

        # --------------------------------------------------------
        # 4. Validate required files
        # --------------------------------------------------------

        if question_paper is None:
            return jsonify({
                "success": False,
                "error": "Question paper is required.",
            }), 400

        if answer_script is None:
            return jsonify({
                "success": False,
                "error": "Answer script is required.",
            }), 400

        # --------------------------------------------------------
        # 5. Validate filenames
        # --------------------------------------------------------

        if not question_paper.filename:
            return jsonify({
                "success": False,
                "error": "Question paper filename is missing.",
            }), 400

        if not answer_script.filename:
            return jsonify({
                "success": False,
                "error": "Answer script filename is missing.",
            }), 400

        if not allowed_file(
            question_paper.filename
        ):
            return jsonify({
                "success": False,
                "error": (
                    "Unsupported question paper format. "
                    "Use PDF, JPG, JPEG or PNG."
                ),
            }), 400

        if not allowed_file(
            answer_script.filename
        ):
            return jsonify({
                "success": False,
                "error": (
                    "Unsupported answer script format. "
                    "Use PDF, JPG, JPEG or PNG."
                ),
            }), 400

        # --------------------------------------------------------
        # 6. Save question paper
        # --------------------------------------------------------

        qp_path = save_uploaded_file(
            question_paper,
            "question_paper",
        )

        # --------------------------------------------------------
        # 7. Save answer script
        # --------------------------------------------------------

        answer_path = save_uploaded_file(
            answer_script,
            "answer_script",
        )

        # --------------------------------------------------------
        # 8. Save optional rubrics
        # --------------------------------------------------------

        rubric_path = None

        if (
            rubrics is not None
            and rubrics.filename
        ):

            if not allowed_file(
                rubrics.filename
            ):
                return jsonify({
                    "success": False,
                    "error": (
                        "Unsupported rubrics format. "
                        "Use PDF, JPG, JPEG or PNG."
                    ),
                }), 400

            rubric_path = save_uploaded_file(
                rubrics,
                "rubrics",
            )

        # --------------------------------------------------------
        # 9. Build evaluation request
        # --------------------------------------------------------

        evaluation_request = {
            "subject": subject,

            "student_name": student_name,

            "roll_number": roll_number,

            "question_paper": qp_path,

            "answer_script": answer_path,

            "rubrics": rubric_path,
        }

        logger.info(
            "Starting evaluation for student=%s roll=%s subject=%s",
            student_name,
            roll_number,
            subject,
        )

        # --------------------------------------------------------
        # 10. Run Evaluation Agent
        # --------------------------------------------------------

        result = evaluation_agent.evaluate(
            evaluation_request
        )

        logger.info(
            "Evaluation completed for student=%s roll=%s",
            student_name,
            roll_number,
        )

        # --------------------------------------------------------
        # 11. Run Plagiarism Detection Agent
        # --------------------------------------------------------
        #
        # IMPORTANT:
        #
        # The question paper is passed only to identify the
        # examination.
        #
        # It is NOT treated as student answer content.
        #
        # The plagiarism detector compares the COMPLETE answer
        # script.
        # --------------------------------------------------------

        plagiarism_result = (
            plagiarism_detector.check(
                student_name=student_name,

                roll_number=roll_number,

                subject=subject,

                answer_script=answer_path,

                question_paper=qp_path,
            )
        )

        logger.info(
            "Plagiarism check completed for "
            "student=%s roll=%s suspected=%s",
            student_name,
            roll_number,
            plagiarism_result.get(
                "suspected",
                False,
            ),
        )

        # --------------------------------------------------------
        # 12. Return combined result
        # --------------------------------------------------------

        return jsonify({
            "success": True,

            "result": result,

            "plagiarism": plagiarism_result,
        }), 200

    except Exception as exc:

        # --------------------------------------------------------
        # Log complete traceback on server
        # --------------------------------------------------------

        logger.error(
            "Evaluation request failed: %s",
            str(exc),
        )

        logger.error(
            traceback.format_exc()
        )

        # --------------------------------------------------------
        # Return JSON instead of HTML.
        #
        # This prevents frontend errors such as:
        #
        # Unexpected token '<'
        #
        # because Flask's default error page is HTML.
        # --------------------------------------------------------

        return jsonify({
            "success": False,

            "error": str(exc),

            "type": type(exc).__name__,
        }), 500


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route(
    "/health",
    methods=["GET"],
)
def health():
    """
    Health endpoint for Render and deployment monitoring.
    """

    try:

        stored_scripts = (
            plagiarism_detector.count_stored_scripts()
        )

    except Exception:

        stored_scripts = -1

    return jsonify({
        "status": "healthy",

        "service": "LearnSphere AI",

        "evaluation_agent": True,

        "plagiarism_agent": True,

        "stored_plagiarism_scripts": stored_scripts,
    }), 200


# ============================================================
# FILE TOO LARGE
# ============================================================

@app.errorhandler(413)
def request_entity_too_large(error):
    """
    Handle files larger than MAX_CONTENT_LENGTH.
    """

    return jsonify({
        "success": False,

        "error": (
            "Uploaded file is too large. "
            "Maximum allowed size is 50 MB."
        ),
    }), 413


# ============================================================
# NOT FOUND
# ============================================================

@app.errorhandler(404)
def not_found(error):
    """
    Return JSON for unknown API routes.
    """

    if request.path.startswith(
        "/api/"
    ) or request.path == "/evaluate":

        return jsonify({
            "success": False,
            "error": "Endpoint not found.",
        }), 404

    return render_template(
        "index.html"
    ), 200


# ============================================================
# INTERNAL SERVER ERROR
# ============================================================

@app.errorhandler(500)
def internal_server_error(error):
    """
    Return JSON instead of Flask's HTML error page.
    """

    logger.error(
        "Unhandled Flask 500 error: %s",
        str(error),
    )

    return jsonify({
        "success": False,

        "error": (
            "Internal server error. "
            "Check the Render logs for details."
        ),
    }), 500


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                5000,
            )
        ),
        debug=False,
    )
