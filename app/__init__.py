import os
import traceback

from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

from app.evaluation.evaluator import EvaluationAgent


# ==========================================================
# LOAD ENVIRONMENT VARIABLES
# ==========================================================

load_dotenv()


# ==========================================================
# CREATE FLASK APP
# ==========================================================

app = Flask(__name__)


# ==========================================================
# UPLOAD CONFIGURATION
# ==========================================================

BASE_DIR = os.path.abspath(
    os.path.dirname(__file__)
)

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "uploads"
)

ALLOWED_EXTENSIONS = {
    "pdf",
    "jpg",
    "jpeg",
    "png"
}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Maximum upload size = 50 MB
app.config["MAX_CONTENT_LENGTH"] = (
    50 * 1024 * 1024
)

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


# ==========================================================
# CREATE EVALUATION AGENT
# ==========================================================

evaluation_agent = EvaluationAgent()


# ==========================================================
# FILE VALIDATION
# ==========================================================

def allowed_file(filename):

    return (
        "." in filename
        and
        filename.rsplit(
            ".",
            1
        )[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ==========================================================
# HOME PAGE
# ==========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ==========================================================
# EVALUATION API
# ==========================================================

@app.route(
    "/evaluate",
    methods=["POST"]
)
def evaluate():

    try:

        print(
            "\n======================================"
        )

        print(
            "STARTING NEW EVALUATION"
        )

        print(
            "======================================"
        )

        # --------------------------------------------------
        # GET FORM DATA
        # --------------------------------------------------

        subject = request.form.get(
            "subject",
            ""
        ).strip()

        student_name = request.form.get(
            "student_name",
            ""
        ).strip()

        roll_number = request.form.get(
            "roll_number",
            ""
        ).strip()

        # --------------------------------------------------
        # GET FILES
        # --------------------------------------------------

        question_paper = request.files.get(
            "question_paper"
        )

        answer_script = request.files.get(
            "answer_script"
        )

        rubrics = request.files.get(
            "rubrics"
        )

        # --------------------------------------------------
        # LOG BASIC INFORMATION
        # --------------------------------------------------

        print(
            "Subject:",
            subject
        )

        print(
            "Student:",
            student_name
        )

        print(
            "Roll Number:",
            roll_number
        )

        # --------------------------------------------------
        # VALIDATE SUBJECT
        # --------------------------------------------------

        if not subject:

            return jsonify({

                "success": False,

                "error":
                    "Subject is required."

            }), 400

        # --------------------------------------------------
        # VALIDATE QUESTION PAPER
        # --------------------------------------------------

        if not question_paper:

            return jsonify({

                "success": False,

                "error":
                    "Question paper is required."

            }), 400

        if not question_paper.filename:

            return jsonify({

                "success": False,

                "error":
                    "Question paper filename is missing."

            }), 400

        if not allowed_file(
            question_paper.filename
        ):

            return jsonify({

                "success": False,

                "error":
                    "Unsupported question paper format. "
                    "Use PDF, JPG, JPEG or PNG."

            }), 400

        # --------------------------------------------------
        # VALIDATE ANSWER SCRIPT
        # --------------------------------------------------

        if not answer_script:

            return jsonify({

                "success": False,

                "error":
                    "Answer script is required."

            }), 400

        if not answer_script.filename:

            return jsonify({

                "success": False,

                "error":
                    "Answer script filename is missing."

            }), 400

        if not allowed_file(
            answer_script.filename
        ):

            return jsonify({

                "success": False,

                "error":
                    "Unsupported answer script format. "
                    "Use PDF, JPG, JPEG or PNG."

            }), 400

        # --------------------------------------------------
        # SAVE QUESTION PAPER
        # --------------------------------------------------

        qp_filename = secure_filename(
            question_paper.filename
        )

        qp_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            qp_filename
        )

        question_paper.save(
            qp_path
        )

        print(
            "Question Paper Saved:",
            qp_path
        )

        # --------------------------------------------------
        # SAVE ANSWER SCRIPT
        # --------------------------------------------------

        answer_filename = secure_filename(
            answer_script.filename
        )

        answer_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            answer_filename
        )

        answer_script.save(
            answer_path
        )

        print(
            "Answer Script Saved:",
            answer_path
        )

        # --------------------------------------------------
        # SAVE OPTIONAL RUBRICS
        # --------------------------------------------------

        rubric_path = None

        if (
            rubrics
            and
            rubrics.filename
        ):

            if not allowed_file(
                rubrics.filename
            ):

                return jsonify({

                    "success": False,

                    "error":
                        "Unsupported rubrics format. "
                        "Use PDF, JPG, JPEG or PNG."

                }), 400

            rubric_filename = secure_filename(
                rubrics.filename
            )

            rubric_path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                rubric_filename
            )

            rubrics.save(
                rubric_path
            )

            print(
                "Rubrics Saved:",
                rubric_path
            )

        else:

            print(
                "No rubrics uploaded. "
                "AI will generate evaluation criteria."
            )

        # --------------------------------------------------
        # BUILD EVALUATION REQUEST
        # --------------------------------------------------

        evaluation_request = {

            "subject":
                subject,

            "student_name":
                student_name,

            "roll_number":
                roll_number,

            "question_paper":
                qp_path,

            "answer_script":
                answer_path,

            "rubrics":
                rubric_path
        }

        print(
            "\n--------------------------------------"
        )

        print(
            "SENDING REQUEST TO EVALUATION AGENT"
        )

        print(
            "--------------------------------------"
        )

        # --------------------------------------------------
        # RUN EVALUATION AGENT
        # --------------------------------------------------

        result = evaluation_agent.evaluate(
            evaluation_request
        )

        print(
            "\n--------------------------------------"
        )

        print(
            "EVALUATION COMPLETED SUCCESSFULLY"
        )

        print(
            "--------------------------------------"
        )

        # --------------------------------------------------
        # RETURN SUCCESS JSON
        # --------------------------------------------------

        return jsonify({

            "success":
                True,

            "result":
                result

        }), 200

    except Exception as error:

        # --------------------------------------------------
        # PRINT COMPLETE ERROR TO RENDER LOG
        # --------------------------------------------------

        print(
            "\n======================================"
        )

        print(
            "EVALUATION ERROR"
        )

        print(
            "======================================"
        )

        print(
            "Error:",
            str(error)
        )

        print(
            "\nFULL TRACEBACK:"
        )

        traceback.print_exc()

        print(
            "\n======================================"
        )

        # --------------------------------------------------
        # ALWAYS RETURN JSON
        # --------------------------------------------------

        return jsonify({

            "success":
                False,

            "error":
                str(error)

        }), 500


# ==========================================================
# HEALTH CHECK
# ==========================================================

@app.route(
    "/health",
    methods=["GET"]
)
def health():

    return jsonify({

        "status":
            "running",

        "service":
            "Smart Education Evaluation Agent",

        "evaluation_agent":
            "ready"

    }), 200


# ==========================================================
# GLOBAL 413 HANDLER
# FILE TOO LARGE
# ==========================================================

@app.errorhandler(413)
def file_too_large(error):

    return jsonify({

        "success":
            False,

        "error":
            "Uploaded file is too large. "
            "Maximum allowed size is 50 MB."

    }), 413


# ==========================================================
# GLOBAL 404 HANDLER
# ==========================================================

@app.errorhandler(404)
def page_not_found(error):

    return jsonify({

        "success":
            False,

        "error":
            "Requested endpoint was not found."

    }), 404


# ==========================================================
# GLOBAL 500 HANDLER
# ==========================================================

@app.errorhandler(500)
def internal_server_error(error):

    return jsonify({

        "success":
            False,

        "error":
            "Internal server error. "
            "Check the Render logs for details."

    }), 500
