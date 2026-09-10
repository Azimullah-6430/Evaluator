import os

from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

from app.evaluation.evaluator import EvaluationAgent

load_dotenv()

app = Flask(__name__)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

ALLOWED_EXTENSIONS = {"pdf", "jpg", "jpeg", "png"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

evaluation_agent = EvaluationAgent()


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/evaluate", methods=["POST"])
def evaluate():
    try:
        subject = request.form.get("subject", "").strip()
        student_name = request.form.get("student_name", "").strip()
        roll_number = request.form.get("roll_number", "").strip()

        question_paper = request.files.get("question_paper")
        answer_script = request.files.get("answer_script")
        rubrics = request.files.get("rubrics")

        if not subject:
            return jsonify({
                "success": False,
                "error": "Subject is required."
            }), 400

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

        qp_filename = secure_filename(question_paper.filename)
        qp_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            qp_filename
        )
        question_paper.save(qp_path)

        answer_filename = secure_filename(answer_script.filename)
        answer_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            answer_filename
        )
        answer_script.save(answer_path)

        rubric_path = None

        if rubrics and rubrics.filename:
            if not allowed_file(rubrics.filename):
                return jsonify({
                    "success": False,
                    "error": "Unsupported rubrics format."
                }), 400

            rubric_filename = secure_filename(rubrics.filename)
            rubric_path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                rubric_filename
            )
            rubrics.save(rubric_path)

        evaluation_request = {
            "subject": subject,
            "student_name": student_name,
            "roll_number": roll_number,
            "question_paper": qp_path,
            "answer_script": answer_path,
            "rubrics": rubric_path
        }

        result = evaluation_agent.evaluate(evaluation_request)

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


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "running",
        "service": "Smart Education Evaluation Agent"
    })


if __name__ == "__main__":
    app.run()
