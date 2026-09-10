import os
import json
import base64
import mimetypes
import re

import requests
import fitz  # PyMuPDF


class EvaluationAgent:
    """
    Core AI Evaluation Agent for handwritten answer scripts.

    Responsibilities:
    1. Analyze the question paper.
    2. Determine subject and total marks.
    3. Analyze the handwritten answer script.
    4. Detect question numbers even when answers are out of order.
    5. Use optional rubrics when provided.
    6. Evaluate answers question-by-question.
    7. Calculate marks, percentage and grade.
    8. Generate detailed feedback.
    """

    def __init__(self):

        # -------------------------------------------------
        # AI Configuration
        # -------------------------------------------------

        self.api_key = os.getenv("GEMINI_API_KEY")

        self.model = os.getenv(
            "GEMINI_MODEL",
            "gemini-2.5-flash"
        )

        self.api_url = (
            "https://generativelanguage.googleapis.com/v1beta/"
            f"models/{self.model}:generateContent"
        )

        # -------------------------------------------------
        # Safety / Evaluation Settings
        # -------------------------------------------------

        self.max_file_size = 50 * 1024 * 1024

        self.supported_extensions = {
            ".pdf",
            ".jpg",
            ".jpeg",
            ".png"
        }

    # =====================================================
    # MAIN EVALUATION FUNCTION
    # =====================================================

    def evaluate(self, evaluation_request):

        """
        Main entry point used by Flask.

        evaluation_request contains:

        {
            "subject": "...",
            "student_name": "...",
            "roll_number": "...",
            "question_paper": "...",
            "answer_script": "...",
            "rubrics": "..." or None
        }
        """

        # -------------------------------------------------
        # Validate request
        # -------------------------------------------------

        self._validate_request(evaluation_request)

        subject = evaluation_request["subject"]
        student_name = evaluation_request.get(
            "student_name",
            ""
        )

        roll_number = evaluation_request.get(
            "roll_number",
            ""
        )

        question_paper_path = evaluation_request[
            "question_paper"
        ]

        answer_script_path = evaluation_request[
            "answer_script"
        ]

        rubrics_path = evaluation_request.get(
            "rubrics"
        )

        # -------------------------------------------------
        # STEP 1
        # Analyze Question Paper
        # -------------------------------------------------

        question_paper_data = self.analyze_question_paper(
            question_paper_path,
            subject
        )

        # -------------------------------------------------
        # STEP 2
        # Analyze Answer Script
        # -------------------------------------------------

        answer_script_data = self.analyze_answer_script(
            answer_script_path,
            subject,
            question_paper_data
        )

        # -------------------------------------------------
        # STEP 3
        # Analyze Rubrics if supplied
        # -------------------------------------------------

        rubric_data = None

        if rubrics_path:
            rubric_data = self.analyze_rubrics(
                rubrics_path,
                subject,
                question_paper_data
            )

        # -------------------------------------------------
        # STEP 4
        # Evaluate Answers
        # -------------------------------------------------

        evaluation = self.evaluate_answers(
            subject=subject,
            question_paper=question_paper_data,
            answer_script=answer_script_data,
            rubrics=rubric_data
        )

        # -------------------------------------------------
        # STEP 5
        # Calculate Final Result
        # -------------------------------------------------

        final_result = self.calculate_final_result(
            evaluation=evaluation,
            question_paper=question_paper_data
        )

        # -------------------------------------------------
        # STEP 6
        # Add Student Information
        # -------------------------------------------------

        final_result["student"] = {
            "name": student_name,
            "roll_number": roll_number,
            "subject": subject
        }

        final_result["evaluation_metadata"] = {
            "rubrics_used": rubrics_path is not None,
            "evaluation_engine": "AI Evaluation Agent",
            "question_paper_used_as_source_of_truth": True
        }

        return final_result

    # =====================================================
    # VALIDATION
    # =====================================================

    def _validate_request(self, request):

        required_fields = [
            "subject",
            "question_paper",
            "answer_script"
        ]

        for field in required_fields:

            if not request.get(field):

                raise ValueError(
                    f"Required field missing: {field}"
                )

        # -------------------------------------------------
        # Check API key
        # -------------------------------------------------

        if not self.api_key:

            raise ValueError(
                "GEMINI_API_KEY is not configured. "
                "Add your Gemini API key to the .env file."
            )

        # -------------------------------------------------
        # Check files
        # -------------------------------------------------

        files_to_check = [
            request["question_paper"],
            request["answer_script"]
        ]

        if request.get("rubrics"):
            files_to_check.append(
                request["rubrics"]
            )

        for file_path in files_to_check:

            if not os.path.exists(file_path):

                raise FileNotFoundError(
                    f"File not found: {file_path}"
                )

            extension = os.path.splitext(
                file_path
            )[1].lower()

            if extension not in self.supported_extensions:

                raise ValueError(
                    f"Unsupported file type: {extension}"
                )

    # =====================================================
    # QUESTION PAPER ANALYZER
    # =====================================================

    def analyze_question_paper(
        self,
        file_path,
        subject
    ):

        """
        Extracts the question structure.

        The question paper is the SOURCE OF TRUTH.

        The AI must determine:

        - Subject
        - Total marks
        - Question numbers
        - Question text
        - Marks
        - Question type
        - Subquestions
        """

        content = self._prepare_file_content(
            file_path
        )

        prompt = self._question_paper_prompt(
            subject
        )

        response = self._call_gemini(
            prompt,
            content
        )

        data = self._parse_json_response(
            response
        )

        # -------------------------------------------------
        # Critical validation
        # -------------------------------------------------

        if "total_marks" not in data:

            raise ValueError(
                "Question paper analysis did not "
                "identify total marks."
            )

        if not data.get("questions"):

            raise ValueError(
                "No questions were extracted from "
                "the question paper."
            )

        # -------------------------------------------------
        # Force numeric total marks
        # -------------------------------------------------

        data["total_marks"] = self._safe_number(
            data["total_marks"]
        )

        return data

    # =====================================================
    # ANSWER SCRIPT ANALYZER
    # =====================================================

    def analyze_answer_script(
        self,
        file_path,
        subject,
        question_paper
    ):

        """
        Analyzes handwritten answer sheets.

        IMPORTANT:

        The agent does NOT assume that:

        Page 1 = Q1
        Page 2 = Q2
        Page 3 = Q3

        Instead, it identifies the question number
        written by the student.
        """

        content = self._prepare_file_content(
            file_path
        )

        question_summary = json.dumps(
            question_paper,
            ensure_ascii=False,
            indent=2
        )

        prompt = f"""
You are the handwritten answer-script analysis
component of an academic evaluation system.

SUBJECT:
{subject}

QUESTION PAPER STRUCTURE:
{question_summary}

Your task is to read the student's handwritten
answer script.

IMPORTANT RULES:

1. Do NOT assume answers are written in question
   paper order.

2. The student may answer:

   Q1
   Q5
   Q2
   Q10
   Q3

3. Identify the actual question number written
   by the student.

4. Detect answers across multiple pages.

5. If an answer continues on the next page,
   combine the pages.

6. Do not confuse page number with question number.

7. Preserve mathematical expressions as accurately
   as possible.

8. Preserve important units, formulas and equations.

9. If handwriting is unclear, use surrounding
   context and the question paper to interpret it.

10. Never invent an answer that is not reasonably
    visible in the script.

Return ONLY valid JSON.

Required format:

{{
    "answers": [
        {{
            "question_number": "1",
            "answer_text": "...",
            "page_numbers": [1, 2],
            "confidence": 0.95
        }}
    ]
}}

confidence must be between 0 and 1.
"""

        response = self._call_gemini(
            prompt,
            content
        )

        data = self._parse_json_response(
            response
        )

        if not isinstance(
            data.get("answers"),
            list
        ):

            raise ValueError(
                "Answer script analysis failed."
            )

        return data

    # =====================================================
    # RUBRIC ANALYZER
    # =====================================================

    def analyze_rubrics(
        self,
        file_path,
        subject,
        question_paper
    ):

        """
        Rubrics are OPTIONAL.

        If provided, they are used as an evaluation
        reference.

        If not provided, the evaluator creates its
        own criteria from the question paper.
        """

        content = self._prepare_file_content(
            file_path
        )

        question_summary = json.dumps(
            question_paper,
            ensure_ascii=False,
            indent=2
        )

        prompt = f"""
You are a rubric analysis component for an
academic evaluation system.

SUBJECT:
{subject}

QUESTION PAPER:
{question_summary}

Read the uploaded rubric.

Extract:

- question number
- marking criteria
- expected concepts
- marks for each criterion
- partial marking rules
- important keywords
- required steps where applicable

Do not change the maximum marks defined by the
question paper.

Return ONLY valid JSON.

Format:

{{
    "rubrics": [
        {{
            "question_number": "1",
            "criteria": [
                {{
                    "criterion": "...",
                    "marks": 2
                }}
            ]
        }}
    ]
}}
"""

        response = self._call_gemini(
            prompt,
            content
        )

        return self._parse_json_response(
            response
        )

    # =====================================================
    # ANSWER EVALUATION ENGINE
    # =====================================================

    def evaluate_answers(
        self,
        subject,
        question_paper,
        answer_script,
        rubrics=None
    ):

        """
        Evaluate every question individually.
        """

        question_data = json.dumps(
            question_paper,
            ensure_ascii=False,
            indent=2
        )

        answer_data = json.dumps(
            answer_script,
            ensure_ascii=False,
            indent=2
        )

        rubric_data = json.dumps(
            rubrics,
            ensure_ascii=False,
            indent=2
        ) if rubrics else "NO RUBRICS PROVIDED"

        prompt = f"""
You are an expert academic answer-sheet evaluator.

SUBJECT:
{subject}

==================================================
QUESTION PAPER — SOURCE OF TRUTH
==================================================

{question_data}

==================================================
STUDENT ANSWERS
==================================================

{answer_data}

==================================================
RUBRICS
==================================================

{rubric_data}

==================================================
CORE EVALUATION RULES
==================================================

RULE 1 — QUESTION PAPER IS AUTHORITATIVE

Use the question paper as the ONLY source for:

- question numbers
- question text
- maximum marks
- total marks
- question type

Never assume a default exam total such as 50 or 100.

If the question paper is for 20 marks,
the final maximum must be 20.

RULE 2 — SUBJECT CONSISTENCY

Evaluate ONLY the supplied subject:

{subject}

Do not use concepts from another subject.

For example:

If subject = Mathematics,
do not produce Physics feedback.

RULE 3 — QUESTION ORDER

The student may answer questions in any order.

Match answers using their detected question number.

RULE 4 — UNANSWERED QUESTIONS

If a question has no corresponding student answer,
mark it as unanswered.

Do not invent an answer.

RULE 5 — RUBRICS

If rubrics are provided, use them.

If rubrics are not provided, create appropriate
evaluation criteria from the question itself.

RULE 6 — PARTIAL MARKS

Award partial marks when the student demonstrates
partial understanding.

Do not automatically give zero simply because
the final answer is incorrect.

RULE 7 — LONG ANSWERS

For long answers, evaluate:

- conceptual correctness
- completeness
- relevance
- explanation
- important points
- examples where required
- formulas where required
- conclusion where appropriate

RULE 8 — 16-MARK QUESTIONS

For high-mark questions, evaluate the answer
holistically and criterion-by-criterion.

Do not treat a 16-mark answer like a 1-mark answer.

RULE 9 — MATHEMATICS / NUMERICAL QUESTIONS

Check:

- formula
- substitution
- calculation
- units
- intermediate steps
- final answer

Award appropriate partial marks when the method
is correct but the final calculation contains
an error.

RULE 10 — MCQ

For MCQs:

Correct answer → full marks.

Incorrect answer → zero unless the question paper
explicitly specifies another marking scheme.

RULE 11 — NO HALLUCINATION

Do not invent content that is not present.

RULE 12 — FEEDBACK

For every attempted question provide:

- marks obtained
- maximum marks
- correctness
- what was done well
- mistakes
- missing points
- what should have been written
- improvement suggestion

==================================================
RETURN FORMAT
==================================================

Return ONLY valid JSON.

Use this structure:

{{
    "questions": [
        {{
            "question_number": "1",
            "maximum_marks": 2,
            "marks_obtained": 2,
            "status": "correct",
            "question_type": "MCQ",
            "feedback": {{
                "what_was_done_well": "...",
                "mistakes": "...",
                "missing_points": "...",
                "expected_answer": "...",
                "improvement": "..."
            }}
        }}
    ],
    "overall_feedback": "...",
    "strengths": [],
    "weaknesses": []
}}

IMPORTANT:

marks_obtained MUST NEVER exceed maximum_marks.

Evaluate every question from the question paper,
including unanswered questions.
"""

        response = self._call_gemini(
            prompt,
            []
        )

        return self._parse_json_response(
            response
        )

    # =====================================================
    # FINAL RESULT CALCULATOR
    # =====================================================

    def calculate_final_result(
        self,
        evaluation,
        question_paper
    ):

        questions = evaluation.get(
            "questions",
            []
        )

        # -------------------------------------------------
        # Maximum marks come from question paper
        # -------------------------------------------------

        maximum_marks = self._safe_number(
            question_paper.get(
                "total_marks",
                0
            )
        )

        marks_obtained = 0

        validated_questions = []

        for question in questions:

            max_marks = self._safe_number(
                question.get(
                    "maximum_marks",
                    0
                )
            )

            obtained = self._safe_number(
                question.get(
                    "marks_obtained",
                    0
                )
            )

            # ---------------------------------------------
            # Prevent impossible marks
            # ---------------------------------------------

            obtained = max(
                0,
                min(
                    obtained,
                    max_marks
                )
            )

            question["maximum_marks"] = max_marks

            question["marks_obtained"] = obtained

            marks_obtained += obtained

            validated_questions.append(
                question
            )

        # -------------------------------------------------
        # Never allow obtained marks > total marks
        # -------------------------------------------------

        marks_obtained = min(
            marks_obtained,
            maximum_marks
        )

        # -------------------------------------------------
        # Percentage
        # -------------------------------------------------

        if maximum_marks > 0:

            percentage = (
                marks_obtained /
                maximum_marks
            ) * 100

        else:

            percentage = 0

        # -------------------------------------------------
        # Grade
        # -------------------------------------------------

        grade = self._calculate_grade(
            percentage
        )

        return {
            "maximum_marks": maximum_marks,

            "marks_obtained": round(
                marks_obtained,
                2
            ),

            "percentage": round(
                percentage,
                2
            ),

            "grade": grade,

            "questions": validated_questions,

            "overall_feedback": evaluation.get(
                "overall_feedback",
                ""
            ),

            "strengths": evaluation.get(
                "strengths",
                []
            ),

            "weaknesses": evaluation.get(
                "weaknesses",
                []
            )
        }

    # =====================================================
    # GRADE CALCULATOR
    # =====================================================

    def _calculate_grade(self, percentage):

        if percentage >= 90:
            return "A+"

        if percentage >= 80:
            return "A"

        if percentage >= 70:
            return "B+"

        if percentage >= 60:
            return "B"

        if percentage >= 50:
            return "C"

        if percentage >= 40:
            return "D"

        return "F"

    # =====================================================
    # FILE PROCESSING
    # =====================================================

    def _prepare_file_content(
        self,
        file_path
    ):

        """
        Convert uploaded files into content that can
        be sent to the multimodal AI model.

        PDFs are sent as PDF data.

        Images are sent as base64 image data.
        """

        extension = os.path.splitext(
            file_path
        )[1].lower()

        file_size = os.path.getsize(
            file_path
        )

        if file_size > self.max_file_size:

            raise ValueError(
                "File exceeds the maximum allowed size."
            )

        if extension == ".pdf":

            return [{
                "inline_data": {
                    "mime_type": "application/pdf",
                    "data": self._encode_file(
                        file_path
                    )
                }
            }]

        if extension in {
            ".jpg",
            ".jpeg",
            ".png"
        }:

            mime_type = mimetypes.guess_type(
                file_path
            )[0]

            if not mime_type:

                mime_type = "image/jpeg"

            return [{
                "inline_data": {
                    "mime_type": mime_type,
                    "data": self._encode_file(
                        file_path
                    )
                }
            }]

        raise ValueError(
            "Unsupported file format."
        )

    # =====================================================
    # FILE ENCODING
    # =====================================================

    def _encode_file(
        self,
        file_path
    ):

        with open(
            file_path,
            "rb"
        ) as file:

            return base64.b64encode(
                file.read()
            ).decode("utf-8")

    # =====================================================
    # GEMINI API CALL
    # =====================================================

    def _call_gemini(
        self,
        prompt,
        file_content
    ):

        """
        Send prompt + document/image content
        to Gemini.
        """

        parts = [
            {
                "text": prompt
            }
        ]

        # Add uploaded document/image content
        if file_content:

            parts.extend(
                file_content
            )

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": parts
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json"
            }
        }

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key
        }

        response = requests.post(
            self.api_url,
            headers=headers,
            json=payload,
            timeout=180
        )

        # -------------------------------------------------
        # API error handling
        # -------------------------------------------------

        if response.status_code != 200:

            try:

                error_data = response.json()

            except Exception:

                error_data = response.text

            raise RuntimeError(
                "Gemini API error: "
                f"{error_data}"
            )

        data = response.json()

        # -------------------------------------------------
        # Extract generated text
        # -------------------------------------------------

        try:

            candidates = data[
                "candidates"
            ]

            text = candidates[0][
                "content"
            ][
                "parts"
            ][0][
                "text"
            ]

            return text

        except (
            KeyError,
            IndexError,
            TypeError
        ):

            raise RuntimeError(
                "Unexpected response received "
                "from Gemini API."
            )

    # =====================================================
    # JSON PARSER
    # =====================================================

    def _parse_json_response(
        self,
        response
    ):

        """
        Clean and parse JSON returned by AI.
        """

        if isinstance(
            response,
            dict
        ):

            return response

        text = response.strip()

        # -------------------------------------------------
        # Remove markdown code fences
        # -------------------------------------------------

        text = re.sub(
            r"^```json\s*",
            "",
            text,
            flags=re.IGNORECASE
        )

        text = re.sub(
            r"^```\s*",
            "",
            text
        )

        text = re.sub(
            r"\s*```$",
            "",
            text
        )

        text = text.strip()

        try:

            return json.loads(
                text
            )

        except json.JSONDecodeError:

            # ---------------------------------------------
            # Attempt to find JSON object
            # ---------------------------------------------

            start = text.find("{")
            end = text.rfind("}")

            if start != -1 and end != -1:

                json_text = text[
                    start:end + 1
                ]

                try:

                    return json.loads(
                        json_text
                    )

                except json.JSONDecodeError:
                    pass

            raise ValueError(
                "AI returned invalid JSON."
            )

    # =====================================================
    # NUMBER CONVERSION
    # =====================================================

    def _safe_number(
        self,
        value
    ):

        try:

            if isinstance(
                value,
                str
            ):

                value = value.replace(
                    ",",
                    ""
                ).strip()

            return float(value)

        except (
            ValueError,
            TypeError
        ):

            return 0
