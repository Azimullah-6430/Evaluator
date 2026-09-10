import os
import json
import base64
import re
from typing import Dict, Any, List, Optional

import requests
from dotenv import load_dotenv


# ==========================================================
# LOAD ENVIRONMENT VARIABLES
# ==========================================================

load_dotenv()


# ==========================================================
# EVALUATION AGENT
# ==========================================================

class EvaluationAgent:
    """
    Smart Education System - AI Evaluation Agent

    Core principles:

    1. Every uploaded question paper is treated as a new exam.
    2. The uploaded question paper is the source of truth.
    3. Total marks are extracted dynamically.
    4. Maximum marks for each question are extracted dynamically.
    5. Student answers may appear in any order.
    6. Rubrics are optional.
    7. AI marks can never exceed question maximum marks.
    8. Final marks can never exceed examination total marks.
    9. Subject supplied by the teacher is enforced.
    10. Missing question-paper marks are never silently guessed.
    """

    # ======================================================
    # INITIALIZATION
    # ======================================================

    def __init__(self):

        # Render must contain GEMINI_API_KEY
        self.api_key = os.getenv("GEMINI_API_KEY")

        # IMPORTANT:
        # Use the currently configured Gemini model.
        # This default prevents accidental fallback to
        # the previously unavailable gemini-3.5-flash.
        self.model = os.getenv(
            "GEMINI_MODEL",
            "gemini-3.6-flash"
        ).strip()

        # Gemini REST endpoint
        self.api_url = (
            "https://generativelanguage.googleapis.com/"
            f"v1beta/models/{self.model}:generateContent"
        )

        # Maximum request time
        self.timeout = 180

    # ======================================================
    # MAIN PIPELINE
    # ======================================================

    def evaluate(
        self,
        request_data: Dict[str, Any]
    ) -> Dict[str, Any]:

        print("\n======================================")
        print("EVALUATION AGENT STARTED")
        print("======================================")

        # --------------------------------------------------
        # Validate request
        # --------------------------------------------------

        self._validate_request(request_data)

        subject = str(
            request_data["subject"]
        ).strip()

        print("Subject:", subject)
        print("Gemini model:", self.model)

        # --------------------------------------------------
        # STEP 1
        # Analyze question paper
        # --------------------------------------------------

        print("\n[1/5] Analyzing question paper...")

        question_paper = self.analyze_question_paper(
            request_data["question_paper"],
            subject
        )

        print(
            "Question paper extracted successfully."
        )

        # --------------------------------------------------
        # STEP 2
        # Validate question paper
        # --------------------------------------------------

        print(
            "\n[2/5] Validating question paper structure..."
        )

        self._validate_question_paper_structure(
            question_paper
        )

        print(
            "Total marks:",
            question_paper["total_marks"]
        )

        print(
            "Questions extracted:",
            len(question_paper["questions"])
        )

        # --------------------------------------------------
        # STEP 3
        # Analyze answer script
        # --------------------------------------------------

        print(
            "\n[3/5] Analyzing handwritten answer script..."
        )

        answer_script = self.analyze_answer_script(
            request_data["answer_script"],
            subject,
            question_paper
        )

        print(
            "Answer script analyzed successfully."
        )

        # --------------------------------------------------
        # STEP 4
        # Optional rubric
        # --------------------------------------------------

        rubric_data = None

        if request_data.get("rubrics"):

            print(
                "\n[4/5] Analyzing uploaded rubric..."
            )

            rubric_data = self.analyze_rubrics(
                request_data["rubrics"],
                subject,
                question_paper
            )

            print(
                "Rubric analyzed successfully."
            )

        else:

            print(
                "\n[4/5] No rubric supplied."
            )

            print(
                "AI will generate evaluation criteria."
            )

        # --------------------------------------------------
        # STEP 5
        # Evaluate answers
        # --------------------------------------------------

        print(
            "\n[5/5] Evaluating answers..."
        )

        evaluation = self.evaluate_answers(
            subject=subject,
            question_paper=question_paper,
            answer_script=answer_script,
            rubrics=rubric_data
        )

        print(
            "AI evaluation completed."
        )

        # --------------------------------------------------
        # SERVER-SIDE MARK CALCULATION
        # --------------------------------------------------

        print(
            "\nCalculating final marks safely..."
        )

        final_result = self.calculate_final_result(
            question_paper,
            evaluation
        )

        # --------------------------------------------------
        # Student information
        # --------------------------------------------------

        final_result["student_name"] = (
            request_data.get(
                "student_name",
                ""
            )
        )

        final_result["roll_number"] = (
            request_data.get(
                "roll_number",
                ""
            )
        )

        final_result["subject"] = subject

        print(
            "\n======================================"
        )

        print(
            "EVALUATION COMPLETED"
        )

        print(
            "Marks:",
            final_result["marks_obtained"],
            "/",
            final_result["maximum_marks"]
        )

        print(
            "Percentage:",
            final_result["percentage"]
        )

        print(
            "Grade:",
            final_result["grade"]
        )

        print(
            "======================================\n"
        )

        return final_result

    # ======================================================
    # REQUEST VALIDATION
    # ======================================================

    def _validate_request(
        self,
        request_data: Dict[str, Any]
    ):

        if not isinstance(
            request_data,
            dict
        ):

            raise ValueError(
                "Invalid evaluation request."
            )

        if not self.api_key:

            raise ValueError(
                "GEMINI_API_KEY is not configured. "
                "Add GEMINI_API_KEY to the Render Environment Variables."
            )

        required_fields = [
            "subject",
            "question_paper",
            "answer_script"
        ]

        for field in required_fields:

            if not request_data.get(field):

                raise ValueError(
                    f"Required field missing: {field}"
                )

        # Verify files exist before calling Gemini

        for field in [
            "question_paper",
            "answer_script"
        ]:

            path = request_data[field]

            if not os.path.exists(path):

                raise FileNotFoundError(
                    f"{field} file not found: {path}"
                )

    # ======================================================
    # QUESTION PAPER PROMPT
    # ======================================================

    def _question_paper_prompt(
        self,
        subject: str
    ) -> str:

        return f"""
You are the Question Paper Analysis Engine of a
high-accuracy academic evaluation system.

The teacher supplied subject is:

{subject}

The uploaded question paper is the ONLY authoritative
source of truth.

This paper may be completely different from every
previous examination.

Analyze THIS uploaded paper independently.

==========================================================
SUBJECT
==========================================================

The subject supplied by the teacher is:

{subject}

Stay strictly within this subject.

Do not introduce another subject.

==========================================================
TOTAL MARKS
==========================================================

Extract the actual examination total marks from the
uploaded question paper.

NEVER assume:

20
50
100

The paper may have any total.

Look for:

Total Marks
Maximum Marks
Max Marks
Total
section totals
marking schemes
instruction-based totals

Examples:

Total Marks: 20

means:

"total_marks": 20

Maximum Marks: 50

means:

"total_marks": 50

If the paper clearly contains:

Part A = 10
Part B = 20
Part C = 20

then:

"total_marks": 50

ONLY if those values clearly represent the examination
mark distribution.

==========================================================
QUESTION MARKS
==========================================================

Extract the maximum marks for EVERY question.

Examples:

1. Define operating system. [2]

2. Explain process scheduling. [5]

3. Explain deadlock with example. [10]

Return:

1 -> 2
2 -> 5
3 -> 10

DO NOT assign default marks.

DO NOT assume all questions have equal marks.

DO NOT use marks from previous papers.

==========================================================
MARK EXTRACTION
==========================================================

Look carefully for:

[1]
[2]
[5]
[10]

(1)
(2)
(5)
(10)

1 Mark
2 Marks
5 Marks
10 Marks

1M
2M
5M
10M

tables

section schemes such as:

10 x 1 = 10
5 x 2 = 10
4 x 5 = 20

If individual question marks exist,
prefer those over assumptions.

==========================================================
QUESTION NUMBERS
==========================================================

Extract exact question numbers.

Examples:

1
2
3
3(a)
3(b)
4(i)
4(ii)
5
10
16

Do not invent question numbers.

==========================================================
QUESTION TEXT
==========================================================

Extract the complete question text as accurately as
possible.

Preserve:

- equations
- mathematical symbols
- options
- technical terms
- numerical values
- diagrams described in text
- important instructions

==========================================================
QUESTION TYPE
==========================================================

Identify the question type.

Possible values:

MCQ
True/False
Fill in the Blank
Very Short Answer
Short Answer
Long Answer
Essay
Numerical
Mathematical Problem
Derivation
Theory
Programming
Case Study
Matching
Other

==========================================================
SECTIONS
==========================================================

Detect sections such as:

Part A
Part B
Part C
Section I
Section II
Section III

==========================================================
SUBQUESTIONS
==========================================================

Preserve subquestions separately.

Example:

3(a)
3(b)
3(c)

Do not incorrectly merge them.

==========================================================
CHOICES
==========================================================

Detect:

Answer any 5
Answer any 3
Attempt either 4(a) or 4(b)
Internal choice
OR

Preserve this information.

==========================================================
NO INVENTION
==========================================================

Never invent:

- questions
- marks
- total marks
- sections
- choices
- question numbers

If something is genuinely unreadable,
return null or mark it uncertain.

Do NOT guess.

==========================================================
SOURCE OF TRUTH
==========================================================

The extracted question paper controls the entire
evaluation.

The evaluator MUST use:

question.maximum_marks

as the absolute maximum for that question.

The final result MUST use:

question_paper.total_marks

as the examination maximum.

==========================================================
OUTPUT
==========================================================

Return ONLY valid JSON.

Use this structure:

{{
    "subject": "{subject}",
    "total_marks": 50,
    "total_marks_source": "question_paper",
    "questions": [
        {{
            "question_number": "1",
            "question_text": "Example question",
            "maximum_marks": 2,
            "question_type": "MCQ",
            "section": "Part A",
            "subquestions": [],
            "choice_information": null
        }}
    ]
}}

IMPORTANT:

total_marks MUST come from THIS uploaded paper.

maximum_marks MUST come from THIS uploaded paper.

Never use marks from previous evaluations.

Never use hardcoded marks.

Return ONLY JSON.
"""

    # ======================================================
    # ANALYZE QUESTION PAPER
    # ======================================================

    def analyze_question_paper(
        self,
        file_path: str,
        subject: str
    ) -> Dict[str, Any]:

        prompt = self._question_paper_prompt(
            subject
        )

        response = self._call_gemini(
            prompt,
            [file_path]
        )

        result = self._parse_json_response(
            response
        )

        return result

    # ======================================================
    # QUESTION PAPER VALIDATION
    # ======================================================

    def _validate_question_paper_structure(
        self,
        question_paper: Dict[str, Any]
    ):

        if not isinstance(
            question_paper,
            dict
        ):

            raise ValueError(
                "Question paper analysis did not return valid data."
            )

        total_marks = question_paper.get(
            "total_marks"
        )

        questions = question_paper.get(
            "questions"
        )

        # --------------------------------------------------
        # Total marks
        # --------------------------------------------------

        if total_marks is None:

            raise ValueError(
                "Unable to extract total marks from the question paper."
            )

        try:

            total_marks = float(
                total_marks
            )

        except Exception:

            raise ValueError(
                f"Extracted total marks are invalid: {total_marks}"
            )

        if total_marks <= 0:

            raise ValueError(
                "Question paper total marks must be greater than zero."
            )

        # --------------------------------------------------
        # Questions
        # --------------------------------------------------

        if not isinstance(
            questions,
            list
        ) or not questions:

            raise ValueError(
                "No questions could be extracted from the question paper."
            )

        # --------------------------------------------------
        # Validate every question
        # --------------------------------------------------

        seen_numbers = set()

        for question in questions:

            if not isinstance(
                question,
                dict
            ):

                raise ValueError(
                    "Invalid question structure returned by AI."
                )

            question_number = question.get(
                "question_number"
            )

            maximum_marks = question.get(
                "maximum_marks"
            )

            if not question_number:

                raise ValueError(
                    "A question was extracted without a question number."
                )

            normalized = self._normalize_question_number(
                str(question_number)
            )

            # Duplicate question numbers are dangerous
            if normalized in seen_numbers:

                raise ValueError(
                    f"Duplicate question number detected: "
                    f"{question_number}"
                )

            seen_numbers.add(
                normalized
            )

            # --------------------------------------------------
            # Maximum marks MUST exist
            # --------------------------------------------------

            if maximum_marks is None:

                raise ValueError(
                    f"Maximum marks missing for question "
                    f"{question_number}."
                )

            try:

                marks = float(
                    maximum_marks
                )

            except Exception:

                raise ValueError(
                    f"Invalid marks for question "
                    f"{question_number}: {maximum_marks}"
                )

            if marks <= 0:

                raise ValueError(
                    f"Maximum marks must be greater than zero "
                    f"for question {question_number}."
                )

        question_paper["total_marks"] = total_marks

    # ======================================================
    # ANSWER SCRIPT PROMPT
    # ======================================================

    def _answer_script_prompt(
        self,
        subject: str,
        question_paper: Dict[str, Any]
    ) -> str:

        question_structure = json.dumps(
            question_paper,
            indent=2,
            ensure_ascii=False
        )

        return f"""
You are the Answer Script Analysis Engine.

Subject:

{subject}

The question paper below is authoritative.

QUESTION PAPER:

{question_structure}

Analyze the uploaded handwritten answer script.

==========================================================
ANSWER ORDER
==========================================================

The student may answer questions in ANY order.

Examples:

Question 5
Question 2
Question 10
Question 1
Question 7

Or:

Q10 appears before Q3.

Or:

Part B is answered before Part A.

Do NOT assume page order equals question order.

Identify answers using their actual question number.

==========================================================
QUESTION MATCHING
==========================================================

Every detected answer MUST correspond to a question
present in the supplied question paper.

Do not create new question numbers.

Do not merge unrelated answers.

==========================================================
HANDWRITING
==========================================================

Read the handwritten content carefully.

If handwriting is unclear:

- preserve the uncertainty
- do not invent text
- use a lower confidence score

==========================================================
MISSING ANSWERS
==========================================================

Only return answers that actually appear.

Do not create answers for unanswered questions.

==========================================================
OUTPUT
==========================================================

Return ONLY valid JSON.

{{
    "answers": [
        {{
            "question_number": "5",
            "answer_text": "...",
            "confidence": 0.95
        }}
    ]
}}
"""

    # ======================================================
    # ANALYZE ANSWER SCRIPT
    # ======================================================

    def analyze_answer_script(
        self,
        file_path: str,
        subject: str,
        question_paper: Dict[str, Any]
    ) -> Dict[str, Any]:

        prompt = self._answer_script_prompt(
            subject,
            question_paper
        )

        response = self._call_gemini(
            prompt,
            [file_path]
        )

        result = self._parse_json_response(
            response
        )

        if not isinstance(
            result,
            dict
        ):

            raise ValueError(
                "Answer script analysis returned invalid data."
            )

        if "answers" not in result:

            raise ValueError(
                "Answer script analysis did not return an answers list."
            )

        if not isinstance(
            result["answers"],
            list
        ):

            raise ValueError(
                "Answer script answers must be a list."
            )

        return result

    # ======================================================
    # RUBRIC PROMPT
    # ======================================================

    def _rubric_prompt(
        self,
        subject: str,
        question_paper: Dict[str, Any]
    ) -> str:

        question_structure = json.dumps(
            question_paper,
            indent=2,
            ensure_ascii=False
        )

        return f"""
You are an academic rubric analysis engine.

Subject:

{subject}

Question paper:

{question_structure}

Analyze the uploaded rubric.

Map rubric criteria to the appropriate questions.

Do NOT modify the maximum marks from the question paper.

The question paper is always the hard upper limit.

If the rubric conflicts with the question paper,
the question paper maximum marks must win.

Return ONLY valid JSON.

Expected structure:

{{
    "rubrics": [
        {{
            "question_number": "1",
            "criteria": [
                {{
                    "criterion": "...",
                    "marks": 1
                }}
            ]
        }}
    ]
}}
"""

    # ======================================================
    # ANALYZE RUBRICS
    # ======================================================

    def analyze_rubrics(
        self,
        file_path: str,
        subject: str,
        question_paper: Dict[str, Any]
    ) -> Dict[str, Any]:

        prompt = self._rubric_prompt(
            subject,
            question_paper
        )

        response = self._call_gemini(
            prompt,
            [file_path]
        )

        result = self._parse_json_response(
            response
        )

        return result

    # ======================================================
    # EVALUATION PROMPT
    # ======================================================

    def _evaluation_prompt(
        self,
        subject: str,
        question_paper: Dict[str, Any],
        answer_script: Dict[str, Any],
        rubrics: Optional[Dict[str, Any]]
    ) -> str:

        qp_json = json.dumps(
            question_paper,
            indent=2,
            ensure_ascii=False
        )

        answer_json = json.dumps(
            answer_script,
            indent=2,
            ensure_ascii=False
        )

        if rubrics:

            rubric_json = json.dumps(
                rubrics,
                indent=2,
                ensure_ascii=False
            )

        else:

            rubric_json = "NO RUBRIC PROVIDED"

        return f"""
You are a high-accuracy academic answer evaluation engine.

SUBJECT:

{subject}

==========================================================
QUESTION PAPER — SOURCE OF TRUTH
==========================================================

{qp_json}

==========================================================
STUDENT ANSWERS
==========================================================

{answer_json}

==========================================================
RUBRIC
==========================================================

{rubric_json}

==========================================================
CORE RULES
==========================================================

1. Evaluate ONLY using the supplied question paper.

2. Stay strictly within the supplied subject.

3. Never use marks from previous examinations.

4. Every question has a supplied maximum_marks.

5. Award marks using:

0 <= awarded_marks <= maximum_marks

6. NEVER exceed maximum_marks.

7. Partial marks should be awarded when academically justified.

8. Long answers must be evaluated for:

- correctness
- conceptual understanding
- explanation
- relevant points
- examples
- calculations
- derivations
- conclusion
- required terminology

9. Numerical problems must consider:

- formula
- substitution
- calculation
- units
- logical steps
- final answer

10. Mathematics questions:

Correct intermediate steps can receive partial marks.

11. MCQs:

Compare the student's selected answer with the correct answer.

12. If an answer is missing:

awarded_marks = 0

13. Do not invent an answer.

14. Student answer order must NOT affect marks.

15. Match using question number.

16. Use the supplied rubric when available.

17. If no rubric is supplied, create academically appropriate
evaluation criteria from the actual question.

18. Feedback must correspond to the actual question.

19. Never generate feedback from another subject.

20. Do not blindly trust any AI-computed total.

==========================================================
FEEDBACK
==========================================================

For EVERY question provide:

- question number
- maximum marks
- awarded marks
- status
- what was done well
- mistakes
- missing points
- expected answer
- improvement suggestion

==========================================================
IMPORTANT MARKING RULE
==========================================================

Before returning each evaluation verify:

awarded_marks <= maximum_marks

==========================================================
OUTPUT
==========================================================

Return ONLY valid JSON.

Use:

{{
    "evaluations": [
        {{
            "question_number": "1",
            "maximum_marks": 2,
            "awarded_marks": 2,
            "status": "Correct",
            "what_was_done_well": "...",
            "mistakes": [],
            "missing_points": [],
            "expected_answer": "...",
            "improvement": "..."
        }}
    ],
    "overall_feedback": "..."
}}

DO NOT include markdown.

DO NOT include explanations outside JSON.
"""

    # ======================================================
    # EVALUATE ANSWERS
    # ======================================================

    def evaluate_answers(
        self,
        subject: str,
        question_paper: Dict[str, Any],
        answer_script: Dict[str, Any],
        rubrics: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:

        prompt = self._evaluation_prompt(
            subject,
            question_paper,
            answer_script,
            rubrics
        )

        response = self._call_gemini(
            prompt,
            []
        )

        result = self._parse_json_response(
            response
        )

        if not isinstance(
            result,
            dict
        ):

            raise ValueError(
                "Evaluation result is not a JSON object."
            )

        evaluations = result.get(
            "evaluations"
        )

        if evaluations is None:

            raise ValueError(
                "Gemini evaluation response does not contain "
                "'evaluations'."
            )

        if not isinstance(
            evaluations,
            list
        ):

            raise ValueError(
                "'evaluations' must be a list."
            )

        return result

    # ======================================================
    # FINAL RESULT CALCULATION
    # ======================================================

    def calculate_final_result(
        self,
        question_paper: Dict[str, Any],
        evaluation: Dict[str, Any]
    ) -> Dict[str, Any]:

        questions = question_paper[
            "questions"
        ]

        ai_evaluations = evaluation.get(
            "evaluations",
            []
        )

        # --------------------------------------------------
        # Create AI lookup
        # --------------------------------------------------

        evaluation_map = {}

        for item in ai_evaluations:

            if not isinstance(
                item,
                dict
            ):
                continue

            q_number = str(
                item.get(
                    "question_number",
                    ""
                )
            ).strip()

            if not q_number:
                continue

            normalized_number = (
                self._normalize_question_number(
                    q_number
                )
            )

            evaluation_map[
                normalized_number
            ] = item

        # --------------------------------------------------
        # Final evaluations
        # --------------------------------------------------

        final_evaluations = []

        obtained_marks = 0.0

        # IMPORTANT:
        # Iterate over the QUESTION PAPER,
        # not AI output.
        #
        # Therefore the question paper controls:
        #
        # - questions
        # - question numbers
        # - maximum marks
        # - final total
        # --------------------------------------------------

        for question in questions:

            q_number = str(
                question[
                    "question_number"
                ]
            ).strip()

            maximum_marks = float(
                question[
                    "maximum_marks"
                ]
            )

            normalized_number = (
                self._normalize_question_number(
                    q_number
                )
            )

            ai_item = evaluation_map.get(
                normalized_number
            )

            # --------------------------------------------------
            # AI evaluated question
            # --------------------------------------------------

            if ai_item:

                raw_awarded = ai_item.get(
                    "awarded_marks",
                    0
                )

                try:

                    awarded_marks = float(
                        raw_awarded
                    )

                except Exception:

                    awarded_marks = 0.0

                # --------------------------------------------------
                # HARD SAFETY LIMIT
                # --------------------------------------------------

                awarded_marks = max(
                    0.0,
                    min(
                        awarded_marks,
                        maximum_marks
                    )
                )

                final_item = dict(
                    ai_item
                )

                final_item[
                    "question_number"
                ] = q_number

                final_item[
                    "maximum_marks"
                ] = self._clean_number(
                    maximum_marks
                )

                final_item[
                    "awarded_marks"
                ] = self._clean_number(
                    awarded_marks
                )

            # --------------------------------------------------
            # No answer / no evaluation
            # --------------------------------------------------

            else:

                awarded_marks = 0.0

                final_item = {

                    "question_number":
                        q_number,

                    "maximum_marks":
                        self._clean_number(
                            maximum_marks
                        ),

                    "awarded_marks":
                        0,

                    "status":
                        "Not Answered",

                    "what_was_done_well":
                        "",

                    "mistakes":
                        [],

                    "missing_points":
                        [
                            "No answer detected for this question."
                        ],

                    "expected_answer":
                        "",

                    "improvement":
                        (
                            "Attempt the question and provide "
                            "the required explanation."
                        )
                }

            obtained_marks += awarded_marks

            final_evaluations.append(
                final_item
            )

        # --------------------------------------------------
        # Examination total
        # --------------------------------------------------

        total_marks = float(
            question_paper[
                "total_marks"
            ]
        )

        # --------------------------------------------------
        # HARD FINAL SAFETY LIMIT
        # --------------------------------------------------

        obtained_marks = min(
            obtained_marks,
            total_marks
        )

        # --------------------------------------------------
        # Percentage
        # --------------------------------------------------

        if total_marks > 0:

            percentage = (
                obtained_marks /
                total_marks
            ) * 100

        else:

            percentage = 0.0

        # --------------------------------------------------
        # Grade
        # --------------------------------------------------

        grade = self._calculate_grade(
            percentage
        )

        return {

            "maximum_marks":
                self._clean_number(
                    total_marks
                ),

            "marks_obtained":
                self._clean_number(
                    obtained_marks
                ),

            "percentage":
                round(
                    percentage,
                    2
                ),

            "grade":
                grade,

            "question_wise_evaluation":
                final_evaluations,

            "overall_feedback":
                evaluation.get(
                    "overall_feedback",
                    ""
                )
        }

    # ======================================================
    # GRADE
    # ======================================================

    def _calculate_grade(
        self,
        percentage: float
    ) -> str:

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

    # ======================================================
    # QUESTION NUMBER NORMALIZATION
    # ======================================================

    def _normalize_question_number(
        self,
        question_number: str
    ) -> str:

        value = str(
            question_number
        ).strip().lower()

        value = value.replace(
            "question",
            ""
        )

        value = value.replace(
            "q.",
            ""
        )

        value = value.replace(
            "q",
            ""
        )

        value = value.replace(
            " ",
            ""
        )

        value = value.rstrip(
            ".:"
        )

        return value

    # ======================================================
    # GEMINI API CALL
    # ======================================================

    def _call_gemini(
        self,
        prompt: str,
        file_paths: List[str]
    ) -> str:

        # --------------------------------------------------
        # API KEY
        # --------------------------------------------------

        if not self.api_key:

            raise ValueError(
                "GEMINI_API_KEY is not configured. "
                "Check Render → Environment."
            )

        # --------------------------------------------------
        # MODEL
        # --------------------------------------------------

        if not self.model:

            raise ValueError(
                "GEMINI_MODEL is empty."
            )

        print(
            "\nCalling Gemini:"
        )

        print(
            "Model:",
            self.model
        )

        print(
            "Files:",
            len(file_paths)
        )

        # --------------------------------------------------
        # Content parts
        # --------------------------------------------------

        parts = [
            {
                "text": prompt
            }
        ]

        # --------------------------------------------------
        # Attach files
        # --------------------------------------------------

        for file_path in file_paths:

            if not file_path:
                continue

            if not os.path.exists(
                file_path
            ):

                raise FileNotFoundError(
                    f"File not found: {file_path}"
                )

            mime_type = (
                self._get_mime_type(
                    file_path
                )
            )

            print(
                "Attaching:",
                os.path.basename(
                    file_path
                ),
                mime_type
            )

            with open(
                file_path,
                "rb"
            ) as file:

                file_bytes = file.read()

            if not file_bytes:

                raise ValueError(
                    f"Uploaded file is empty: {file_path}"
                )

            encoded = (
                base64.b64encode(
                    file_bytes
                ).decode(
                    "utf-8"
                )
            )

            parts.append(
                {
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": encoded
                    }
                }
            )

        # --------------------------------------------------
        # Request payload
        # --------------------------------------------------

        payload = {

            "contents": [
                {
                    "parts": parts
                }
            ],

            "generationConfig": {

                "temperature": 0.1,

                "responseMimeType":
                    "application/json"
            }
        }

        headers = {

            "Content-Type":
                "application/json",

            "x-goog-api-key":
                self.api_key
        }

        # --------------------------------------------------
        # API REQUEST
        # --------------------------------------------------

        try:

            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )

        except requests.Timeout as error:

            raise RuntimeError(
                "Gemini API request timed out after "
                f"{self.timeout} seconds."
            ) from error

        except requests.RequestException as error:

            raise RuntimeError(
                f"Could not connect to Gemini API: {error}"
            ) from error

        # --------------------------------------------------
        # API ERROR
        # --------------------------------------------------

        if not response.ok:

            try:

                error_data = response.json()

            except Exception:

                error_data = response.text

            raise RuntimeError(
                "Gemini API request failed.\n"
                f"HTTP Status: {response.status_code}\n"
                f"Model: {self.model}\n"
                f"Response: {error_data}"
            )

        # --------------------------------------------------
        # Parse API response
        # --------------------------------------------------

        try:

            data = response.json()

        except Exception as error:

            raise RuntimeError(
                "Gemini returned a non-JSON HTTP response."
            ) from error

        # --------------------------------------------------
        # Check candidates
        # --------------------------------------------------

        candidates = data.get(
            "candidates"
        )

        if not candidates:

            # Gemini may return a prompt/safety block
            # without candidates.

            raise RuntimeError(
                "Gemini returned no candidates.\n"
                f"Full response: {data}"
            )

        # --------------------------------------------------
        # Extract text safely
        # --------------------------------------------------

        try:

            candidate = candidates[0]

            content = candidate.get(
                "content"
            )

            if not content:

                raise RuntimeError(
                    "Gemini candidate has no content."
                )

            response_parts = content.get(
                "parts"
            )

            if not response_parts:

                raise RuntimeError(
                    "Gemini response contains no parts."
                )

            text = response_parts[0].get(
                "text"
            )

            if not text:

                raise RuntimeError(
                    "Gemini response contains no text."
                )

            return text

        except Exception as error:

            raise RuntimeError(
                "Gemini returned an unexpected response structure.\n"
                f"Response: {data}"
            ) from error

    # ======================================================
    # JSON PARSER
    # ======================================================

    def _parse_json_response(
        self,
        response: str
    ) -> Dict[str, Any]:

        if response is None:

            raise ValueError(
                "AI returned no response."
            )

        text = str(
            response
        ).strip()

        if not text:

            raise ValueError(
                "AI returned an empty response."
            )

        # --------------------------------------------------
        # Remove markdown fences
        # --------------------------------------------------

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

        # --------------------------------------------------
        # Direct JSON
        # --------------------------------------------------

        try:

            parsed = json.loads(
                text
            )

            if not isinstance(
                parsed,
                dict
            ):

                raise ValueError(
                    "AI JSON response must be an object."
                )

            return parsed

        except json.JSONDecodeError:
            pass

        # --------------------------------------------------
        # Find JSON object inside response
        # --------------------------------------------------

        start = text.find(
            "{"
        )

        end = text.rfind(
            "}"
        )

        if (
            start != -1
            and
            end != -1
            and
            end > start
        ):

            json_text = text[
                start:end + 1
            ]

            try:

                parsed = json.loads(
                    json_text
                )

                if not isinstance(
                    parsed,
                    dict
                ):

                    raise ValueError(
                        "AI JSON response must be an object."
                    )

                return parsed

            except json.JSONDecodeError as error:

                # Keep a small portion for debugging
                preview = text[:1000]

                raise ValueError(
                    "AI returned invalid JSON.\n"
                    f"Response preview: {preview}"
                ) from error

        # --------------------------------------------------
        # No JSON found
        # --------------------------------------------------

        raise ValueError(
            "AI returned invalid JSON.\n"
            f"Response preview: {text[:1000]}"
        )

    # ======================================================
    # MIME TYPE
    # ======================================================

    def _get_mime_type(
        self,
        file_path: str
    ) -> str:

        extension = (
            os.path.splitext(
                file_path
            )[1]
            .lower()
        )

        mime_types = {

            ".pdf":
                "application/pdf",

            ".jpg":
                "image/jpeg",

            ".jpeg":
                "image/jpeg",

            ".png":
                "image/png"
        }

        mime_type = mime_types.get(
            extension
        )

        if not mime_type:

            raise ValueError(
                f"Unsupported file type: {extension}"
            )

        return mime_type

    # ======================================================
    # CLEAN NUMBER
    # ======================================================

    def _clean_number(
        self,
        value: float
    ):

        value = float(
            value
        )

        if value.is_integer():

            return int(
                value
            )

        return round(
            value,
            2
        )
