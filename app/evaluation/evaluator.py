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
    Smart Education System - Academic Evaluation Agent.

    Design principles:
    - Uploaded question paper is the source of truth.
    - Total marks are extracted from the uploaded paper.
    - Maximum marks are extracted for each question.
    - Student answers can appear in any order.
    - Rubrics are optional.
    - Marks are always clamped server-side.
    - Final marks are calculated by the server.
    - Gemini model is fixed in code.
    """

    # ======================================================
    # INITIALIZATION
    # ======================================================

    def __init__(self):

        # Render only needs this environment variable.
        self.api_key = os.getenv("GEMINI_API_KEY")

        # IMPORTANT:
        # Do NOT read GEMINI_MODEL from Render.
        # The model is fixed here.
        self.model = "gemini-3.6-flash"

        self.api_url = (
            "https://generativelanguage.googleapis.com/"
            f"v1beta/models/{self.model}:generateContent"
        )

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

        self._validate_request(request_data)

        subject = str(
            request_data.get("subject", "")
        ).strip()

        print("Subject:", subject)
        print("Gemini model:", self.model)

        # --------------------------------------------------
        # 1. QUESTION PAPER
        # --------------------------------------------------

        print("\n[1/5] Analyzing question paper...")

        question_paper = self.analyze_question_paper(
            request_data["question_paper"],
            subject
        )

        self._validate_question_paper_structure(
            question_paper
        )

        print(
            "Question paper extracted successfully."
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
        # 2. ANSWER SCRIPT
        # --------------------------------------------------

        print("\n[2/5] Analyzing handwritten answer script...")

        answer_script = self.analyze_answer_script(
            request_data["answer_script"],
            subject,
            question_paper
        )

        print(
            "Answer script analyzed successfully."
        )

        # --------------------------------------------------
        # 3. RUBRIC
        # --------------------------------------------------

        rubric_data = None

        if request_data.get("rubrics"):

            print("\n[3/5] Analyzing uploaded rubric...")

            rubric_data = self.analyze_rubrics(
                request_data["rubrics"],
                subject,
                question_paper
            )

            print("Rubric analyzed successfully.")

        else:

            print("\n[3/5] No rubric supplied.")
            print("AI will generate evaluation criteria.")

        # --------------------------------------------------
        # 4. EVALUATION
        # --------------------------------------------------

        print("\n[4/5] Evaluating answers...")

        evaluation = self.evaluate_answers(
            subject=subject,
            question_paper=question_paper,
            answer_script=answer_script,
            rubrics=rubric_data
        )

        print("AI evaluation completed.")

        # --------------------------------------------------
        # 5. FINAL CALCULATION
        # --------------------------------------------------

        print("\n[5/5] Calculating final marks...")

        final_result = self.calculate_final_result(
            question_paper,
            evaluation
        )

        final_result["student_name"] = str(
            request_data.get("student_name", "")
        ).strip()

        final_result["roll_number"] = str(
            request_data.get("roll_number", "")
        ).strip()

        final_result["subject"] = subject

        print("\n======================================")
        print("EVALUATION COMPLETED")
        print("Marks:",
              final_result["marks_obtained"],
              "/",
              final_result["maximum_marks"])
        print("Percentage:",
              final_result["percentage"])
        print("Grade:",
              final_result["grade"])
        print("======================================\n")

        return final_result

    # ======================================================
    # REQUEST VALIDATION
    # ======================================================

    def _validate_request(
        self,
        request_data: Dict[str, Any]
    ) -> None:

        if not isinstance(request_data, dict):
            raise ValueError(
                "Invalid evaluation request."
            )

        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is not configured. "
                "Add GEMINI_API_KEY to Render Environment Variables."
            )

        required_fields = [
            "subject",
            "question_paper",
            "answer_script"
        ]

        for field in required_fields:

            value = request_data.get(field)

            if not value:
                raise ValueError(
                    f"Required field missing: {field}"
                )

        for field in [
            "question_paper",
            "answer_script"
        ]:

            path = request_data[field]

            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"{field} file not found: {path}"
                )

        # Rubric is optional.
        rubric = request_data.get("rubrics")

        if rubric and not os.path.isfile(rubric):
            raise FileNotFoundError(
                f"Rubrics file not found: {rubric}"
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

TEACHER-SUPPLIED SUBJECT:
{subject}

The uploaded question paper is the ONLY authoritative
source of truth.

Analyze the uploaded paper independently.

Do not use information from previous examinations.

==========================================================
SUBJECT
==========================================================

The teacher supplied subject is:

{subject}

Stay strictly within this subject.

==========================================================
TOTAL MARKS
==========================================================

Extract the actual examination total marks from THIS paper.

Look for:

- Total Marks
- Maximum Marks
- Max Marks
- Total
- section totals
- explicit examination instructions
- marking schemes

Examples:

Total Marks: 20
Maximum Marks: 50

If the paper explicitly says:

Part A = 10
Part B = 20
Part C = 20

then total_marks may be 50.

Do not assume 20, 50, or 100.

==========================================================
QUESTION MAXIMUM MARKS
==========================================================

Extract the maximum marks for EVERY question or subquestion.

Examples:

1. Define operating system. [2]
2. Explain process scheduling. [5]
3. Explain deadlock. [10]

Return:

1 -> 2
2 -> 5
3 -> 10

Look for:

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

Also inspect tables and section schemes.

DO NOT assign default marks.

DO NOT assume equal marks.

DO NOT use marks from previous papers.

==========================================================
QUESTION NUMBERS
==========================================================

Preserve exact question numbers.

Examples:

1
2
3(a)
3(b)
4(i)
4(ii)
10
16

Do not invent question numbers.

==========================================================
QUESTION TEXT
==========================================================

Extract the complete question text as accurately as possible.

Preserve:

- equations
- mathematical symbols
- options
- numerical values
- technical terminology
- diagrams described in text
- instructions

==========================================================
QUESTION TYPE
==========================================================

Classify each question as one of:

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

Keep subquestions separate.

Example:

3(a)
3(b)
3(c)

Do not merge them incorrectly.

==========================================================
CHOICES
==========================================================

Detect instructions such as:

Answer any 5
Answer any 3
Answer either 4(a) or 4(b)
Internal choice
OR

Preserve the choice information.

==========================================================
NO INVENTION
==========================================================

Never invent:

- questions
- question numbers
- marks
- total marks
- sections
- choices

If something is genuinely unreadable,
use null and mark it uncertain.

Do not guess.

==========================================================
SOURCE OF TRUTH
==========================================================

The extracted question paper controls evaluation.

question.maximum_marks is the absolute maximum for
that question.

question_paper.total_marks is the examination maximum.

==========================================================
OUTPUT
==========================================================

Return ONLY valid JSON.

Use exactly this structure:

{{
    "subject": "{subject}",
    "total_marks": null,
    "total_marks_source": "question_paper",
    "questions": [
        {{
            "question_number": "1",
            "question_text": "...",
            "maximum_marks": null,
            "question_type": "MCQ",
            "section": "Part A",
            "subquestions": [],
            "choice_information": null
        }}
    ]
}}

IMPORTANT:

- total_marks must come from THIS uploaded paper.
- maximum_marks must come from THIS uploaded paper.
- Never use hardcoded marks.
- Never use marks from previous evaluations.
- Return ONLY JSON.
"""

    # ======================================================
    # ANALYZE QUESTION PAPER
    # ======================================================

    def analyze_question_paper(
        self,
        file_path: str,
        subject: str
    ) -> Dict[str, Any]:

        response = self._call_gemini(
            self._question_paper_prompt(subject),
            [file_path]
        )

        result = self._parse_json_response(
            response
        )

        return result

    # ======================================================
    # VALIDATE QUESTION PAPER
    # ======================================================

    def _validate_question_paper_structure(
        self,
        question_paper: Dict[str, Any]
    ) -> None:

        if not isinstance(question_paper, dict):
            raise ValueError(
                "Question paper analysis did not return valid data."
            )

        total_marks = question_paper.get(
            "total_marks"
        )

        questions = question_paper.get(
            "questions"
        )

        if total_marks is None:
            raise ValueError(
                "Unable to extract total marks from the question paper."
            )

        try:
            total_marks = float(total_marks)
        except (TypeError, ValueError):
            raise ValueError(
                f"Extracted total marks are invalid: {total_marks}"
            )

        if total_marks <= 0:
            raise ValueError(
                "Question paper total marks must be greater than zero."
            )

        if not isinstance(questions, list) or not questions:
            raise ValueError(
                "No questions could be extracted from the question paper."
            )

        seen_numbers = set()

        for question in questions:

            if not isinstance(question, dict):
                raise ValueError(
                    "Invalid question structure returned by AI."
                )

            q_number = question.get(
                "question_number"
            )

            maximum_marks = question.get(
                "maximum_marks"
            )

            if q_number is None:
                raise ValueError(
                    "A question was extracted without a question number."
                )

            normalized = self._normalize_question_number(
                q_number
            )

            if not normalized:
                raise ValueError(
                    f"Invalid question number: {q_number}"
                )

            if normalized in seen_numbers:
                raise ValueError(
                    f"Duplicate question number detected: {q_number}"
                )

            seen_numbers.add(normalized)

            if maximum_marks is None:
                raise ValueError(
                    f"Maximum marks missing for question {q_number}."
                )

            try:
                marks = float(maximum_marks)
            except (TypeError, ValueError):
                raise ValueError(
                    f"Invalid marks for question "
                    f"{q_number}: {maximum_marks}"
                )

            if marks <= 0:
                raise ValueError(
                    f"Maximum marks must be greater than zero "
                    f"for question {q_number}."
                )

        question_paper["total_marks"] = self._clean_number(
            total_marks
        )

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

SUBJECT:
{subject}

The question paper below is authoritative.

QUESTION PAPER:
{question_structure}

Analyze the uploaded handwritten answer script.

==========================================================
ANSWER ORDER
==========================================================

The student may answer questions in ANY order.

For example:

Q5
Q2
Q10
Q1
Q7

Or:

Q10 may appear before Q3.

Or Part B may appear before Part A.

Never assume page order equals question order.

==========================================================
QUESTION MATCHING
==========================================================

Match each answer using its actual question number.

Every detected answer must correspond to a question in
the supplied question paper.

Do not create new question numbers.

Do not merge unrelated answers.

==========================================================
HANDWRITING
==========================================================

Read handwriting carefully.

Preserve mathematical symbols and equations where possible.

If handwriting is unclear:

- do not invent content
- preserve uncertainty
- lower the confidence score

==========================================================
MISSING ANSWERS
==========================================================

Return only answers that actually appear.

Do not create answers for unanswered questions.

==========================================================
OUTPUT
==========================================================

Return ONLY valid JSON.

Use:

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

        response = self._call_gemini(
            self._answer_script_prompt(
                subject,
                question_paper
            ),
            [file_path]
        )

        result = self._parse_json_response(
            response
        )

        if not isinstance(result, dict):
            raise ValueError(
                "Answer script analysis returned invalid data."
            )

        answers = result.get("answers")

        if answers is None:
            raise ValueError(
                "Answer script analysis did not return an answers list."
            )

        if not isinstance(answers, list):
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

SUBJECT:
{subject}

QUESTION PAPER:
{question_structure}

Analyze the uploaded rubric.

Map rubric criteria to the correct questions.

Do NOT modify question maximum marks.

The question paper is the absolute upper limit.

If the rubric conflicts with the question paper,
the question paper maximum marks win.

Return ONLY valid JSON.

Use:

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

        response = self._call_gemini(
            self._rubric_prompt(
                subject,
                question_paper
            ),
            [file_path]
        )

        return self._parse_json_response(
            response
        )

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

1. Evaluate ONLY questions present in the question paper.

2. Stay strictly within the supplied subject.

3. Never use marks from previous examinations.

4. Use the question paper maximum_marks.

5. awarded_marks must satisfy:

0 <= awarded_marks <= maximum_marks

6. Never exceed maximum_marks.

7. Award partial marks when academically justified.

8. Long answers must consider:

- correctness
- conceptual understanding
- explanation
- relevant points
- examples
- terminology
- completeness
- conclusion where required

9. Numerical problems must consider:

- formula
- substitution
- calculation
- units
- logical steps
- final answer

10. Mathematics questions must receive partial marks
for correct intermediate work when justified.

11. MCQs must be evaluated against the question's
actual options and correct answer.

12. If the student did not answer a question:

awarded_marks = 0

13. Never invent student content.

14. Student answer order must not affect evaluation.

15. Match answers using question number.

16. Use the supplied rubric when available.

17. If there is no rubric, derive academically reasonable
criteria from the actual question.

18. Feedback must correspond to the actual question.

19. Never generate feedback from another subject.

20. Do not calculate the final examination total yourself.
The server will calculate it.

==========================================================
FEEDBACK
==========================================================

For every evaluated question provide:

- question number
- maximum marks
- awarded marks
- status
- what was done well
- mistakes
- missing points
- expected answer
- improvement

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

Do not include markdown.
Do not include explanations outside JSON.
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

        if not isinstance(result, dict):
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

        if not isinstance(evaluations, list):
            raise ValueError(
                "'evaluations' must be a list."
            )

        return result

    # ======================================================
    # FINAL RESULT
    # ======================================================

    def calculate_final_result(
        self,
        question_paper: Dict[str, Any],
        evaluation: Dict[str, Any]
    ) -> Dict[str, Any]:

        questions = question_paper["questions"]

        ai_evaluations = evaluation.get(
            "evaluations",
            []
        )

        evaluation_map = {}

        for item in ai_evaluations:

            if not isinstance(item, dict):
                continue

            q_number = item.get(
                "question_number"
            )

            if q_number is None:
                continue

            normalized = self._normalize_question_number(
                q_number
            )

            if not normalized:
                continue

            evaluation_map[normalized] = item

        final_evaluations = []
        obtained_marks = 0.0

        # Question paper controls the final result.
        for question in questions:

            q_number = str(
                question["question_number"]
            ).strip()

            maximum_marks = float(
                question["maximum_marks"]
            )

            normalized = self._normalize_question_number(
                q_number
            )

            ai_item = evaluation_map.get(
                normalized
            )

            if ai_item is None:

                awarded_marks = 0.0

                final_item = {
                    "question_number": q_number,
                    "maximum_marks": self._clean_number(
                        maximum_marks
                    ),
                    "awarded_marks": 0,
                    "status": "Not Answered",
                    "what_was_done_well": "",
                    "mistakes": [],
                    "missing_points": [
                        "No answer detected for this question."
                    ],
                    "expected_answer": "",
                    "improvement": (
                        "Attempt the question and provide "
                        "the required explanation."
                    )
                }

            else:

                raw_marks = ai_item.get(
                    "awarded_marks",
                    0
                )

                try:
                    awarded_marks = float(
                        raw_marks
                    )
                except (TypeError, ValueError):
                    awarded_marks = 0.0

                # HARD SAFETY LIMIT
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

                final_item["question_number"] = q_number

                # Question paper maximum wins.
                final_item["maximum_marks"] = (
                    self._clean_number(
                        maximum_marks
                    )
                )

                final_item["awarded_marks"] = (
                    self._clean_number(
                        awarded_marks
                    )
                )

            obtained_marks += awarded_marks

            final_evaluations.append(
                final_item
            )

        # --------------------------------------------------
        # TOTAL MARKS
        # --------------------------------------------------

        total_marks = float(
            question_paper["total_marks"]
        )

        # Never exceed exam total.
        obtained_marks = min(
            obtained_marks,
            total_marks
        )

        obtained_marks = max(
            0.0,
            obtained_marks
        )

        # --------------------------------------------------
        # PERCENTAGE
        # --------------------------------------------------

        percentage = 0.0

        if total_marks > 0:
            percentage = (
                obtained_marks /
                total_marks
            ) * 100

        # --------------------------------------------------
        # GRADE
        # --------------------------------------------------

        grade = self._calculate_grade(
            percentage
        )

        return {
            "maximum_marks": self._clean_number(
                total_marks
            ),
            "marks_obtained": self._clean_number(
                obtained_marks
            ),
            "percentage": round(
                percentage,
                2
            ),
            "grade": grade,
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
        question_number: Any
    ) -> str:

        value = str(
            question_number
        ).strip().lower()

        value = re.sub(
            r"^\s*question\s*",
            "",
            value
        )

        value = re.sub(
            r"^\s*q\s*\.?\s*",
            "",
            value
        )

        value = value.strip()

        value = value.rstrip(
            ".:"
        )

        value = re.sub(
            r"\s+",
            "",
            value
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

        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is not configured. "
                "Check Render Environment Variables."
            )

        print("\nCalling Gemini...")
        print("Model:", self.model)
        print("Files:", len(file_paths))

        parts = [
            {
                "text": prompt
            }
        ]

        # --------------------------------------------------
        # ATTACH FILES
        # --------------------------------------------------

        for file_path in file_paths:

            if not file_path:
                continue

            if not os.path.isfile(file_path):
                raise FileNotFoundError(
                    f"File not found: {file_path}"
                )

            mime_type = self._get_mime_type(
                file_path
            )

            file_size = os.path.getsize(
                file_path
            )

            print(
                "Attaching:",
                os.path.basename(file_path),
                "|",
                mime_type,
                "|",
                file_size,
                "bytes"
            )

            if file_size <= 0:
                raise ValueError(
                    f"Uploaded file is empty: {file_path}"
                )

            with open(
                file_path,
                "rb"
            ) as file:

                file_bytes = file.read()

            encoded = base64.b64encode(
                file_bytes
            ).decode("utf-8")

            parts.append(
                {
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": encoded
                    }
                }
            )

        # --------------------------------------------------
        # GEMINI PAYLOAD
        # --------------------------------------------------

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

        # --------------------------------------------------
        # REQUEST
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
                f"Gemini API request timed out after "
                f"{self.timeout} seconds."
            ) from error

        except requests.RequestException as error:

            raise RuntimeError(
                f"Could not connect to Gemini API: {error}"
            ) from error

        print(
            "Gemini HTTP status:",
            response.status_code
        )

        # --------------------------------------------------
        # API ERROR
        # --------------------------------------------------

        if not response.ok:

            try:
                error_data = response.json()
            except ValueError:
                error_data = response.text[:5000]

            raise RuntimeError(
                "Gemini API request failed.\n"
                f"HTTP Status: {response.status_code}\n"
                f"Model: {self.model}\n"
                f"Response: {error_data}"
            )

        # --------------------------------------------------
        # RESPONSE JSON
        # --------------------------------------------------

        try:

            data = response.json()

        except ValueError as error:

            raise RuntimeError(
                "Gemini returned a non-JSON HTTP response.\n"
                f"Response preview: {response.text[:2000]}"
            ) from error

        # --------------------------------------------------
        # CANDIDATES
        # --------------------------------------------------

        candidates = data.get(
            "candidates"
        )

        if not candidates:
            raise RuntimeError(
                "Gemini returned no candidates.\n"
                f"Full response: {json.dumps(data)[:5000]}"
            )

        candidate = candidates[0]

        if not isinstance(candidate, dict):
            raise RuntimeError(
                "Gemini returned an invalid candidate."
            )

        # --------------------------------------------------
        # CHECK FINISH REASON
        # --------------------------------------------------

        finish_reason = candidate.get(
            "finishReason"
        )

        if finish_reason and finish_reason not in {
            "STOP",
            "MAX_TOKENS"
        }:

            raise RuntimeError(
                "Gemini stopped generation unexpectedly.\n"
                f"Finish reason: {finish_reason}\n"
                f"Response: {json.dumps(data)[:5000]}"
            )

        # --------------------------------------------------
        # CONTENT
        # --------------------------------------------------

        content = candidate.get(
            "content"
        )

        if not isinstance(content, dict):
            raise RuntimeError(
                "Gemini candidate has no valid content.\n"
                f"Response: {json.dumps(data)[:5000]}"
            )

        response_parts = content.get(
            "parts"
        )

        if not isinstance(
            response_parts,
            list
        ) or not response_parts:

            raise RuntimeError(
                "Gemini response contains no parts.\n"
                f"Response: {json.dumps(data)[:5000]}"
            )

        text_parts = []

        for part in response_parts:

            if not isinstance(part, dict):
                continue

            text = part.get("text")

            if text:
                text_parts.append(
                    str(text)
                )

        if not text_parts:
            raise RuntimeError(
                "Gemini response contains no text.\n"
                f"Response: {json.dumps(data)[:5000]}"
            )

        return "\n".join(
            text_parts
        ).strip()

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
            r"^\s*```(?:json)?\s*",
            "",
            text,
            flags=re.IGNORECASE
        )

        text = re.sub(
            r"\s*```\s*$",
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
        # Extract JSON object
        # --------------------------------------------------

        start = text.find("{")
        end = text.rfind("}")

        if start != -1 and end > start:

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

                raise ValueError(
                    "AI returned invalid JSON.\n"
                    f"Response preview: {text[:2000]}"
                ) from error

        raise ValueError(
            "AI returned invalid JSON.\n"
            f"Response preview: {text[:2000]}"
        )

    # ======================================================
    # MIME TYPE
    # ======================================================

    def _get_mime_type(
        self,
        file_path: str
    ) -> str:

        extension = os.path.splitext(
            file_path
        )[1].lower()

        mime_types = {
            ".pdf": "application/pdf",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png"
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

        value = float(value)

        if value.is_integer():
            return int(value)

        return round(
            value,
            2
        )
