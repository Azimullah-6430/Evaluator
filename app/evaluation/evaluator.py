import os
import json
import base64
import re
from typing import Dict, Any, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()


class EvaluationAgent:
    """
    Smart Education System - AI Evaluation Agent

    Main principles:
    1. Every question paper is treated as a new examination.
    2. The uploaded question paper is the source of truth.
    3. Total marks are extracted dynamically.
    4. Marks for every question are extracted dynamically.
    5. Answers can appear in any order.
    6. Rubrics are optional.
    7. Marks can never exceed the marks assigned to a question.
    8. Final marks can never exceed the extracted total marks.
    """

    def __init__(self):

        # IMPORTANT:
        # Render Environment Variable must be named GEMINI_API_KEY
        self.api_key = os.getenv("GEMINI_API_KEY")

        self.model = os.getenv(
            "GEMINI_MODEL",
            "gemini-3.4-flash"
        )

        self.api_url = (
            "https://generativelanguage.googleapis.com/"
            f"v1beta/models/{self.model}:generateContent"
        )

        self.timeout = 180

    # ==========================================================
    # MAIN EVALUATION PIPELINE
    # ==========================================================

    def evaluate(self, request_data: Dict[str, Any]) -> Dict[str, Any]:

        self._validate_request(request_data)

        subject = request_data["subject"]

        # ------------------------------------------------------
        # STEP 1
        # Analyze the uploaded question paper.
        # This happens EVERY TIME a new paper is uploaded.
        # ------------------------------------------------------

        question_paper = self.analyze_question_paper(
            request_data["question_paper"],
            subject
        )

        # ------------------------------------------------------
        # STEP 2
        # Validate extracted question paper structure.
        # ------------------------------------------------------

        self._validate_question_paper_structure(
            question_paper
        )

        # ------------------------------------------------------
        # STEP 3
        # Analyze handwritten answer script.
        # Answers may be written in ANY ORDER.
        # ------------------------------------------------------

        answer_script = self.analyze_answer_script(
            request_data["answer_script"],
            subject,
            question_paper
        )

        # ------------------------------------------------------
        # STEP 4
        # Optional rubrics
        # ------------------------------------------------------

        rubric_data = None

        if request_data.get("rubrics"):

            rubric_data = self.analyze_rubrics(
                request_data["rubrics"],
                subject,
                question_paper
            )

        # ------------------------------------------------------
        # STEP 5
        # Evaluate every detected answer.
        # ------------------------------------------------------

        evaluation = self.evaluate_answers(
            subject=subject,
            question_paper=question_paper,
            answer_script=answer_script,
            rubrics=rubric_data
        )

        # ------------------------------------------------------
        # STEP 6
        # Recalculate marks safely on the server.
        # Do NOT trust AI's total blindly.
        # ------------------------------------------------------

        final_result = self.calculate_final_result(
            question_paper,
            evaluation
        )

        # ------------------------------------------------------
        # STEP 7
        # Student information
        # ------------------------------------------------------

        final_result["student_name"] = request_data.get(
            "student_name",
            ""
        )

        final_result["roll_number"] = request_data.get(
            "roll_number",
            ""
        )

        final_result["subject"] = subject

        return final_result

    # ==========================================================
    # REQUEST VALIDATION
    # ==========================================================

    def _validate_request(
        self,
        request_data: Dict[str, Any]
    ):

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

    # ==========================================================
    # QUESTION PAPER PROMPT
    # ==========================================================

    def _question_paper_prompt(
        self,
        subject: str
    ) -> str:

        return f"""
You are the Question Paper Analysis Engine of a
high-accuracy academic evaluation system.

The user supplied subject is:

{subject}

The uploaded question paper is the ONLY authoritative
source of truth for the examination structure.

This question paper may be completely different from
previous question papers.

You MUST analyze THIS uploaded question paper independently.

==========================================================
CRITICAL RULE 1 — SUBJECT
==========================================================

The subject is:

{subject}

Stay strictly within this subject.

Do not introduce content from another subject.

==========================================================
CRITICAL RULE 2 — TOTAL MARKS
==========================================================

Extract the actual total marks from the uploaded
question paper.

NEVER assume:

20 marks
50 marks
100 marks

The total marks may be any value.

Examples:

If the paper says:

Total Marks: 20

return:

"total_marks": 20

If the paper says:

Maximum Marks: 50

return:

"total_marks": 50

If the paper uses section totals such as:

Part A = 10
Part B = 20
Part C = 20

then calculate:

total_marks = 50

ONLY when the structure clearly indicates those marks
are part of the examination total.

==========================================================
CRITICAL RULE 3 — EACH QUESTION'S MARKS
==========================================================

Extract the maximum marks assigned to EVERY question.

For example:

1. Define OS.                         [2]
2. Explain process scheduling.       [5]
3. Explain deadlock with example.    [10]

Return:

Question 1 → 2 marks
Question 2 → 5 marks
Question 3 → 10 marks

DO NOT assign default marks.

DO NOT assume every question has the same marks.

==========================================================
CRITICAL RULE 4 — QUESTION NUMBERS
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
10
16

Do not invent question numbers.

==========================================================
CRITICAL RULE 5 — QUESTION TEXT
==========================================================

Extract the complete question text as accurately
as possible.

Preserve important mathematical symbols,
equations, options and technical terminology.

==========================================================
CRITICAL RULE 6 — QUESTION TYPE
==========================================================

Identify the type.

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
CRITICAL RULE 7 — SECTIONS
==========================================================

Identify sections such as:

Part A
Part B
Section I
Section II
Section III

==========================================================
CRITICAL RULE 8 — SUBQUESTIONS
==========================================================

Preserve subquestions separately.

For example:

3(a)
3(b)
3(c)

must not be merged incorrectly.

==========================================================
CRITICAL RULE 9 — CHOICE QUESTIONS
==========================================================

Detect structures such as:

Answer any 5
Answer any 3
Attempt either 4(a) or 4(b)
OR
Internal choice

Preserve the choice information.

==========================================================
CRITICAL RULE 10 — NO INVENTION
==========================================================

Never invent:

- questions
- marks
- sections
- question numbers
- choices
- total marks

If something cannot be read confidently,
mark it as uncertain instead of guessing.

==========================================================
CRITICAL RULE 11 — SOURCE OF TRUTH
==========================================================

The extracted question-paper structure will control
the entire evaluation.

The answer evaluator MUST use:

question.maximum_marks

as the absolute maximum marks for that question.

The final evaluator MUST use:

question_paper.total_marks

as the absolute examination total.

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

Do not use marks from previous evaluations.

Do not use hardcoded marks.

Return ONLY JSON.
"""

    # ==========================================================
    # ANALYZE QUESTION PAPER
    # ==========================================================

    def analyze_question_paper(
        self,
        file_path: str,
        subject: str
    ) -> Dict[str, Any]:

        prompt = self._question_paper_prompt(subject)

        response = self._call_gemini(
            prompt,
            [file_path]
        )

        result = self._parse_json_response(response)

        return result

    # ==========================================================
    # QUESTION PAPER VALIDATION
    # ==========================================================

    def _validate_question_paper_structure(
        self,
        question_paper: Dict[str, Any]
    ):

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

        except Exception:

            raise ValueError(
                "Extracted total marks are invalid."
            )

        if total_marks <= 0:

            raise ValueError(
                "Question paper total marks must be greater than zero."
            )

        if not isinstance(questions, list) or not questions:

            raise ValueError(
                "No questions could be extracted from the question paper."
            )

        # Validate every question

        for question in questions:

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

            if maximum_marks is None:

                raise ValueError(
                    f"Maximum marks missing for question "
                    f"{question_number}."
                )

            try:

                marks = float(maximum_marks)

            except Exception:

                raise ValueError(
                    f"Invalid marks for question "
                    f"{question_number}."
                )

            if marks < 0:

                raise ValueError(
                    f"Negative marks found for question "
                    f"{question_number}."
                )

        question_paper["total_marks"] = total_marks

    # ==========================================================
    # ANSWER SCRIPT PROMPT
    # ==========================================================

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

The question paper below is the authoritative
examination structure.

QUESTION PAPER:

{question_structure}

Your task is to analyze the uploaded handwritten
answer script.

==========================================================
IMPORTANT — ANSWERS MAY BE IN ANY ORDER
==========================================================

The student may answer questions in ANY order.

Examples:

Question 5
Question 2
Question 10
Question 1
Question 7

OR:

Q10 appears before Q3.

OR:

Part B is answered before Part A.

Do NOT assume page order equals question order.

Identify each answer using its actual question number.

==========================================================
IMPORTANT — DO NOT INVENT ANSWERS
==========================================================

Only extract answers that actually appear
in the handwritten script.

If an answer is unclear because of handwriting,
preserve the uncertainty.

==========================================================
IMPORTANT — MATCH AGAINST QUESTION PAPER
==========================================================

Every detected answer must be matched to the
corresponding question number from the question paper.

Do not create a new question number.

==========================================================
OUTPUT
==========================================================

Return ONLY JSON:

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

    # ==========================================================
    # ANALYZE ANSWER SCRIPT
    # ==========================================================

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

        return self._parse_json_response(
            response
        )

    # ==========================================================
    # RUBRIC PROMPT
    # ==========================================================

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

Map rubric criteria to the questions where possible.

Do not change the maximum marks obtained
from the question paper.

If the rubric conflicts with the question paper's
maximum marks, the question paper maximum marks
must remain the hard upper limit.

Return ONLY JSON.

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

    # ==========================================================
    # ANALYZE RUBRICS
    # ==========================================================

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

        return self._parse_json_response(
            response
        )

    # ==========================================================
    # EVALUATION PROMPT
    # ==========================================================

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

        rubric_json = json.dumps(
            rubrics,
            indent=2,
            ensure_ascii=False
        ) if rubrics else "NO RUBRIC PROVIDED"

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
CRITICAL EVALUATION RULES
==========================================================

1. Evaluate ONLY according to the supplied question paper.

2. Never evaluate using a different subject.

3. Never use marks from previous papers.

4. Every question has its own maximum_marks.

5. Award:

0 <= awarded_marks <= maximum_marks

6. Never exceed the maximum marks.

7. Partial marks must be awarded when appropriate.

8. For long-answer questions, evaluate:
   - correctness
   - concepts
   - explanation
   - relevant points
   - examples
   - calculations
   - derivations
   - conclusion
   - presentation where academically relevant

9. For numerical questions:
   - evaluate formula
   - substitution
   - calculation
   - units
   - final answer
   - logical steps

10. For mathematical problems:
    correct intermediate steps may receive partial marks.

11. For MCQs:
    compare the student's selected option with
    the correct answer.

12. If an answer is missing:
    awarded_marks = 0

13. If the question is unanswered, do not invent an answer.

14. If the student answered questions in a different order,
    evaluate according to QUESTION NUMBER.

15. Do not penalize the student merely because
    answers are not written sequentially.

16. If rubrics are available, use them.

17. If rubrics are NOT available, generate appropriate
    evaluation criteria from the question itself.

18. Feedback must be specific to the actual question.

19. Never generate feedback from another subject.

20. Do not blindly trust an answer-script total.

==========================================================
FEEDBACK
==========================================================

For every question provide:

- question number
- maximum marks
- awarded marks
- status
- what the student did well
- mistakes
- missing points
- expected answer
- improvement suggestion

==========================================================
OUTPUT
==========================================================

Return ONLY JSON.

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

Remember:

AWARDED MARKS MUST NEVER EXCEED
THE QUESTION'S maximum_marks.
"""

    # ==========================================================
    # EVALUATE ANSWERS
    # ==========================================================

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

        return result

    # ==========================================================
    # FINAL RESULT CALCULATION
    # ==========================================================

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

        # Create lookup by normalized question number

        evaluation_map = {}

        for item in ai_evaluations:

            q_number = str(
                item.get("question_number", "")
            ).strip()

            evaluation_map[
                self._normalize_question_number(q_number)
            ] = item

        final_evaluations = []

        obtained_marks = 0.0

        # ------------------------------------------------------
        # IMPORTANT:
        # Iterate through QUESTION PAPER questions.
        # Therefore the question paper controls:
        #
        # - which questions exist
        # - maximum marks
        # - total marks
        # ------------------------------------------------------

        for question in questions:

            q_number = str(
                question["question_number"]
            ).strip()

            maximum_marks = float(
                question["maximum_marks"]
            )

            normalized_number = (
                self._normalize_question_number(
                    q_number
                )
            )

            ai_item = evaluation_map.get(
                normalized_number
            )

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

                final_item[
                    "question_number"
                ] = q_number

                final_item[
                    "maximum_marks"
                ] = maximum_marks

                final_item[
                    "awarded_marks"
                ] = awarded_marks

            else:

                # Question exists in paper but
                # no corresponding answer was detected.

                awarded_marks = 0.0

                final_item = {
                    "question_number": q_number,
                    "maximum_marks": maximum_marks,
                    "awarded_marks": 0.0,
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

            obtained_marks += awarded_marks

            final_evaluations.append(
                final_item
            )

        # ------------------------------------------------------
        # QUESTION PAPER TOTAL — SOURCE OF TRUTH
        # ------------------------------------------------------

        total_marks = float(
            question_paper["total_marks"]
        )

        # ------------------------------------------------------
        # SECOND SAFETY LIMIT
        # ------------------------------------------------------

        obtained_marks = min(
            obtained_marks,
            total_marks
        )

        # ------------------------------------------------------
        # Percentage
        # ------------------------------------------------------

        percentage = (
            obtained_marks / total_marks
        ) * 100

        # ------------------------------------------------------
        # Grade
        # ------------------------------------------------------

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

    # ==========================================================
    # GRADE CALCULATION
    # ==========================================================

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

    # ==========================================================
    # QUESTION NUMBER NORMALIZATION
    # ==========================================================

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

        return value

    # ==========================================================
    # GEMINI API CALL
    # ==========================================================

    def _call_gemini(
        self,
        prompt: str,
        file_paths: List[str]
    ) -> str:

        if not self.api_key:

            raise ValueError(
                "GEMINI_API_KEY is not configured."
            )

        parts = [
            {
                "text": prompt
            }
        ]

        for file_path in file_paths:

            if not file_path:
                continue

            if not os.path.exists(file_path):

                raise FileNotFoundError(
                    f"File not found: {file_path}"
                )

            mime_type = self._get_mime_type(
                file_path
            )

            with open(
                file_path,
                "rb"
            ) as file:

                encoded = base64.b64encode(
                    file.read()
                ).decode("utf-8")

            parts.append(
                {
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": encoded
                    }
                }
            )

        payload = {
            "contents": [
                {
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
            timeout=self.timeout
        )

        if not response.ok:

            try:
                error_data = response.json()

            except Exception:

                error_data = response.text

            raise RuntimeError(
                f"Gemini API error "
                f"{response.status_code}: "
                f"{error_data}"
            )

        data = response.json()

        try:

            return (
                data["candidates"][0]
                ["content"]["parts"][0]
                ["text"]
            )

        except (
            KeyError,
            IndexError,
            TypeError
        ):

            raise RuntimeError(
                "Gemini returned an unexpected response."
            )

    # ==========================================================
    # JSON PARSER
    # ==========================================================

    def _parse_json_response(
        self,
        response: str
    ) -> Dict[str, Any]:

        if not response:

            raise ValueError(
                "AI returned an empty response."
            )

        text = response.strip()

        # Remove markdown code fences if Gemini
        # returns them despite instructions.

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

        try:

            return json.loads(
                text
            )

        except json.JSONDecodeError:

            # Attempt to locate JSON object

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

    # ==========================================================
    # MIME TYPE
    # ==========================================================

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

        return mime_types.get(
            extension,
            "application/octet-stream"
        )

    # ==========================================================
    # CLEAN NUMBER
    # ==========================================================

    def _clean_number(
        self,
        value: float
    ):

        if float(value).is_integer():

            return int(value)

        return round(
            float(value),
            2
        )
