import os
import json
import base64
import random
import time
import traceback
import re

import requests


class EvaluationAgent:
    """
    AI-based handwritten answer script evaluation agent.

    The question paper is always treated as the source of truth
    for total marks and maximum marks per question.
    """

    def __init__(self):
        # ---------------------------------------------------------
        # GEMINI CONFIGURATION
        # ---------------------------------------------------------
        self.api_key = os.getenv("GEMINI_API_KEY")

        # Keep model hardcoded.
        # Render only needs GEMINI_API_KEY.
        self.model = "gemini-3.6-flash"

        self.api_url = (
            "https://generativelanguage.googleapis.com/"
            f"v1beta/models/{self.model}:generateContent"
        )

        self.timeout = 180

        print("=" * 70)
        print("EVALUATION AGENT INITIALIZED")
        print(f"Model: {self.model}")
        print(
            "Gemini API Key: "
            + ("CONFIGURED" if self.api_key else "NOT CONFIGURED")
        )
        print("=" * 70)

    # =========================================================
    # MAIN EVALUATION PIPELINE
    # =========================================================

    def evaluate(self, request):
        """
        Main evaluation pipeline.
        """

        print("=" * 70)
        print("EVALUATION AGENT STARTED")
        print("=" * 70)

        try:
            # -------------------------------------------------
            # STEP 1 - VALIDATE REQUEST
            # -------------------------------------------------
            print("[1/5] Validating request...")
            self._validate_request(request)

            # -------------------------------------------------
            # STEP 2 - ANALYZE QUESTION PAPER
            # -------------------------------------------------
            print("[2/5] Analyzing question paper...")

            question_paper = self.analyze_question_paper(
                request["question_paper"]
            )

            self._validate_question_paper_structure(
                question_paper
            )

            print(
                f"Question paper total marks: "
                f"{question_paper['total_marks']}"
            )

            print(
                f"Questions detected: "
                f"{len(question_paper['questions'])}"
            )

            # -------------------------------------------------
            # STEP 3 - ANALYZE ANSWER SCRIPT
            # -------------------------------------------------
            print("[3/5] Analyzing answer script...")

            answer_script = self.analyze_answer_script(
                request["answer_script"],
                question_paper
            )

            # -------------------------------------------------
            # STEP 4 - RUBRICS
            # -------------------------------------------------
            print("[4/5] Processing rubrics...")

            rubric = None

            if request.get("rubrics"):
                print("Rubrics uploaded.")
                rubric = self.analyze_rubrics(
                    request["rubrics"],
                    question_paper
                )
            else:
                print(
                    "No rubrics uploaded. "
                    "AI will generate evaluation criteria."
                )

            # -------------------------------------------------
            # STEP 5 - EVALUATE ANSWERS
            # -------------------------------------------------
            print("[5/5] Evaluating answers...")

            evaluations = self.evaluate_answers(
                question_paper=question_paper,
                answer_script=answer_script,
                rubric=rubric,
                subject=request["subject"]
            )

            # -------------------------------------------------
            # FINAL RESULT
            # -------------------------------------------------
            final_result = self.calculate_final_result(
                question_paper=question_paper,
                evaluations=evaluations
            )

            # Student information
            final_result["student"] = {
                "name": request.get("student_name", ""),
                "roll_number": request.get("roll_number", ""),
                "subject": request.get("subject", "")
            }

            final_result["question_paper"] = question_paper
            final_result["answer_analysis"] = answer_script

            print("=" * 70)
            print("EVALUATION COMPLETED")
            print(
                f"Final Marks: "
                f"{final_result['obtained_marks']}/"
                f"{final_result['total_marks']}"
            )
            print("=" * 70)

            return final_result

        except Exception as error:

            print("=" * 70)
            print("EVALUATION ERROR")
            print(f"Error: {error}")
            print("FULL TRACEBACK:")
            traceback.print_exc()
            print("=" * 70)

            raise RuntimeError(
                f"Evaluation failed: {str(error)}"
            ) from error

    # =========================================================
    # REQUEST VALIDATION
    # =========================================================

    def _validate_request(self, request):

        if not self.api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not configured. "
                "Add GEMINI_API_KEY to Render Environment Variables."
            )

        if not request:
            raise ValueError(
                "Evaluation request is empty."
            )

        required_fields = [
            "question_paper",
            "answer_script",
            "subject"
        ]

        for field in required_fields:

            if not request.get(field):
                raise ValueError(
                    f"Missing required field: {field}"
                )

        if not os.path.exists(
            request["question_paper"]["path"]
        ):
            raise FileNotFoundError(
                "Question paper file not found."
            )

        if not os.path.exists(
            request["answer_script"]["path"]
        ):
            raise FileNotFoundError(
                "Answer script file not found."
            )

    # =========================================================
    # QUESTION PAPER ANALYSIS
    # =========================================================

    def analyze_question_paper(self, question_paper):

        prompt = """
You are an expert examination-paper analysis agent.

Analyze the uploaded QUESTION PAPER carefully.

IMPORTANT RULE:

The uploaded question paper is the ONLY source of truth
for the examination structure and marks.

DO NOT assume:
- 100 marks
- 50 marks
- 20 marks
- 10 marks per question
- 5 marks per question
- any fixed examination pattern

Extract the actual structure from the uploaded paper.

You MUST identify:

1. Subject
2. Total examination marks
3. All questions
4. Question numbers
5. Maximum marks for every question
6. Sections
7. Subquestions
8. Question type
9. MCQ options
10. Internal choices
11. "Answer any N" instructions
12. Numerical questions
13. Long-answer questions
14. Short-answer questions
15. Marks associated with each subquestion

If marks are explicitly printed, use them.

If a question contains subquestions such as:

1(a) - 2 marks
1(b) - 3 marks

represent them separately.

If a section says:

Answer any 2 questions.
Each question carries 10 marks.

record the choice information.

Do NOT invent marks.

Return ONLY valid JSON.

Required format:

{
    "subject": "",
    "total_marks": 0,
    "duration": "",
    "instructions": [],
    "sections": [],
    "questions": [
        {
            "question_number": "1",
            "question_text": "",
            "maximum_marks": 0,
            "question_type": "",
            "section": "",
            "is_optional": false,
            "choice_group": "",
            "subquestions": [],
            "options": []
        }
    ]
}

For subquestions:

"subquestions": [
    {
        "question_number": "1(a)",
        "question_text": "",
        "maximum_marks": 0,
        "question_type": ""
    }
]

Be extremely careful with marks.
"""

        response = self._call_gemini(
            prompt,
            files=[question_paper]
        )

        return self._parse_json_response(
            response,
            "question paper analysis"
        )

    # =========================================================
    # QUESTION PAPER VALIDATION
    # =========================================================

    def _validate_question_paper_structure(
        self,
        question_paper
    ):

        if not isinstance(question_paper, dict):
            raise ValueError(
                "Question paper analysis is not a JSON object."
            )

        total_marks = question_paper.get("total_marks")

        if total_marks is None:
            raise ValueError(
                "Total marks missing from question paper."
            )

        try:
            total_marks = float(total_marks)
        except (TypeError, ValueError):
            raise ValueError(
                "Total marks extracted from question paper "
                "is not numeric."
            )

        if total_marks <= 0:
            raise ValueError(
                "Invalid total marks extracted from question paper."
            )

        questions = question_paper.get("questions")

        if not isinstance(questions, list):
            raise ValueError(
                "Questions missing from question paper analysis."
            )

        if len(questions) == 0:
            raise ValueError(
                "No questions detected in question paper."
            )

        seen = set()

        for question in questions:

            qno = str(
                question.get("question_number", "")
            ).strip()

            if not qno:
                raise ValueError(
                    "A question is missing its question number."
                )

            normalized = self._normalize_question_number(qno)

            if normalized in seen:
                raise ValueError(
                    f"Duplicate question number detected: {qno}"
                )

            seen.add(normalized)

            marks = question.get("maximum_marks")

            if marks is None:
                raise ValueError(
                    f"Maximum marks missing for question {qno}."
                )

            try:
                marks = float(marks)
            except (TypeError, ValueError):
                raise ValueError(
                    f"Invalid maximum marks for question {qno}."
                )

            if marks <= 0:
                raise ValueError(
                    f"Maximum marks must be greater than 0 "
                    f"for question {qno}."
                )

    # =========================================================
    # ANSWER SCRIPT ANALYSIS
    # =========================================================

    def analyze_answer_script(
        self,
        answer_script,
        question_paper
    ):

        prompt = f"""
You are an expert handwritten answer-sheet analysis agent.

Analyze the uploaded student's ANSWER SCRIPT.

The question paper structure below is the source of truth:

{json.dumps(question_paper, indent=2)}

IMPORTANT:

The student may answer questions in ANY ORDER.

Example:

Page 1:
Question 10

Page 2:
Question 5

Page 3:
Question 2

This is completely valid.

DO NOT assume page order equals question order.

You MUST detect the actual question number written by
the student.

Identify:

- question number
- subquestion number
- student's answer
- whether answer is present
- whether answer is blank
- MCQ selected option
- numerical answer
- mathematical expressions
- derivations
- diagrams
- partially visible answers
- crossed-out answers
- multiple attempts

Match each answer to the corresponding question
number from the question paper.

DO NOT create answers that are not present.

DO NOT move an answer to another question simply
because of page order.

Return ONLY valid JSON.

Format:

{{
    "answers": [
        {{
            "question_number": "1",
            "answer_present": true,
            "answer_text": "",
            "selected_option": "",
            "answer_type": "",
            "confidence": 0.0,
            "notes": ""
        }}
    ]
}}
"""

        response = self._call_gemini(
            prompt,
            files=[answer_script]
        )

        return self._parse_json_response(
            response,
            "answer script analysis"
        )

    # =========================================================
    # RUBRIC ANALYSIS
    # =========================================================

    def analyze_rubrics(
        self,
        rubrics,
        question_paper
    ):

        prompt = f"""
You are an examination rubric analysis agent.

Analyze the uploaded rubric.

Question paper:

{json.dumps(question_paper, indent=2)}

Extract:

- question-wise criteria
- required concepts
- expected answer points
- mark allocation
- partial-mark rules
- numerical evaluation rules
- MCQ evaluation rules
- long-answer evaluation criteria

The rubric is optional.

If the rubric contains information that conflicts with
the question paper's maximum marks, the question paper
maximum marks must be respected.

Return ONLY valid JSON.
"""

        response = self._call_gemini(
            prompt,
            files=[rubrics]
        )

        return self._parse_json_response(
            response,
            "rubric analysis"
        )

    # =========================================================
    # ANSWER EVALUATION
    # =========================================================

    def evaluate_answers(
        self,
        question_paper,
        answer_script,
        rubric,
        subject
    ):

        rubric_text = (
            json.dumps(rubric, indent=2)
            if rubric
            else "NO RUBRIC PROVIDED. Generate suitable evaluation criteria."
        )

        prompt = f"""
You are a highly accurate university examination evaluator.

SUBJECT:
{subject}

QUESTION PAPER:

{json.dumps(question_paper, indent=2)}

STUDENT ANSWERS:

{json.dumps(answer_script, indent=2)}

RUBRIC:

{rubric_text}

============================================================
CORE RULES
============================================================

1. The QUESTION PAPER is the source of truth.

2. Never change the maximum marks of a question.

3. Never assume a fixed total such as 20, 50 or 100.

4. Use the exact marks extracted from the question paper.

5. Evaluate each question independently.

6. The student may answer questions in ANY ORDER.

7. Match answers using QUESTION NUMBER, not page order.

8. Do not confuse answers from different questions.

9. Do not invent content that the student did not write.

10. If an answer is missing, award 0 unless the paper's
    structure requires a different treatment.

============================================================
QUESTION TYPES
============================================================

MCQ:
- Check selected option against the correct answer.
- Correct = full marks.
- Incorrect = 0.
- Do not award partial marks unless the question paper
  explicitly allows it.

SHORT ANSWER:
Evaluate:
- correctness
- relevance
- required concepts
- factual accuracy

LONG ANSWER:
Evaluate:
- conceptual correctness
- completeness
- important points
- explanation
- examples
- derivation where required
- diagrams where required
- conclusion where appropriate

NUMERICAL:
Check:
- formula
- substitution
- calculation
- units
- final answer

If the method is correct but arithmetic has a small error,
award appropriate partial marks.

MATHEMATICS:
Check:
- approach
- formulas
- intermediate calculations
- logical steps
- final answer

DERIVATION:
Check every meaningful mathematical step.

============================================================
HANDWRITING
============================================================

The uploaded answer script may be handwritten.

Do not penalize the student merely because handwriting
style is different.

If text is readable, evaluate it normally.

If something is genuinely unreadable, do not invent it.

============================================================
PARTIAL MARKS
============================================================

Award partial marks when the student demonstrates
partial understanding.

Never exceed the question's maximum marks.

============================================================
FEEDBACK
============================================================

For every attempted question provide:

- awarded marks
- maximum marks
- correctness
- what was done well
- missing points
- what should have been written
- improvement suggestion

============================================================
OPTIONAL QUESTIONS
============================================================

Respect the question paper's instructions such as:

"Answer any 2"

"Attempt any 3"

"Answer any one"

Do not automatically treat every optional question as
a required question.

Identify choice groups carefully.

============================================================
OUTPUT
============================================================

Return ONLY valid JSON.

Use this exact structure:

{{
    "evaluations": [
        {{
            "question_number": "1",
            "answer_present": true,
            "maximum_marks": 10,
            "awarded_marks": 8,
            "question_type": "long_answer",
            "correct": true,
            "confidence": 0.95,
            "feedback": {{
                "what_was_done_well": [],
                "missing_points": [],
                "expected_answer": "",
                "improvement": ""
            }}
        }}
    ]
}}

IMPORTANT:

awarded_marks MUST be between:

0 <= awarded_marks <= maximum_marks

Do not use marks from another question.

Do not change maximum_marks.

Return JSON only.
"""

        response = self._call_gemini(
            prompt
        )

        return self._parse_json_response(
            response,
            "answer evaluation"
        )

    # =========================================================
    # FINAL SCORE CALCULATION
    # =========================================================

    def calculate_final_result(
        self,
        question_paper,
        evaluations
    ):

        total_marks = float(
            question_paper["total_marks"]
        )

        question_list = question_paper["questions"]

        evaluation_list = evaluations.get(
            "evaluations",
            []
        )

        evaluation_map = {}

        for evaluation in evaluation_list:

            qno = str(
                evaluation.get(
                    "question_number",
                    ""
                )
            ).strip()

            normalized = self._normalize_question_number(
                qno
            )

            if normalized:
                evaluation_map[normalized] = evaluation

        final_evaluations = []

        obtained_marks = 0.0

        for question in question_list:

            qno = str(
                question["question_number"]
            ).strip()

            normalized = self._normalize_question_number(
                qno
            )

            maximum_marks = float(
                question["maximum_marks"]
            )

            evaluation = evaluation_map.get(
                normalized
            )

            if evaluation:

                try:
                    awarded_marks = float(
                        evaluation.get(
                            "awarded_marks",
                            0
                        )
                    )
                except (TypeError, ValueError):
                    awarded_marks = 0.0

                # SAFETY CLAMP
                awarded_marks = max(
                    0.0,
                    min(
                        awarded_marks,
                        maximum_marks
                    )
                )

                evaluation["awarded_marks"] = (
                    awarded_marks
                )

                evaluation["maximum_marks"] = (
                    maximum_marks
                )

                obtained_marks += awarded_marks

                final_evaluations.append(
                    evaluation
                )

            else:

                final_evaluations.append(
                    {
                        "question_number": qno,
                        "answer_present": False,
                        "maximum_marks": maximum_marks,
                        "awarded_marks": 0,
                        "question_type": question.get(
                            "question_type",
                            ""
                        ),
                        "correct": False,
                        "confidence": 1.0,
                        "feedback": {
                            "what_was_done_well": [],
                            "missing_points": [
                                "No answer detected."
                            ],
                            "expected_answer": "",
                            "improvement": (
                                "Attempt the question "
                                "with relevant concepts "
                                "and supporting details."
                            )
                        }
                    }
                )

        # Never allow obtained marks above exam total.
        obtained_marks = min(
            obtained_marks,
            total_marks
        )

        percentage = (
            obtained_marks / total_marks
        ) * 100

        percentage = round(
            percentage,
            2
        )

        obtained_marks = round(
            obtained_marks,
            2
        )

        grade = self._calculate_grade(
            percentage
        )

        return {
            "obtained_marks": obtained_marks,
            "total_marks": total_marks,
            "percentage": percentage,
            "grade": grade,
            "evaluations": final_evaluations
        }

    # =========================================================
    # GRADE
    # =========================================================

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

    # =========================================================
    # QUESTION NUMBER NORMALIZATION
    # =========================================================

    def _normalize_question_number(self, value):

        if value is None:
            return ""

        value = str(value).strip().lower()

        value = value.replace(
            "question",
            ""
        )

        value = re.sub(
            r"^q\s*\.?\s*",
            "",
            value
        )

        value = value.replace(
            " ",
            ""
        )

        return value

    # =========================================================
    # GEMINI API CALL
    # =========================================================

    def _call_gemini(
        self,
        prompt,
        files=None
    ):
        """
        Calls Gemini REST API.

        Automatically retries temporary:
        429
        500
        502
        503
        504

        errors using exponential backoff.
        """

        if not self.api_key:

            raise RuntimeError(
                "GEMINI_API_KEY is not configured. "
                "Add GEMINI_API_KEY to Render Environment Variables."
            )

        model = "gemini-3.6-flash"

        api_url = (
            "https://generativelanguage.googleapis.com/"
            f"v1beta/models/{model}:generateContent"
        )

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key
        }

        parts = [
            {
                "text": prompt
            }
        ]

        # -----------------------------------------------------
        # ATTACH FILES
        # -----------------------------------------------------

        if files:

            for file_info in files:

                file_path = file_info.get(
                    "path"
                )

                mime_type = file_info.get(
                    "mime_type"
                )

                if not file_path:

                    raise RuntimeError(
                        "File path missing."
                    )

                if not os.path.exists(
                    file_path
                ):

                    raise FileNotFoundError(
                        f"File not found: {file_path}"
                    )

                with open(
                    file_path,
                    "rb"
                ) as file:

                    encoded_file = (
                        base64.b64encode(
                            file.read()
                        )
                        .decode("utf-8")
                    )

                parts.append(
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": encoded_file
                        }
                    }
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

        # -----------------------------------------------------
        # RETRY CONFIGURATION
        # -----------------------------------------------------

        max_retries = 6

        retry_delays = [
            5,
            10,
            20,
            40,
            60
        ]

        temporary_status_codes = {
            429,
            500,
            502,
            503,
            504
        }

        # -----------------------------------------------------
        # API REQUEST LOOP
        # -----------------------------------------------------

        for attempt in range(
            max_retries
        ):

            try:

                print(
                    f"Calling Gemini API "
                    f"(attempt "
                    f"{attempt + 1}/"
                    f"{max_retries})..."
                )

                response = requests.post(
                    api_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )

                print(
                    f"Gemini HTTP status: "
                    f"{response.status_code}"
                )

                # =================================================
                # SUCCESS
                # =================================================

                if response.status_code == 200:

                    try:

                        data = response.json()

                    except ValueError:

                        raise RuntimeError(
                            "Gemini returned HTTP 200 "
                            "but the response was not valid JSON."
                        )

                    candidates = data.get(
                        "candidates",
                        []
                    )

                    if not candidates:

                        raise RuntimeError(
                            "Gemini returned no candidates."
                        )

                    candidate = candidates[0]

                    finish_reason = candidate.get(
                        "finishReason"
                    )

                    if finish_reason == "SAFETY":

                        raise RuntimeError(
                            "Gemini blocked the request "
                            "for safety reasons."
                        )

                    if finish_reason == "MAX_TOKENS":

                        raise RuntimeError(
                            "Gemini response was truncated "
                            "because it reached the output token limit."
                        )

                    content = candidate.get(
                        "content",
                        {}
                    )

                    response_parts = content.get(
                        "parts",
                        []
                    )

                    text_parts = []

                    for part in response_parts:

                        text = part.get(
                            "text"
                        )

                        if text:
                            text_parts.append(
                                text
                            )

                    if not text_parts:

                        raise RuntimeError(
                            "Gemini returned an empty response."
                        )

                    return "\n".join(
                        text_parts
                    )

                # =================================================
                # TEMPORARY ERROR
                # =================================================

                if response.status_code in temporary_status_codes:

                    try:

                        error_data = response.json()

                    except ValueError:

                        error_data = response.text

                    print(
                        f"Temporary Gemini error "
                        f"{response.status_code}: "
                        f"{error_data}"
                    )

                    if attempt == max_retries - 1:

                        raise RuntimeError(
                            "Gemini API is temporarily "
                            "unavailable after "
                            f"{max_retries} attempts. "
                            f"HTTP Status: "
                            f"{response.status_code}. "
                            "Please try the evaluation again."
                        )

                    delay = retry_delays[
                        attempt
                    ]

                    jitter = random.uniform(
                        0,
                        2
                    )

                    total_delay = (
                        delay + jitter
                    )

                    print(
                        f"Retrying Gemini in "
                        f"{total_delay:.1f} seconds..."
                    )

                    time.sleep(
                        total_delay
                    )

                    continue

                # =================================================
                # PERMANENT ERROR
                # =================================================

                try:

                    error_data = response.json()

                except ValueError:

                    error_data = response.text

                raise RuntimeError(
                    "Gemini API request failed. "
                    f"HTTP Status: "
                    f"{response.status_code} "
                    f"Model: {model} "
                    f"Response: {error_data}"
                )

            # =====================================================
            # TIMEOUT
            # =====================================================

            except requests.exceptions.Timeout:

                print(
                    "Gemini request timed out "
                    f"(attempt "
                    f"{attempt + 1}/"
                    f"{max_retries})."
                )

                if attempt == max_retries - 1:

                    raise RuntimeError(
                        "Gemini API request timed out "
                        f"after {max_retries} attempts."
                    )

                delay = retry_delays[
                    attempt
                ]

                jitter = random.uniform(
                    0,
                    2
                )

                time.sleep(
                    delay + jitter
                )

            # =====================================================
            # CONNECTION ERROR
            # =====================================================

            except requests.exceptions.ConnectionError as error:

                print(
                    f"Gemini connection error: "
                    f"{error}"
                )

                if attempt == max_retries - 1:

                    raise RuntimeError(
                        "Could not connect to Gemini API "
                        f"after {max_retries} attempts."
                    )

                delay = retry_delays[
                    attempt
                ]

                jitter = random.uniform(
                    0,
                    2
                )

                time.sleep(
                    delay + jitter
                )

    # =========================================================
    # JSON PARSER
    # =========================================================

    def _parse_json_response(
        self,
        response,
        context="Gemini response"
    ):

        if not response:

            raise RuntimeError(
                f"{context} returned an empty response."
            )

        text = str(
            response
        ).strip()

        # Remove markdown code fences.
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

        # ---------------------------------------------------------
        # DIRECT JSON
        # ---------------------------------------------------------

        try:

            return json.loads(
                text
            )

        except json.JSONDecodeError:
            pass

        # ---------------------------------------------------------
        # FIND JSON OBJECT
        # ---------------------------------------------------------

        first_brace = text.find(
            "{"
        )

        last_brace = text.rfind(
            "}"
        )

        if (
            first_brace != -1
            and last_brace != -1
            and last_brace > first_brace
        ):

            json_text = text[
                first_brace:last_brace + 1
            ]

            try:

                return json.loads(
                    json_text
                )

            except json.JSONDecodeError as error:

                raise RuntimeError(
                    f"Invalid JSON returned by "
                    f"{context}: {error}"
                )

        # ---------------------------------------------------------
        # NOTHING WORKED
        # ---------------------------------------------------------

        raise RuntimeError(
            f"Could not parse JSON from "
            f"{context}."
        )
