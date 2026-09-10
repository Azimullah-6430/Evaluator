import os
import json
import base64
import random
import time
import traceback
import re
from typing import Any, Dict, List, Optional

import requests


class EvaluationAgent:
    """Multimodal examination evaluator for printed QPs and handwritten scripts."""

    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.model = "gemini-3.6-flash"
        self.api_url = (
            "https://generativelanguage.googleapis.com/"
            f"/v1beta/models/{self.model}:generateContent"
        )
        self.timeout = 180
        self.max_retries = 6

        print("=" * 70)
        print("EVALUATION AGENT INITIALIZED")
        print(f"MODEL: {self.model}")
        print("API KEY:", "CONFIGURED" if self.api_key else "NOT CONFIGURED")
        print("=" * 70)

    def evaluate(self, request: Dict[str, Any]) -> Dict[str, Any]:
        print("=" * 70)
        print("EVALUATION AGENT STARTED")
        print("=" * 70)
        try:
            request = self._normalize_request(request)
            self._validate_request(request)

            print("[1/3] Reading question paper structure...")
            qp = self.analyze_question_paper(request["question_paper"])
            qp = self._normalize_question_paper(qp)
            self._validate_question_paper_structure(qp)

            print(f"Detected total marks: {qp['total_marks']}")
            print(f"Detected questions: {len(qp['questions'])}")

            print("[2/3] Evaluating original handwritten answer script visually...")
            result = self.evaluate_visual_script(
                question_paper=request["question_paper"],
                answer_script=request["answer_script"],
                question_paper_structure=qp,
                subject=request["subject"],
                rubrics=request.get("rubrics"),
            )

            print("[3/3] Validating and calculating final marks...")
            result = self._normalize_evaluations(result)
            final = self.calculate_final_result(qp, result)

            final["student"] = {
                "name": request.get("student_name", ""),
                "roll_number": request.get("roll_number", ""),
                "subject": request.get("subject", ""),
            }
            final["question_paper"] = qp

            print("=" * 70)
            print("EVALUATION COMPLETED")
            print(f"FINAL: {final['obtained_marks']}/{final['total_marks']}")
            print("=" * 70)
            return final

        except Exception as exc:
            print("=" * 70)
            print("EVALUATION ERROR")
            print(f"ERROR: {exc}")
            traceback.print_exc()
            print("=" * 70)
            raise RuntimeError(f"Evaluation failed: {exc}") from exc

    def _normalize_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize file inputs so both path strings and file-info objects work.

        Older versions of the Flask route may pass a saved file path directly,
        while newer versions pass {"path": ...}. Accept both forms to prevent
        avoidable "information must be an object" failures.
        """
        if not isinstance(request, dict):
            raise ValueError("Evaluation request must be an object.")

        normalized = dict(request)
        for field in ("question_paper", "answer_script", "rubrics"):
            value = normalized.get(field)
            if isinstance(value, str) and value.strip():
                normalized[field] = {"path": value.strip()}
            elif isinstance(value, dict):
                # Accept common path key variants from older route code.
                if not value.get("path"):
                    for key in ("filepath", "file_path", "saved_path", "filename"):
                        if value.get(key):
                            value = dict(value)
                            value["path"] = value[key]
                            break
                normalized[field] = value
        return normalized

    def _validate_request(self, request: Dict[str, Any]) -> None:
        if not self.api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not configured. Add GEMINI_API_KEY to Render Environment Variables."
            )
        if not isinstance(request, dict):
            raise ValueError("Evaluation request must be an object.")
        for field in ("question_paper", "answer_script", "subject"):
            if not request.get(field):
                raise ValueError(f"Missing required field: {field}")
        self._validate_file(request["question_paper"], "question paper")
        self._validate_file(request["answer_script"], "answer script")
        if request.get("rubrics"):
            self._validate_file(request["rubrics"], "rubrics")

    def _validate_file(self, info: Dict[str, Any], label: str) -> None:
        if not isinstance(info, dict):
            raise ValueError(f"{label} information must be an object.")
        path = info.get("path")
        if not path:
            raise ValueError(f"{label} path is missing.")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{label} file not found: {path}")

    # ------------------------------------------------------------------
    # QUESTION PAPER: one multimodal call. The QP remains the source of truth.
    # ------------------------------------------------------------------
    def analyze_question_paper(self, question_paper: Dict[str, Any]) -> Dict[str, Any]:
        prompt = r"""
You are an examination-paper structure extraction specialist.

Inspect EVERY PAGE of the uploaded question paper. The paper may be a photo or
scanned PDF, may be rotated 90 degrees, and may have perspective distortion.
Mentally rotate/rectify it before reading.

The uploaded paper is the ONLY source of truth for:
- total examination marks
- maximum marks per question/subquestion
- section structure
- internal choices
- "answer any N" rules
- question numbering

Never assume a 20/50/100 mark pattern.
Never infer marks from a generic school/university pattern when the paper does
not explicitly support that inference.

Read all pages before producing the answer. Preserve exact question numbering,
including forms such as 1(a), 1(b), 7(a)(i), etc.

Important: a table or section can contain choice groups. Record the rule rather
than treating every alternative as compulsory.

Return ONLY valid JSON in exactly this high-level shape:
{
  "subject": "",
  "total_marks": 0,
  "duration": "",
  "instructions": [],
  "sections": [
    {
      "name": "",
      "marks": 0,
      "selection_rule": "",
      "question_numbers": []
    }
  ],
  "questions": [
    {
      "question_number": "1",
      "question_text": "",
      "maximum_marks": 0,
      "question_type": "mcq|short_answer|long_answer|numerical|derivation|assertion_reason|other",
      "section": "",
      "is_optional": false,
      "choice_group": "",
      "choice_group_size": null,
      "subquestions": [],
      "options": []
    }
  ]
}

For every scored item, maximum_marks MUST be the marks assigned by the paper.
If a parent question only groups separately scored subquestions, put its own
maximum_marks as 0 and put the real marks on subquestions.
"""
        text = self._call_gemini(prompt, [question_paper])
        return self._parse_json_response(text, "question paper analysis")

    def _normalize_question_paper(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(data, dict):
            raise ValueError("Question paper response must be a JSON object.")

        total = self._number(data.get("total_marks"))
        if total is None or total <= 0:
            raise ValueError("Could not extract a valid total mark from the question paper.")

        raw = data.get("questions", [])
        if isinstance(raw, dict):
            raw = [dict(v, question_number=str(k)) if isinstance(v, dict) else {"question_number": str(k), "maximum_marks": v}
                   for k, v in raw.items()]
        if not isinstance(raw, list):
            raise ValueError("Question paper 'questions' must be a list.")

        questions = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            qno = item.get("question_number", item.get("number"))
            if qno is None:
                continue
            marks = self._number(item.get("maximum_marks", item.get("max_marks")))
            if marks is None:
                raise ValueError(f"Maximum marks missing for question {qno}.")
            subs = item.get("subquestions", [])
            if isinstance(subs, dict):
                subs = [dict(v, question_number=str(k)) if isinstance(v, dict) else {"question_number": str(k), "maximum_marks": v}
                        for k, v in subs.items()]
            if not isinstance(subs, list):
                subs = []

            clean_subs = []
            for sub in subs:
                if not isinstance(sub, dict):
                    continue
                sqno = sub.get("question_number", sub.get("number"))
                smarks = self._number(sub.get("maximum_marks", sub.get("max_marks")))
                if sqno is not None and smarks is not None:
                    clean_subs.append({
                        "question_number": str(sqno).strip(),
                        "question_text": str(sub.get("question_text", "")),
                        "maximum_marks": smarks,
                        "question_type": str(sub.get("question_type", "")),
                    })

            questions.append({
                "question_number": str(qno).strip(),
                "question_text": str(item.get("question_text", "")),
                "maximum_marks": marks,
                "question_type": str(item.get("question_type", "")),
                "section": str(item.get("section", "")),
                "is_optional": bool(item.get("is_optional", False)),
                "choice_group": str(item.get("choice_group", "")),
                "choice_group_size": item.get("choice_group_size"),
                "subquestions": clean_subs,
                "options": item.get("options", []) if isinstance(item.get("options", []), list) else [],
            })

        if not questions:
            raise ValueError("No valid questions were extracted from the question paper.")

        data["total_marks"] = total
        data["questions"] = questions
        return data

    def _validate_question_paper_structure(self, qp: Dict[str, Any]) -> None:
        seen = set()
        for q in qp["questions"]:
            qno = q["question_number"]
            norm = self._normalize_qno(qno)
            if norm in seen:
                raise ValueError(f"Duplicate question number detected: {qno}")
            seen.add(norm)
            if q["maximum_marks"] < 0:
                raise ValueError(f"Invalid marks for question {qno}")
            for sub in q.get("subquestions", []):
                if sub["maximum_marks"] <= 0:
                    raise ValueError(f"Invalid marks for subquestion {sub['question_number']}")

    # ------------------------------------------------------------------
    # DIRECT VISUAL EVALUATION: original QP + original handwritten script
    # ------------------------------------------------------------------
    def evaluate_visual_script(
        self,
        question_paper: Dict[str, Any],
        answer_script: Dict[str, Any],
        question_paper_structure: Dict[str, Any],
        subject: str,
        rubrics: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        rubric_instruction = (
            "A rubric was uploaded. Use it as secondary grading guidance, but never exceed "
            "the question paper's marks."
            if rubrics else
            "No rubric was uploaded. Generate question-specific grading criteria from the question itself."
        )

        prompt = f"""
You are the final multimodal examination evaluator.

SUBJECT PROVIDED BY TEACHER:
{subject}

QUESTION-PAPER STRUCTURE ALREADY EXTRACTED:
{json.dumps(question_paper_structure, ensure_ascii=False, indent=2)}

{rubric_instruction}

You have the ORIGINAL QUESTION PAPER and the ORIGINAL HANDWRITTEN ANSWER SCRIPT
attached to this request. The images/PDF pages are authoritative visual evidence.

================ VISUAL READING RULES ================

1. Inspect EVERY PAGE of the answer script before scoring anything.
2. Pages may be photographed, rotated 90/180 degrees, skewed, shadowed, or have
   perspective distortion. Mentally rotate/rectify each page before reading.
3. The handwriting is ordinary student handwriting in ruled notebooks. Read the
   actual writing, not just OCR-like guesses.
4. Red ticks/corrections/teacher marks are NOT automatically part of the student's
   answer. Do not use teacher marks as evidence of correctness unless the task
   explicitly provides an official correction key.
5. A page may contain answers to several questions.
6. A student can answer in ANY ORDER. Q10 before Q3, Q5 before Q2, etc. is valid.
7. Match answers by the question number written by the student, never by page order.
8. If a question number is written unclearly, inspect nearby text and the question
   paper. Do not silently assign it to another question.
9. Do not invent unreadable handwriting. If a small part is unreadable, mark that
   part uncertain and grade only what can actually be established.
10. Preserve mathematical symbols, fractions, powers, signs, units, equations,
    diagrams and intermediate working as accurately as possible.

================ QUESTION-PAPER RULES ================

11. The QUESTION PAPER is the source of truth for maximum marks.
12. Never use a hardcoded total such as 20, 50, or 100.
13. Never use a hardcoded mark per question.
14. Respect sections and internal choices.
15. If the paper says "answer any 2/3/4", do NOT award compulsory marks for every
    alternative. Determine which alternatives the student actually attempted and
    apply the paper's selection rule.
16. If a parent question contains scored subquestions, score the subquestions
    separately and do not double-count the parent.

================ EVALUATION RULES ================

17. MCQ: compare the student's selected option with the correct answer derived from
    the question. Correct gets the full question marks; incorrect gets zero unless
    the paper explicitly provides partial credit.
18. Assertion-Reason: evaluate both statements and the logical relationship using
    the options printed in the paper.
19. Mathematics/numerical: check the method, formula, substitutions, calculations,
    signs, units and final answer. Give justified partial marks for correct work with
    a minor arithmetic/final-answer error.
20. Derivations/proofs: award marks for valid logical steps, not only the final line.
21. Long answers: grade relevance, conceptual correctness, completeness, explanation,
    examples/diagrams where required, and important missing points.
22. Short answers: grade against the exact concept asked by the question.
23. Do not award marks merely because an answer is long.
24. Do not penalize spelling/grammar when the academic meaning is clear, unless it
    changes the technical meaning.
25. Never award more than the maximum marks.
26. Never give negative marks unless the uploaded paper explicitly requires them.
27. If the answer is blank, award 0.
28. If the student crossed out an answer and clearly supplied a replacement, grade
    the final replacement. If both attempts are visibly unresolved, report that.
29. If multiple answers are given for an MCQ and the final selection is unclear,
    do not guess.

================ ACCURACY SAFETY ====================

Before finalizing, perform a second internal check:
A. Every scored question in the paper has been considered.
B. Every student answer has been matched by QUESTION NUMBER.
C. No answer was shifted because of page order.
D. No question received another question's marks.
E. No awarded mark exceeds its maximum.
F. The final total respects optional-choice rules.
G. The final total cannot exceed the examination total.

================ OUTPUT ==============================

Return ONLY valid JSON.

Use this structure:
{{
  "evaluations": [
    {{
      "question_number": "1",
      "answer_present": true,
      "maximum_marks": 1,
      "awarded_marks": 1,
      "question_type": "mcq",
      "choice_group": "",
      "correct": true,
      "confidence": 0.95,
      "answer_summary": "Short faithful summary of what the student wrote",
      "feedback": {{
        "what_was_done_well": [],
        "missing_points": [],
        "expected_answer": "Correct/ideal answer based on the question",
        "improvement": "Specific improvement"
      }}
    }}
  ],
  "choice_groups": [
    {{
      "choice_group": "",
      "required_attempts": 2,
      "attempted_questions": [],
      "counted_questions": [],
      "ignored_optional_questions": []
    }}
  ],
  "overall_feedback": "",
  "evaluation_notes": []
}}

IMPORTANT:
- Return an evaluation entry for each actual scored question/subquestion.
- maximum_marks must match the question paper structure exactly.
- choice_group must match the question paper structure exactly when applicable.
- awarded_marks must be numeric and satisfy 0 <= awarded_marks <= maximum_marks.
- Do not output markdown or code fences.
"""

        files = [question_paper, answer_script]
        if rubrics:
            files.append(rubrics)
        text = self._call_gemini(prompt, files)
        return self._parse_json_response(text, "visual answer evaluation")

    # ------------------------------------------------------------------
    # FINAL VALIDATION / SCORE
    # ------------------------------------------------------------------
    def _normalize_evaluations(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(data, dict):
            raise ValueError("Evaluation response must be a JSON object.")

        raw = data.get("evaluations", [])
        if isinstance(raw, dict):
            raw = [dict(v, question_number=str(k)) if isinstance(v, dict) else {"question_number": str(k), "awarded_marks": 0}
                   for k, v in raw.items()]
        if not isinstance(raw, list):
            raise ValueError("Evaluation 'evaluations' must be a list.")

        clean = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            qno = item.get("question_number")
            if qno is None:
                continue
            awarded = self._number(item.get("awarded_marks", 0))
            maximum = self._number(item.get("maximum_marks", 0))
            if awarded is None:
                awarded = 0.0
            if maximum is None:
                maximum = 0.0
            feedback = item.get("feedback", {})
            if not isinstance(feedback, dict):
                feedback = {}
            clean.append({
                "question_number": str(qno).strip(),
                "answer_present": bool(item.get("answer_present", False)),
                "maximum_marks": maximum,
                "awarded_marks": awarded,
                "question_type": str(item.get("question_type", "")),
                "choice_group": str(item.get("choice_group", "")),
                "optional_status": str(item.get("optional_status", "")),
                "correct": bool(item.get("correct", False)),
                "confidence": item.get("confidence", 0),
                "answer_summary": str(item.get("answer_summary", "")),
                "feedback": {
                    "what_was_done_well": self._as_list(feedback.get("what_was_done_well", [])),
                    "missing_points": self._as_list(feedback.get("missing_points", [])),
                    "expected_answer": str(feedback.get("expected_answer", "")),
                    "improvement": str(feedback.get("improvement", "")),
                },
            })
        data["evaluations"] = clean
        return data

    def calculate_final_result(self, qp: Dict[str, Any], ai: Dict[str, Any]) -> Dict[str, Any]:
        total = float(qp["total_marks"])
        eval_map = {}
        for e in ai.get("evaluations", []):
            key = self._normalize_qno(e.get("question_number", ""))
            if key:
                eval_map[key] = e

        final_evaluations = []
        raw_obtained = 0.0

        # Flatten parent/subquestion structure. If a parent has scored subs,
        # use those subs and do not double count the parent.
        scored_items = []
        for q in qp["questions"]:
            subs = q.get("subquestions", [])
            if subs:
                for s in subs:
                    scored_items.append({
                        **s,
                        "section": q.get("section", ""),
                        "is_optional": q.get("is_optional", False),
                        "choice_group": q.get("choice_group", ""),
                    })
            elif q.get("maximum_marks", 0) > 0:
                scored_items.append(q)

        for q in scored_items:
            qno = str(q["question_number"])
            key = self._normalize_qno(qno)
            max_marks = float(q["maximum_marks"])
            e = eval_map.get(key)

            if e is None:
                e = {
                    "question_number": qno,
                    "answer_present": False,
                    "maximum_marks": max_marks,
                    "awarded_marks": 0.0,
                    "question_type": q.get("question_type", ""),
                    "correct": False,
                    "confidence": 1.0,
                    "answer_summary": "No answer detected for this question.",
                    "choice_group": str(q.get("choice_group", "")),
                    "optional_status": "",
                    "feedback": {
                        "what_was_done_well": [],
                        "missing_points": ["No answer detected."],
                        "expected_answer": "",
                        "improvement": "Attempt the question with the required concepts and working.",
                    },
                }
            else:
                e = dict(e)
                awarded = self._number(e.get("awarded_marks", 0)) or 0.0
                e["maximum_marks"] = max_marks
                e["choice_group"] = str(q.get("choice_group", e.get("choice_group", "")))
                e["awarded_marks"] = round(max(0.0, min(awarded, max_marks)), 2)

            raw_obtained += float(e["awarded_marks"])
            final_evaluations.append(e)

        # Apply optional-choice rules conservatively.
        # The AI supplies counted_questions when it has identified a choice group.
        # We only override scoring when the group has an explicit required count and
        # the counted list is reliable.
        final_evaluations = self._apply_choice_rules(
            final_evaluations,
            ai.get("choice_groups", []),
        )

        obtained = min(
            total,
            round(sum(float(e.get("awarded_marks", 0)) for e in final_evaluations), 2)
        )
        percentage = round((obtained / total) * 100, 2) if total else 0.0

        return {
            "obtained_marks": obtained,
            "total_marks": total,
            "percentage": percentage,
            "grade": self._grade(percentage),
            "evaluations": final_evaluations,
            "overall_feedback": str(ai.get("overall_feedback", "")),
            "evaluation_notes": self._as_list(ai.get("evaluation_notes", [])),
        }

    def _apply_choice_rules(self, evaluations: List[Dict[str, Any]], groups: Any) -> List[Dict[str, Any]]:
        """Apply explicit optional-question selections without guessing group membership.

        The question-paper extraction supplies the authoritative choice_group on each
        scored item. The evaluator response supplies counted_questions. We only change
        marks when BOTH pieces of information agree.
        """
        if not isinstance(groups, list):
            return evaluations

        by_q = {self._normalize_qno(e.get("question_number")): e for e in evaluations}

        # Build authoritative membership from the extracted question paper fields
        # already copied into each evaluation by calculate_final_result.
        for group in groups:
            if not isinstance(group, dict):
                continue

            group_name = str(group.get("choice_group", "")).strip()
            required = self._number(group.get("required_attempts"))
            counted = group.get("counted_questions", [])

            if not group_name or required is None or required <= 0:
                continue
            if not isinstance(counted, list):
                continue

            counted_keys = {
                self._normalize_qno(x) for x in counted
                if x is not None and str(x).strip()
            }
            if not counted_keys:
                continue

            for key, evaluation in by_q.items():
                if str(evaluation.get("choice_group", "")).strip() != group_name:
                    continue
                if key not in counted_keys:
                    evaluation["awarded_marks"] = 0.0
                    evaluation["optional_status"] = "ignored_optional_alternative"
                else:
                    evaluation["optional_status"] = "counted_attempt"

        return evaluations

    # ------------------------------------------------------------------
    # GEMINI REQUEST WITH RETRIES
    # ------------------------------------------------------------------
    def _call_gemini(self, prompt: str, files: Optional[List[Dict[str, Any]]] = None) -> str:
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured.")

        parts = [{"text": prompt}]
        for info in files or []:
            if not isinstance(info, dict):
                raise ValueError("Invalid file information supplied to Gemini.")
            path = info.get("path")
            if not path or not os.path.isfile(path):
                raise FileNotFoundError(f"File not found: {path}")
            mime = info.get("mime_type") or info.get("mimetype") or info.get("content_type")
            if not mime:
                mime = self._guess_mime(path)
            with open(path, "rb") as fh:
                encoded = base64.b64encode(fh.read()).decode("utf-8")
            parts.append({"inline_data": {"mime_type": mime, "data": encoded}})

        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
            },
        }
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }

        retry_delays = [5, 10, 20, 40, 60]
        temporary_codes = {429, 500, 502, 503, 504}

        for attempt in range(self.max_retries):
            try:
                print(f"Calling Gemini API (attempt {attempt + 1}/{self.max_retries})...")
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                print(f"Gemini HTTP status: {response.status_code}")

                if response.status_code == 200:
                    try:
                        data = response.json()
                    except ValueError as exc:
                        raise RuntimeError("Gemini returned HTTP 200 but invalid JSON envelope.") from exc

                    candidates = data.get("candidates", [])
                    if not candidates:
                        raise RuntimeError("Gemini returned no candidates.")
                    candidate = candidates[0]
                    reason = candidate.get("finishReason")
                    if reason == "SAFETY":
                        raise RuntimeError("Gemini blocked the request for safety reasons.")
                    if reason == "MAX_TOKENS":
                        raise RuntimeError("Gemini response was truncated by the output token limit.")

                    content = candidate.get("content") or {}
                    response_parts = content.get("parts") or []
                    texts = [p.get("text", "") for p in response_parts if isinstance(p, dict) and p.get("text")]
                    if not texts:
                        raise RuntimeError("Gemini returned an empty response.")
                    return "\n".join(texts)

                try:
                    err = response.json()
                except ValueError:
                    err = response.text

                if response.status_code in temporary_codes:
                    print(f"Temporary Gemini error {response.status_code}: {err}")
                    if attempt == self.max_retries - 1:
                        raise RuntimeError(
                            f"Gemini remained unavailable after {self.max_retries} attempts. "
                            f"HTTP {response.status_code}: {err}"
                        )
                    delay = retry_delays[min(attempt, len(retry_delays) - 1)] + random.uniform(0, 2)
                    print(f"Retrying Gemini in {delay:.1f} seconds...")
                    time.sleep(delay)
                    continue

                raise RuntimeError(
                    f"Gemini API request failed. HTTP Status: {response.status_code} "
                    f"Model: {self.model} Response: {err}"
                )

            except requests.exceptions.Timeout as exc:
                if attempt == self.max_retries - 1:
                    raise RuntimeError("Gemini request timed out after multiple attempts.") from exc
                time.sleep(retry_delays[min(attempt, len(retry_delays) - 1)])
            except requests.exceptions.ConnectionError as exc:
                if attempt == self.max_retries - 1:
                    raise RuntimeError("Could not connect to Gemini after multiple attempts.") from exc
                time.sleep(retry_delays[min(attempt, len(retry_delays) - 1)])

        raise RuntimeError("Gemini request failed unexpectedly.")

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------
    @staticmethod
    def _number(value: Any) -> Optional[float]:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if value is None:
            return None
        m = re.search(r"-?\d+(?:\.\d+)?", str(value))
        return float(m.group()) if m else None

    @staticmethod
    def _as_list(value: Any) -> List[Any]:
        if isinstance(value, list):
            return value
        if value in (None, ""):
            return []
        return [value]

    @staticmethod
    def _guess_mime(path: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        return {
            ".pdf": "application/pdf",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
        }.get(ext, "application/octet-stream")

    @staticmethod
    def _normalize_qno(value: Any) -> str:
        s = str(value or "").strip().lower()
        s = re.sub(r"^question\s*", "", s)
        s = re.sub(r"^q\s*\.?\s*", "", s)
        s = re.sub(r"\s+", "", s)
        return s

    @staticmethod
    def _grade(p: float) -> str:
        if p >= 90:
            return "A+"
        if p >= 80:
            return "A"
        if p >= 70:
            return "B+"
        if p >= 60:
            return "B"
        if p >= 50:
            return "C"
        if p >= 40:
            return "D"
        return "F"

    @staticmethod
    def _parse_json_response(text: str, context: str) -> Dict[str, Any]:
        if not text:
            raise RuntimeError(f"{context} returned an empty response.")
        cleaned = re.sub(r"^```(?:json)?\s*", "", str(text).strip(), flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
        try:
            obj = json.loads(cleaned)
            if not isinstance(obj, dict):
                raise RuntimeError(f"{context} returned JSON that is not an object.")
            return obj
        except json.JSONDecodeError:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start >= 0 and end > start:
                try:
                    obj = json.loads(cleaned[start:end + 1])
                    if isinstance(obj, dict):
                        return obj
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"Invalid JSON from {context}: {exc}") from exc
        raise RuntimeError(f"Could not parse JSON from {context}.")
