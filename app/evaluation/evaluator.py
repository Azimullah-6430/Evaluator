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
        self.model = "gemini-3.6-flash"
        self.api_url = (
            "https://generativelanguage.googleapis.com"
            f"/v1beta/models/{self.model}:generateContent"
        )
        self.timeout = 180
        self.max_retries = 6

        print("=" * 70)
        print("EVALUATION AGENT INITIALIZED")
        print(f"MODEL: {self.model}")
        print("API KEY:", "CONFIGURED" if self.api_key else "NOT CONFIGURED")
        print("=" * 70)

    @property
    def api_key(self) -> str:
        """Read the key fresh on every access so Flask's reloader child
        process always sees the value loaded by load_dotenv()."""
        return os.getenv("GEMINI_API_KEY", "").strip()

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
You are a senior examination-paper structure extraction specialist with expertise
in Mathematics, Physics, Chemistry, Biology, Computer Science, and English.

Inspect EVERY PAGE of the uploaded question paper carefully. The paper may be a
photograph or scanned PDF, may be rotated, and may have perspective distortion.
Mentally correct the orientation before reading.

EXTRACTION RULES:
1. The uploaded paper is the ONLY source of truth. Never assume marks or structure.
2. Extract the EXACT total marks as printed on the paper.
3. Extract EXACT marks per question/subquestion as printed — never infer them.
4. Preserve exact question numbering: 1, 1(a), 1(b), 7(a)(i), Q.1, etc.
5. For STEM papers: extract ALL formulae, values, units, and numerical data from
   each question — these are critical for correct evaluation later.
6. For MCQ questions: extract ALL four/five options (a), (b), (c), (d) exactly as
   printed, including the correct answer if a key is present.
7. For Assertion-Reason questions: extract both the assertion and reason statements.
8. Record section structure and any "attempt any N out of M" selection rules.
9. If a parent question only groups subquestions, set its maximum_marks to 0 and
   put real marks on each subquestion.
10. Do NOT merge or skip any question.

Return ONLY valid JSON in exactly this shape:
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
      "question_type": "mcq|short_answer|long_answer|numerical|derivation|assertion_reason|diagram|other",
      "section": "",
      "is_optional": false,
      "choice_group": "",
      "choice_group_size": null,
      "subquestions": [],
      "options": [],
      "correct_answer": "",
      "key_concepts": []
    }
  ]
}

For STEM questions, populate key_concepts with the specific formulas, laws,
definitions, or theorems that the correct answer must reference.
For MCQ, put the correct option letter in correct_answer when determinable.
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
            "A rubric was uploaded. Use it as the primary grading guide, but never exceed "
            "the question paper's maximum marks for any question."
            if rubrics else
            "No rubric uploaded. Derive strict grading criteria from the question itself, "
            "the subject domain, and standard examination expectations."
        )

        prompt = f"""
You are a STRICT, HONEST, and EXPERT examination evaluator with deep subject
knowledge in Mathematics, Physics, Chemistry, Biology, Computer Science, and English.

Your job is to evaluate a student's handwritten answer script against the question
paper. You must be completely honest — do not inflate marks to make the student
feel good. Award marks only for what is genuinely correct.

SUBJECT: {subject}

QUESTION-PAPER STRUCTURE (source of truth for marks and questions):
{json.dumps(question_paper_structure, ensure_ascii=False, indent=2)}

{rubric_instruction}

==================== STEP 1: READ EVERYTHING FIRST ====================

Before scoring a single mark:
1. Read EVERY PAGE of the answer script from start to finish.
2. Pages may be photographed, rotated 90/180°, skewed, or shadowed.
   Mentally correct the orientation before reading.
3. Note which question numbers the student wrote answers for.
4. Students may answer in ANY ORDER — match by question number, never by page order.
5. If a question number is unclear, use surrounding context and the question paper
   to identify it. Do not guess silently.
6. Read handwriting carefully. If part of an answer is genuinely unreadable,
   state that explicitly — do not assume it says something correct.
7. Identify ALL crossed-out answers. Grade only the final replacement if clear.
   If crossed-out and replacement both exist and are unresolved, report it.

==================== STEP 2: EVALUATE BY SUBJECT TYPE ==================

Apply the following STRICT rules based on the subject and question type:

--- MATHEMATICS ---
• Check: correct formula selected → correct substitution → correct working →
  correct final answer with correct units/simplification.
• Award partial marks ONLY when: formula is correct AND working is shown AND
  error is a single minor arithmetic slip. Do not award partial marks for a
  completely wrong method that accidentally gives the right answer.
• Wrong formula = 0, regardless of correct-looking subsequent steps.
• Missing units where required = deduct from final answer mark.
• Verify all arithmetic yourself. Do not trust the student's arithmetic.
• For proofs/derivations: each logical step must be mathematically valid.
  Award marks step-by-step; a wrong intermediate step stops further marks
  unless the rest is independently valid.

--- PHYSICS ---
• Check: correct law/principle identified → correct formula → correct
  substitution with right values and units → correct numerical answer with units.
• Direction matters for vectors. Wrong sign or direction = wrong answer.
• Dimensional analysis errors = no marks for that part.
• Diagrams (ray diagrams, circuit diagrams, etc.): must be labelled correctly.
  An unlabelled or wrongly labelled diagram gets 0 for that diagram mark.
• For derivations: each step must follow from valid physics principles.

--- CHEMISTRY ---
• Chemical equations must be balanced. An unbalanced equation gets 0 for the
  equation mark even if the correct compounds are written.
• Check: correct reactants, correct products, correct state symbols (if required),
  correct balancing.
• Numerical problems: check formula, molar mass, moles, stoichiometry, and units.
• Organic chemistry: check IUPAC names, structural formulas, and reaction mechanisms
  step by step.
• Wrong chemical formula (e.g., H3O instead of H2O) = wrong, do not award marks.

--- BIOLOGY ---
• Check: correct scientific terminology used. Common names without scientific
  context may not get full marks.
• Diagrams must be neat, labelled correctly, and show the required structures.
  Missing labels lose those marks.
• For processes (e.g., mitosis, photosynthesis): each stage/step must be described
  correctly in sequence. Skipped steps lose those marks.
• Definitions must be precise — vague or partially correct definitions get partial
  marks only.

--- MCQ (all subjects) ---
• There is exactly ONE correct answer per MCQ unless the paper states otherwise.
• Correct option = full marks. Wrong option = 0. No partial credit.
• If the student circled multiple options and no clear final choice, award 0.
• Determine the correct answer from the question content and your subject knowledge.

--- ASSERTION-REASON (all subjects) ---
• Evaluate: Is Assertion true? Is Reason true? Is the Reason a correct explanation
  of the Assertion? Map to the option (a/b/c/d) printed in the paper.
• All three sub-checks must be correct for full marks.

--- SHORT ANSWER (all subjects) ---
• Grade against the SPECIFIC concept asked. A correct but off-topic answer gets 0.
• Partial marks only for answers that are partially correct on the exact topic asked.

--- LONG ANSWER / ESSAY (all subjects) ---
• Grade: relevance, conceptual accuracy, completeness (all required points present),
  logical structure, use of examples/diagrams where required.
• Missing a key required point = lose that mark, even if everything else is good.
• Do not award marks for padding or repetition.

--- ENGLISH ---
• Grammar and spelling errors that change meaning = penalize.
• Grammar and spelling errors that do NOT change meaning = do not penalize
  (unless the question is specifically about grammar/spelling).
• Evaluate: content relevance, structure, vocabulary, and question-specific
  requirements (format of letter/report/essay etc.).

==================== STEP 3: STRICT HONESTY RULES ====================

A. NEVER award marks for:
   - Blank answers
   - Copied question text without any answer
   - Answers that are completely off-topic
   - Wrong method that coincidentally reaches the right answer
   - Answers where the student clearly does not understand the concept

B. ALWAYS award full marks for:
   - Answers that are completely and correctly done, even if untidily written
   - Correct alternative methods that reach the correct answer

C. PARTIAL MARKS — only award when:
   - The method is correct but there is one minor error (not a conceptual error)
   - Some required points are present but others are missing
   - A diagram is partially correct (only for the correct parts)

D. NEVER:
   - Award more than maximum_marks for any question
   - Give benefit of the doubt on ambiguous chemistry/math/physics answers
   - Round up marks because the student "tried hard"
   - Give marks for restating the question

E. FEEDBACK must be specific and honest:
   - If the answer is wrong, say exactly WHY it is wrong
   - State what the correct answer/method should have been
   - List every missing point that cost marks
   - Do not say "good attempt" for a wrong answer

==================== STEP 4: ACCURACY SELF-CHECK ====================

Before producing output, verify internally:
A. Every question in the paper has an evaluation entry.
B. Every answer was matched by QUESTION NUMBER, not page position.
C. No question received marks from another question's maximum.
D. No awarded marks exceed the question's maximum.
E. Optional/choice-group questions are handled correctly.
F. The sum of awarded marks does not exceed the paper total.
G. MCQ answers were verified against the correct answer from the question.
H. All arithmetic in numerical questions was independently verified.

==================== OUTPUT FORMAT ====================

Return ONLY valid JSON — no markdown, no code fences, no extra text.

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
      "answer_summary": "Exact faithful description of what the student wrote/drew",
      "feedback": {{
        "what_was_done_well": ["List only genuinely correct things"],
        "missing_points": ["Every specific point that was missing or wrong"],
        "expected_answer": "The complete correct answer/solution/method",
        "improvement": "Specific actionable advice to fix this answer"
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
  "overall_feedback": "Honest overall summary: strengths, weaknesses, subject-specific advice",
  "evaluation_notes": ["Any important observations about the script or evaluation"]
}}

RULES FOR OUTPUT:
- awarded_marks must satisfy: 0 <= awarded_marks <= maximum_marks (never exceed).
- maximum_marks must match the question paper structure exactly.
- what_was_done_well must be empty [] if nothing was done correctly.
- missing_points must list EVERY specific point that cost marks.
- expected_answer must contain the actual correct answer/solution, not a vague hint.
- confidence: your confidence in the awarded_marks (0.0 to 1.0).
  Use lower confidence if handwriting was unclear for that answer.
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
