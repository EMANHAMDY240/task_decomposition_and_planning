"""
Reflexion for the Adaptive Learning Path Planning agent.

Sub-task routed here: propose ONE COMPLETE learning path (ordered list of
course_ids) that satisfies every constraint together -- prerequisites,
budget, weekly hours, schedule, skill coverage, and deadline. This is
exactly the sub-task where a single retry (Self-Refine) isn't enough: the
constraints interact. Dropping an expensive course to fix the budget can
break skill coverage; adding a course to cover a missing skill can blow the
weekly-hours limit. The agent needs to learn across multiple attempts
within the same run, carrying forward what specifically went wrong.

Adapted from the toolkit's algorithms/reflexion.py:
- Same trial loop, same capped episodic memory buffer (memory[-memory_size:]).
- Swapped LangChain's BaseChatModel for the team's Anthropic SDK client
  (same pattern as self_refine.py and rag/generate.py).
- The toolkit's environment.evaluate() takes a raw string `attempt`. Ours
  takes a ProposedPath (see environment.py), so each attempt here is
  parsed from the model's JSON course_id list into a ProposedPath before
  being evaluated -- the grounded checks apply identically whether the
  path came from Reflexion, LATS, or a human.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from ..models import EnvironmentFeedback
from .environment import LearningPathEnvironment, ProposedPath

load_dotenv(Path(__file__).parent.parent / ".env")
client = anthropic.Anthropic()
MODEL = "claude-sonnet-4-6"


@dataclass
class ReflexionTrial:
    number: int
    course_ids: list[int]
    feedback: EnvironmentFeedback
    reflection: str | None = None


@dataclass
class ReflexionResult:
    success: bool
    best_course_ids: list[int]
    trials: list[ReflexionTrial]
    memory: list[str]


def _format_catalog(courses: list[dict]) -> str:
    lines = []
    for c in courses:
        lines.append(
            f"- id={c['course_id']} '{c['title']}' price={c['price']} "
            f"weekly_hours={c['weekly_hours']} dates={c['start_date']}..{c['end_date']} "
            f"skills={c['skill_tags']}"
        )
    return "\n".join(lines)


def _parse_course_ids(text: str) -> list[int]:
    match = re.search(r"\[[\d,\s]*\]", text)
    if not match:
        raise ValueError(f"Could not find a JSON list of course_ids in model output: {text!r}")
    return json.loads(match.group(0))


def reflexion(
    student_id: int,
    environment: LearningPathEnvironment,
    catalog_data: dict,
    max_trials: int = 3,
    memory_size: int = 3,
) -> ReflexionResult:
    """catalog_data: the dict returned by get_path_planning_data(student_id)
    -- pass it in once so this function doesn't re-fetch it every trial."""
    if max_trials < 1 or memory_size < 1:
        raise ValueError("max_trials and memory_size must be positive")

    goal = catalog_data["learning_goal"]
    catalog_text = _format_catalog(catalog_data["courses"])
    completed_text = ", ".join(str(c) for c in catalog_data["completed_course_ids"]) or "none"
    required_skills_text = ", ".join(catalog_data["required_skills"])

    memory: list[str] = []
    trials: list[ReflexionTrial] = []
    best_course_ids: list[int] = []
    best_score = -1.0

    for number in range(1, max_trials + 1):
        recalled = "\n".join(f"- {item}" for item in memory[-memory_size:]) or "- No prior trials."
        response = client.messages.create(
            model=MODEL,
            max_tokens=300,
            messages=[{
                "role": "user",
                "content": f"""Propose a complete course path for this student.

Budget: {goal['budget']}, weekly hours available: {goal['weekly_hours_available']}, target date: {goal['target_date']}
Required skills: {required_skills_text}
Already completed courses: {completed_text}

Available courses:
{catalog_text}

Lessons from previous failed attempts:
{recalled}

Respond with ONLY a JSON list of course_ids in the order the student should
take them, e.g. [3, 7, 12]. Apply the lessons above without discussing them.""",
            }],
        )
        raw = response.content[0].text.strip()
        course_ids = _parse_course_ids(raw)

        feedback = environment.evaluate(ProposedPath(student_id=student_id, course_ids=course_ids))
        trial = ReflexionTrial(number=number, course_ids=course_ids, feedback=feedback)

        if feedback.score > best_score:
            best_course_ids, best_score = course_ids, feedback.score

        if feedback.success:
            trials.append(trial)
            return ReflexionResult(True, course_ids, trials, memory[-memory_size:])

        reflect_response = client.messages.create(
            model=MODEL,
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": f"""Failed attempt: course_ids {course_ids}

Grounded environment feedback (score {feedback.score}):
{chr(10).join('- ' + item for item in feedback.details)}

State in one or two sentences, starting with 'I', what went wrong and the
specific strategy to use next trial (e.g. which course to drop or add).
Do not just repeat the feedback verbatim.""",
            }],
        )
        reflection = reflect_response.content[0].text.strip()
        trial.reflection = reflection
        trials.append(trial)
        memory.append(reflection)

    return ReflexionResult(False, best_course_ids, trials, memory[-memory_size:])