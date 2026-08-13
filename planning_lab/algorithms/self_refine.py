"""
Self-Refine for the Adaptive Learning Path Planning agent.

Sub-task this is routed to: generating the human-readable explanation the
student sees, justifying the recommended path (which courses, why, total
cost, timeline). This is cheap to redo -- a single short paragraph -- which
is exactly the kind of output Self-Refine fits (one draft, one critique
against an explicit rubric, one revision). It is NOT used for the path
search itself (that's LATS's job) or for full re-attempts across trials
(that's Reflexion's job, in reflexion.py).

Adapted from the toolkit's algorithms/self_refine.py:
- Same overall shape: deterministic_checks() feed a grounded_report into
  the critique prompt, then a revision step.
- Swapped the LLM client: the toolkit uses LangChain's BaseChatModel
  (ChatMistralAI). This project already calls the Anthropic SDK directly
  elsewhere (see rag/generate.py), so this file matches that pattern
  instead of introducing LangChain as a second LLM-calling convention.

Grounded vs. ungrounded, explicitly:
- UNGROUNDED: the critique_response call below asks the model to judge its
  own draft against a rubric. It can catch vague or incomplete writing, but
  it cannot catch a *factual* error confidently stated in fluent prose --
  a wrong total price or a course that isn't actually in the path reads
  just as "correct-sounding" to the same model that wrote it.
- GROUNDED: grounded_path_checks() below never calls an LLM. It checks the
  draft's claims against the same real data environment.py validates paths
  against (course titles, prices) -- deterministic string/number matching,
  not opinion.
See planning_eval/ for the documented case where grounded_path_checks()
caught a wrong total-cost claim that the ungrounded critique alone missed.
"""

import re
import os
from dataclasses import dataclass
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
client = anthropic.Anthropic()
MODEL = "claude-sonnet-4-6"


def grounded_path_checks(draft: str, path_courses: list[dict], all_courses: list[dict], total_price: float) -> list[str]:
    """Check the draft's factual claims against real course data -- the
    same `courses` records environment.py's evaluate() checks the path
    against. No LLM call, no opinion, just string/number matching."""
    issues: list[str] = []
    draft_lower = draft.lower()

    # Every course actually in the path should be named in the explanation.
    for course in path_courses:
        if course["title"].lower() not in draft_lower:
            issues.append(
                f"Draft doesn't mention '{course['title']}' (course_id "
                f"{course['course_id']}), which is part of the recommended path."
            )

    # A course NOT in the path shouldn't be named -- likely a hallucination.
    path_ids = {c["course_id"] for c in path_courses}
    for course in all_courses:
        if course["course_id"] not in path_ids and course["title"].lower() in draft_lower:
            issues.append(
                f"Draft mentions '{course['title']}', which is NOT part of "
                f"the recommended path -- likely a hallucinated course."
            )

    # The stated total cost should match the real sum of path course prices.
    mentioned_numbers = [float(n) for n in re.findall(r"\d+(?:\.\d{1,2})?", draft)]
    if not any(abs(n - total_price) < 1.0 for n in mentioned_numbers):
        issues.append(
            f"Draft doesn't state the correct total cost (${total_price:.2f}); "
            f"numbers mentioned: {mentioned_numbers or 'none'}."
        )

    return issues


@dataclass
class ReflectionResult:
    draft: str
    critique: str
    revised: str
    grounded_issues: list[str]


def reflect_and_refine(
    goal: str,
    draft: str,
    path_courses: list[dict],
    all_courses: list[dict],
    total_price: float,
) -> ReflectionResult:
    grounded = grounded_path_checks(draft, path_courses, all_courses, total_price)
    grounded_report = "\n".join(f"- {issue}" for issue in grounded) or "- Grounded checks passed."

    # UNGROUNDED step: a separate call, same model, judging the draft's
    # writing quality against a rubric -- it does not re-verify the facts.
    critique_response = client.messages.create(
        model=MODEL,
        max_tokens=400,
        messages=[{
            "role": "user",
            "content": f"""You are a separate critic. Judge this student-facing
explanation against the rubric: clarity, encouraging tone, and whether it
mentions the timeline. Do not rewrite it.

Goal: {goal}

External grounded checks (already run, treat as ground truth):
{grounded_report}

Draft:
{draft}

List concrete writing-quality issues. If there are none, respond exactly PASS.""",
        }],
    )
    critique = critique_response.content[0].text.strip()

    if critique.upper() == "PASS" and not grounded:
        revised = draft
    else:
        revise_response = client.messages.create(
            model=MODEL,
            max_tokens=400,
            messages=[{
                "role": "user",
                "content": f"""Revise this student-facing explanation using both
the grounded checks and the critique below. The grounded checks are facts
you must fix exactly as stated -- don't second-guess them.

Goal: {goal}

Draft:
{draft}

Grounded checks:
{grounded_report}

Critique:
{critique}

Return only the improved explanation.""",
            }],
        )
        revised = revise_response.content[0].text.strip()

    return ReflectionResult(draft, critique, revised, grounded)
