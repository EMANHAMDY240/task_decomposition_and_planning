"""
Grounded Environment for the Adaptive Learning Path Planning agent.

Replaces the toolkit's algorithms/environment.py, which returns a random
beta-distributed score and ignores the candidate entirely:

    score = round(self.rng.betavariate(5.0, 2.0), 4)   # <- fake, delete this

This version calls the real MCP tool get_path_planning_data(student_id)
and checks the *actual* proposed course path against real constraints:
prerequisites, budget, weekly hours, schedule conflicts, skill coverage,
and the target deadline.

Keep the same public shape the toolkit expects (an `evaluate()` method
returning `EnvironmentFeedback`) so reflexion.py and LATS can plug this in
without changing their own code.
"""

from datetime import date
from dataclasses import dataclass


from ..models import EnvironmentFeedback


@dataclass
class ProposedPath:
    """The 'state' this environment evaluates: an ordered list of course_ids
    the agent is proposing for one student, plus which student it's for."""
    student_id: int
    course_ids: list[int]  # in the order the agent proposes to take them


class LearningPathEnvironment:
    """Grounded evaluator: checks a proposed learning path against real
    BrightPeak data (courses, prerequisites, learning_goals) via the
    get_path_planning_data MCP tool. No LLM call, no randomness.

    Follows the same pattern agent/loop.py already uses for db_tool
    questions (`from server import get_student_profile`) -- the MCP tool
    functions are plain Python functions decorated with @mcp.tool(), so
    they can be imported and called directly, no separate MCP client
    object needed here.
    """

    def __init__(self, mcp_server_path: str | None = None):
        # mcp_server_path: optional sys.path entry pointing at your
        # mcp_server/ folder, in case this file is imported from somewhere
        # that doesn't already have it on sys.path (agent/loop.py adds it
        # via sys.path.insert -- mirror that here if needed).
        if mcp_server_path:
            import sys
            sys.path.insert(0, mcp_server_path)

    def evaluate(self, state: ProposedPath) -> EnvironmentFeedback:
        data = self._fetch_data(state.student_id)
        courses_by_id = {c["course_id"]: c for c in data["courses"]}
        issues: list[str] = []

        # ------------------------------------------------------------
        # TODO 1 — Prerequisites (done)
        # ------------------------------------------------------------
        issues.extend(self._check_prerequisites(
            state, data["prerequisites"], data["completed_course_ids"]
        ))

        goal = data["learning_goal"]
        if goal is None:
            return EnvironmentFeedback(
                success=False, score=0.0,
                details=["No learning_goal set for this student; cannot evaluate a path."],
            )

        path_courses = [courses_by_id[cid] for cid in state.course_ids if cid in courses_by_id]
        checks_passed = 1 if not issues else 0  # prerequisites check result so far
        total_checks = 6

        # ------------------------------------------------------------
        # TODO 2 — Budget (done)
        # ------------------------------------------------------------
        total_price = sum(c["price"] for c in path_courses)
        if total_price > goal["budget"]:
            issues.append(
                f"Path costs {total_price:.2f}, which exceeds the budget of {goal['budget']:.2f}."
            )
        else:
            checks_passed += 1

        # ------------------------------------------------------------
        # TODO 3 — Weekly hours / date overlap (done)
        # ------------------------------------------------------------
        hours_ok = True
        for i, a in enumerate(path_courses):
            a_start = date.fromisoformat(a["start_date"])
            a_end = date.fromisoformat(a["end_date"])
            overlapping_hours = a["weekly_hours"]
            for j, b in enumerate(path_courses):
                if i == j:
                    continue
                b_start = date.fromisoformat(b["start_date"])
                b_end = date.fromisoformat(b["end_date"])
                if a_start <= b_end and b_start <= a_end:
                    overlapping_hours += b["weekly_hours"]
            if overlapping_hours > goal["weekly_hours_available"]:
                hours_ok = False
                issues.append(
                    f"Course {a['course_id']} ('{a['title']}') overlaps with other courses "
                    f"totalling {overlapping_hours} weekly hours, exceeding the "
                    f"{goal['weekly_hours_available']} hours/week available."
                )
        if hours_ok:
            checks_passed += 1

        # ------------------------------------------------------------
        # TODO 4 — Schedule ordering for dependent courses (done)
        # ------------------------------------------------------------
        required_by_course: dict[int, list[int]] = {}
        for edge in data["prerequisites"]:
            required_by_course.setdefault(edge["course_id"], []).append(edge["prerequisite_course_id"])

        schedule_ok = True
        completed_ids = set(data["completed_course_ids"])
        for course in path_courses:
            for prereq_id in required_by_course.get(course["course_id"], []):
                if prereq_id in completed_ids or prereq_id not in courses_by_id:
                    continue
                prereq = courses_by_id[prereq_id]
                if prereq_id in [c["course_id"] for c in path_courses]:
                    prereq_end = date.fromisoformat(prereq["end_date"])
                    course_start = date.fromisoformat(course["start_date"])
                    if prereq_end > course_start:
                        schedule_ok = False
                        issues.append(
                            f"Course {course['course_id']} starts {course['start_date']}, before "
                            f"its prerequisite {prereq_id} ends ({prereq['end_date']})."
                        )
        if schedule_ok:
            checks_passed += 1

        # ------------------------------------------------------------
        # TODO 5 — Skill coverage (done)
        # ------------------------------------------------------------
        covered_skills: set[str] = set()
        for c in path_courses:
            covered_skills.update(tag.strip() for tag in c["skill_tags"].split(","))
        required_skills = set(data["required_skills"])
        missing_skills = required_skills - covered_skills
        if missing_skills:
            issues.append(
                f"Path does not cover required skills: {', '.join(sorted(missing_skills))}."
            )
        else:
            checks_passed += 1

        # ------------------------------------------------------------
        # TODO 6 — Deadline (done)
        # ------------------------------------------------------------
        if path_courses:
            latest_end = max(date.fromisoformat(c["end_date"]) for c in path_courses)
            target_date = date.fromisoformat(goal["target_date"])
            if latest_end > target_date:
                issues.append(
                    f"Path finishes {latest_end.isoformat()}, after the target date {goal['target_date']}."
                )
            else:
                checks_passed += 1
        # (an empty path trivially meets the deadline but every other
        # check above would already have flagged it as useless)

        # ------------------------------------------------------------
        # TODO 7 — Combine results (done)
        # ------------------------------------------------------------
        success = len(issues) == 0
        score = round(checks_passed / total_checks, 4)
        return EnvironmentFeedback(success=success, score=score, details=issues)

    def _check_prerequisites(
        self,
        state: ProposedPath,
        prerequisites: list[dict],
        completed_course_ids: list[int],
    ) -> list[str]:
        """Every course in the path must have its prerequisites either
        already completed, or scheduled earlier in the same path."""
        issues: list[str] = []
        completed = set(completed_course_ids)

        # Map course_id -> list of its required prerequisite_course_ids
        required_by_course: dict[int, list[int]] = {}
        for edge in prerequisites:
            required_by_course.setdefault(edge["course_id"], []).append(
                edge["prerequisite_course_id"]
            )

        # position of each course in the proposed path (0-indexed).
        # If a course_id is repeated, this keeps its *first* occurrence,
        # which is fine since a repeated course is itself worth flagging.
        position = {cid: i for i, cid in enumerate(state.course_ids)}

        seen = set()
        for i, course_id in enumerate(state.course_ids):
            if course_id in seen:
                issues.append(f"Course {course_id} appears more than once in the path.")
            seen.add(course_id)

            for prereq_id in required_by_course.get(course_id, []):
                if prereq_id in completed:
                    continue
                prereq_position = position.get(prereq_id)
                if prereq_position is None:
                    issues.append(
                        f"Course {course_id} requires course {prereq_id}, "
                        f"which is neither completed nor included in the path."
                    )
                elif prereq_position >= i:
                    issues.append(
                        f"Course {course_id} requires course {prereq_id}, "
                        f"but {prereq_id} is scheduled at or after it in the path."
                    )
        return issues

    def _fetch_data(self, student_id: int) -> dict:
        # Same pattern as agent/loop.py's _handle_db_tool_question:
        # `from server import get_student_profile`. Requires mcp_server/
        # to be on sys.path (pass mcp_server_path to __init__ if it isn't
        # already, e.g. when this file is run standalone rather than
        # through agent/loop.py which adds it itself).
        from server import get_path_planning_data

        result = get_path_planning_data(student_id)
        if result["status"] != "success":
            raise RuntimeError(f"get_path_planning_data failed: {result['message']}")
        return result["data"]


