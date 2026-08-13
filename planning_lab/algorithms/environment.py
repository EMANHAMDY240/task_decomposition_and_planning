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
# TODO: import whatever client you use to call the MCP server tool,
# e.g. from your agent/ package's MCP client wrapper.


@dataclass
class ProposedPath:
    """The 'state' this environment evaluates: an ordered list of course_ids
    the agent is proposing for one student, plus which student it's for."""
    student_id: int
    course_ids: list[int]  # in the order the agent proposes to take them


class LearningPathEnvironment:
    """Grounded evaluator: checks a proposed learning path against real
    BrightPeak data (courses, prerequisites, learning_goals) via the
    get_path_planning_data MCP tool. No LLM call, no randomness."""

    def __init__(self, mcp_client):
        # TODO: store whatever object lets you call
        # get_path_planning_data(student_id) and get back its `data` dict.
        self.mcp_client = mcp_client

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

        # ------------------------------------------------------------
        # TODO 2 — Budget
        # Sum courses_by_id[cid]["price"] for cid in state.course_ids.
        # Compare against data["learning_goal"]["budget"].
        # ------------------------------------------------------------

        # ------------------------------------------------------------
        # TODO 3 — Weekly hours (needs date overlap logic)
        # For any two courses in the path whose [start_date, end_date]
        # ranges overlap, sum their weekly_hours and compare against
        # data["learning_goal"]["weekly_hours_available"].
        # Hint: parse start_date/end_date with datetime.date.fromisoformat,
        # two ranges [a_start,a_end] and [b_start,b_end] overlap when
        # a_start <= b_end and b_start <= a_end.
        # ------------------------------------------------------------

        # ------------------------------------------------------------
        # TODO 4 — Schedule ordering for dependent courses
        # If course B depends on course A (via prerequisites) and both are
        # in the path (not already completed), A's end_date must be <=
        # B's start_date.
        # ------------------------------------------------------------

        # ------------------------------------------------------------
        # TODO 5 — Skill coverage
        # Union all skill_tags (comma-separated string -> set) across the
        # courses in the path. Every tag in data["required_skills"] must
        # appear in that union.
        # ------------------------------------------------------------

        # ------------------------------------------------------------
        # TODO 6 — Deadline
        # The max end_date among courses in the path must be <=
        # data["learning_goal"]["target_date"].
        # ------------------------------------------------------------

        # ------------------------------------------------------------
        # TODO 7 — Combine results
        # success = (len(issues) == 0)
        # score = fraction of the 6 checks above that passed (e.g. 5/6 = 0.83)
        # Return EnvironmentFeedback(success=success, score=score, details=issues)
        # ------------------------------------------------------------
        raise NotImplementedError("Fill in TODOs 1-7 above")

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
        # TODO: call your MCP tool get_path_planning_data(student_id) here
        # and return its ["data"] dict.
        raise NotImplementedError("Wire this to your MCP client")

