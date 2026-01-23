from ..models.base_model import BaseModel
from ..schemas.solution import Solution
from ..schemas.review import PeerReview
from ..schemas.judgment import Judgment
from ..schemas.refinement import RefinedSolution
from ..schemas.role_assessment import RoleAssessment
from ..schemas.problem import Problem
from ..prompts.judge_prompt import build_judge_prompt
from ..prompts.solver_prompt import build_role_assessment_prompt


class JudgeAgent:
    def __init__(self, judge_id: str, model: BaseModel):
        self.judge_id = judge_id
        self.model = model

    # ===== SYNCHRONOUS METHODS =====

    def assess_role(self, problem: Problem) -> RoleAssessment:
        """Assess which role this model is best suited for."""
        messages = build_role_assessment_prompt(problem)
        result = self.model.generate(
            messages=messages,
            response_model=RoleAssessment,
        )
        return result

    def judge(
            self,
            original_solutions: list[Solution],
            reviews: list[PeerReview],
            refined_solutions: list[RefinedSolution]
    ) -> Judgment:
        """Evaluate all solutions and select the best one."""
        messages = build_judge_prompt(original_solutions, reviews, refined_solutions)
        result = self.model.generate(
            messages=messages,
            response_model=Judgment,
        )
        return result

    # ===== ASYNCHRONOUS METHODS =====

    async def ajudge(
            self,
            original_solutions: list[Solution],
            reviews: list[PeerReview],
            refined_solutions: list[RefinedSolution]
    ) -> Judgment:
        """Async: Evaluate all solutions and select the best one."""
        messages = build_judge_prompt(original_solutions, reviews, refined_solutions)
        result = await self.model.agenerate(
            messages=messages,
            response_model=Judgment,
        )
        return result