from ..models.base_model import BaseModel
from ..schemas.problem import Problem
from ..schemas.solution import Solution
from ..schemas.review import PeerReview
from ..schemas.refinement import RefinedSolution
from ..schemas.role_assessment import RoleAssessment
from ..prompts.solver_prompt import build_solver_prompt, build_role_assessment_prompt
from ..prompts.review_prompt import build_peer_review_prompt
from ..prompts.refinement_prompt import build_refinement_prompt


class SolverAgent:
    def __init__(self, solver_id: str, model: BaseModel):
        self.solver_id = solver_id
        self.model = model

    # ===== SYNCHRONOUS METHODS =====

    def assess_role(self, problem: Problem) -> RoleAssessment:
        """Assess which role this model is best suited for."""
        messages = build_role_assessment_prompt(problem)
        result = self.model.generate(
            messages=messages,
            response_model=RoleAssessment,
        )
        result.model_id = self.solver_id
        return result

    def solve(self, problem: Problem) -> Solution:
        """Generate an independent solution to the problem."""
        messages = build_solver_prompt(problem)
        result = self.model.generate(
            messages=messages,
            response_model=Solution,
        )
        result.solver_id = self.solver_id
        return result

    def review(self, peer_solution: Solution) -> PeerReview:
        """Review another solver's solution."""
        messages = build_peer_review_prompt(peer_solution, self.solver_id)
        result = self.model.generate(
            messages=messages,
            response_model=PeerReview,
        )
        result.reviewer_id = self.solver_id
        result.solution_id = peer_solution.solver_id
        return result

    def refine(
            self,
            original_solution: Solution,
            reviews: list[PeerReview]
    ) -> RefinedSolution:
        """Refine solution based on peer feedback."""
        messages = build_refinement_prompt(original_solution, reviews)
        result = self.model.generate(
            messages=messages,
            response_model=RefinedSolution,
        )
        result.solver_id = self.solver_id
        return result

    # ===== ASYNCHRONOUS METHODS =====

    async def asolve(self, problem: Problem) -> Solution:
        """Async: Generate an independent solution to the problem."""
        messages = build_solver_prompt(problem)
        result = await self.model.agenerate(
            messages=messages,
            response_model=Solution,
        )
        result.solver_id = self.solver_id
        return result

    async def areview(self, peer_solution: Solution) -> PeerReview:
        """Async: Review another solver's solution."""
        messages = build_peer_review_prompt(peer_solution, self.solver_id)
        result = await self.model.agenerate(
            messages=messages,
            response_model=PeerReview,
        )
        result.reviewer_id = self.solver_id
        result.solution_id = peer_solution.solver_id
        return result

    async def arefine(
            self,
            original_solution: Solution,
            reviews: list[PeerReview]
    ) -> RefinedSolution:
        """Async: Refine solution based on peer feedback."""
        messages = build_refinement_prompt(original_solution, reviews)
        result = await self.model.agenerate(
            messages=messages,
            response_model=RefinedSolution,
        )
        result.solver_id = self.solver_id
        return result