from typing import Dict, List, Any
import asyncio
import time

from ..schemas.problem import Problem
from ..schemas.solution import Solution
from ..schemas.review import PeerReview
from ..schemas.refinement import RefinedSolution
from ..schemas.judgment import Judgment
from ..agents.solver import SolverAgent
from ..agents.judge import JudgeAgent
from .role_assignment import assign_roles_deterministic


class DebateRunner:
    def __init__(
            self,
            models: Dict[str, Any],
            verbose: bool = True,
            role_strategy: str = "round_robin"
    ):
        """
        Initialize DebateRunner.

        Args:
            models: Dict mapping model_id -> model instance
            verbose: Whether to print progress logs
            role_strategy: Strategy for role assignment
                          ("round_robin", "fixed", "last_judge")
        """
        self.models = models
        self.model_ids = list(models.keys())
        self.verbose = verbose
        self.role_strategy = role_strategy

    def log(self, message: str):
        if self.verbose:
            print(f"[DebateRunner] {message}")

    async def run_async(self, problem: Problem, problem_index: int = 0) -> Dict:
        """
        Execute the full debate pipeline asynchronously.

        API calls per problem (with async parallelization):
        - Stage 1: 3 calls in parallel (solvers)
        - Stage 2: 6 calls in parallel (reviews)
        - Stage 3: 3 calls in parallel (refinements)
        - Stage 4: 1 call (judge)
        Total: 13 API calls, but only 4 sequential batches

        Returns all artifacts including timing info.
        """
        results = {
            "problem_id": problem.problem_id,
            "problem": problem.model_dump(),
            "timing": {}
        }

        # Stage 0: Deterministic Role Assignment (NO API CALLS)
        self.log("Stage 0: Assigning roles (deterministic)...")
        start = time.time()

        role_assignment = assign_roles_deterministic(
            self.model_ids,
            problem_index=problem_index,
            strategy=self.role_strategy
        )
        results["role_assignment"] = role_assignment
        results["timing"]["role_assignment"] = time.time() - start

        self.log(f"  Judge: {role_assignment['judge']}")
        self.log(f"  Solvers: {role_assignment['solvers']}")

        # Create agents
        solver_agents = [
            SolverAgent(solver_id, self.models[solver_id])
            for solver_id in role_assignment["solvers"]
        ]
        judge_agent = JudgeAgent(
            role_assignment["judge"],
            self.models[role_assignment["judge"]]
        )

        # Stage 1: Parallel Solution Generation
        self.log("Stage 1: Generating solutions in parallel...")
        start = time.time()

        async def solve_with_fallback(agent: SolverAgent) -> Solution:
            try:
                return await agent.asolve(problem)
            except Exception as e:
                self.log(f"  Error from {agent.solver_id}: {e}")
                return Solution(
                    solver_id=agent.solver_id,
                    reasoning=f"Error generating solution: {e}",
                    final_answer="ERROR",
                    confidence=0.0
                )

        solutions = await asyncio.gather(*[
            solve_with_fallback(agent) for agent in solver_agents
        ])
        solutions = list(solutions)

        results["initial_solutions"] = [s.model_dump() for s in solutions]
        results["timing"]["solution_generation"] = time.time() - start
        self.log(f"  Completed in {results['timing']['solution_generation']:.2f}s")

        # Stage 2: Parallel Peer Review
        self.log("Stage 2: Peer review round in parallel...")
        start = time.time()

        async def review_with_fallback(
                reviewer: SolverAgent,
                solution: Solution
        ) -> PeerReview:
            try:
                return await reviewer.areview(solution)
            except Exception as e:
                self.log(f"  Error: {reviewer.solver_id} reviewing {solution.solver_id}: {e}")
                return PeerReview(
                    reviewer_id=reviewer.solver_id,
                    solution_id=solution.solver_id,
                    strengths=["Unable to review"],
                    weaknesses=[str(e)],
                    errors=[],
                    overall_assessment="flawed"
                )

        # Create all review tasks
        review_tasks = []
        for i, reviewer_agent in enumerate(solver_agents):
            for j, solution in enumerate(solutions):
                if i != j:
                    review_tasks.append(
                        review_with_fallback(reviewer_agent, solution)
                    )

        reviews = await asyncio.gather(*review_tasks)
        reviews = list(reviews)

        results["peer_reviews"] = [r.model_dump() for r in reviews]
        results["timing"]["peer_review"] = time.time() - start
        self.log(f"  Completed {len(reviews)} reviews in {results['timing']['peer_review']:.2f}s")

        # Stage 3: Parallel Refinement
        self.log("Stage 3: Refining solutions in parallel...")
        start = time.time()

        async def refine_with_fallback(
                agent: SolverAgent,
                original: Solution,
                agent_reviews: List[PeerReview]
        ) -> RefinedSolution:
            try:
                return await agent.arefine(original, agent_reviews)
            except Exception as e:
                self.log(f"  Error refining {agent.solver_id}: {e}")
                return RefinedSolution(
                    solver_id=agent.solver_id,
                    changes_made=[],
                    refined_reasoning=original.reasoning,
                    refined_answer=original.final_answer,
                    confidence=original.confidence * 0.8
                )

        refine_tasks = []
        for i, agent in enumerate(solver_agents):
            solver_reviews = [r for r in reviews if r.solution_id == agent.solver_id]
            refine_tasks.append(
                refine_with_fallback(agent, solutions[i], solver_reviews)
            )

        refined_solutions = await asyncio.gather(*refine_tasks)
        refined_solutions = list(refined_solutions)

        results["refined_solutions"] = [rs.model_dump() for rs in refined_solutions]
        results["timing"]["refinement"] = time.time() - start
        self.log(f"  Completed in {results['timing']['refinement']:.2f}s")

        # Stage 4: Final Judgment
        self.log("Stage 4: Final judgment...")
        start = time.time()

        try:
            judgment = await judge_agent.ajudge(solutions, reviews, refined_solutions)
        except Exception as e:
            self.log(f"  Error in judgment: {e}")
            best_solution = max(refined_solutions, key=lambda x: x.confidence)
            judgment = Judgment(
                winner_solver_id=best_solution.solver_id,
                reasoning=f"Default selection due to judge error: {e}",
                confidence=0.5
            )

        results["judgment"] = judgment.model_dump()
        results["timing"]["judgment"] = time.time() - start

        # Get final answer
        winner_solution = next(
            (s for s in refined_solutions if s.solver_id == judgment.winner_solver_id),
            refined_solutions[0]
        )
        results["final_answer"] = winner_solution.refined_answer

        # Calculate total time
        results["timing"]["total"] = sum(results["timing"].values())

        self.log(f"Final answer: {results['final_answer']}")
        self.log(f"Winner: {judgment.winner_solver_id} (confidence: {judgment.confidence})")
        self.log(f"Total time: {results['timing']['total']:.2f}s")

        return results

    def run(self, problem: Problem, problem_index: int = 0) -> Dict:
        """
        Synchronous wrapper for run_async.
        """
        return asyncio.run(self.run_async(problem, problem_index))

    async def run_batch_async(self, problems: List[Problem]) -> List[Dict]:
        """
        Run multiple problems. Problems are processed sequentially,
        but each problem uses parallel API calls internally.
        """
        results = []
        for i, problem in enumerate(problems):
            self.log(f"\n{'=' * 50}")
            self.log(f"Problem {i + 1}/{len(problems)}: {problem.problem_id}")
            self.log(f"{'=' * 50}")
            result = await self.run_async(problem, problem_index=i)
            results.append(result)
        return results

    def run_batch(self, problems: List[Problem]) -> List[Dict]:
        """
        Synchronous wrapper for run_batch_async.
        """
        return asyncio.run(self.run_batch_async(problems))