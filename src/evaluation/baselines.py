from typing import Dict, List, Any
import asyncio

from ..schemas.problem import Problem
from ..schemas.solution import Solution
from ..models.base_model import BaseModel
from ..prompts.solver_prompt import build_solver_prompt
from .metrics import answers_match, normalize_answer


async def single_shot_async(
        model: BaseModel,
        model_id: str,
        problem: Problem
) -> Dict:
    """
    Async baseline: Single LLM attempt without debate.
    """
    messages = build_solver_prompt(problem)

    try:
        solution = await model.agenerate(
            messages=messages,
            response_model=Solution
        )
        return {
            "model_id": model_id,
            "problem_id": problem.problem_id,
            "answer": solution.final_answer,
            "confidence": solution.confidence,
            "reasoning": solution.reasoning
        }
    except Exception as e:
        return {
            "model_id": model_id,
            "problem_id": problem.problem_id,
            "answer": "ERROR",
            "confidence": 0.0,
            "reasoning": str(e)
        }


def single_shot(model: BaseModel, model_id: str, problem: Problem) -> Dict:
    """Sync wrapper for single_shot_async."""
    return asyncio.run(single_shot_async(model, model_id, problem))


def majority_vote_from_results(single_shot_results: List[Dict]) -> Dict:
    """Compute majority vote from single shot results."""
    answers = [r["answer"] for r in single_shot_results]

    answer_groups = {}
    for answer in answers:
        found_group = False
        for group_answer in answer_groups:
            if answers_match(answer, group_answer):
                answer_groups[group_answer].append(answer)
                found_group = True
                break
        if not found_group:
            answer_groups[answer] = [answer]

    majority_answer = max(answer_groups.keys(), key=lambda x: len(answer_groups[x]))
    vote_count = len(answer_groups[majority_answer])

    return {
        "answer": majority_answer,
        "vote_count": vote_count,
        "total_votes": len(single_shot_results),
        "all_answers": answers,
        "individual_solutions": single_shot_results
    }


async def run_all_baselines_async(
        models: Dict[str, BaseModel],
        problems: List[Problem],
        ground_truth: Dict[str, str],
        verbose: bool = True
) -> Dict[str, Any]:
    """
    Run all baseline methods with async parallelization.

    For each problem, all models run in parallel.
    """
    results = {
        "single_shot": {model_id: [] for model_id in models},
        "majority_vote": [],
        "accuracy": {
            "single_shot": {},
            "majority_vote": 0.0
        }
    }

    for i, problem in enumerate(problems):
        if verbose:
            print(f"[Baselines] Problem {i + 1}/{len(problems)}: {problem.problem_id}")

        # Run all models in parallel for this problem
        tasks = [
            single_shot_async(model, model_id, problem)
            for model_id, model in models.items()
        ]
        problem_results = await asyncio.gather(*tasks)

        # Store results
        for result in problem_results:
            model_id = result["model_id"]
            results["single_shot"][model_id].append(result)

        # Compute majority vote
        vote_result = majority_vote_from_results(list(problem_results))
        vote_result["problem_id"] = problem.problem_id
        results["majority_vote"].append(vote_result)

    # Calculate accuracies
    for model_id in models:
        model_results = results["single_shot"][model_id]
        correct = sum(
            1 for r in model_results
            if answers_match(r["answer"], ground_truth.get(r["problem_id"], ""))
        )
        results["accuracy"]["single_shot"][model_id] = correct / len(problems) if problems else 0.0

    correct = sum(
        1 for r in results["majority_vote"]
        if answers_match(r["answer"], ground_truth.get(r["problem_id"], ""))
    )
    results["accuracy"]["majority_vote"] = correct / len(problems) if problems else 0.0

    return results


def run_all_baselines(
        models: Dict[str, BaseModel],
        problems: List[Problem],
        ground_truth: Dict[str, str],
        verbose: bool = True
) -> Dict[str, Any]:
    """Sync wrapper for run_all_baselines_async."""
    return asyncio.run(run_all_baselines_async(models, problems, ground_truth, verbose))