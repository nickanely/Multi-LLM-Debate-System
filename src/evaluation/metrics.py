from typing import List, Dict, Any
import numpy as np


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    if answer is None:
        return ""
    # Convert to lowercase, strip whitespace
    answer = str(answer).lower().strip()
    # Remove common prefixes
    for prefix in ["the answer is", "answer:", "final answer:"]:
        if answer.startswith(prefix):
            answer = answer[len(prefix):].strip()
    # Remove punctuation at the end
    answer = answer.rstrip(".,;:")
    return answer


def answers_match(answer1: str, answer2: str) -> bool:
    """Check if two answers match (with normalization)."""
    norm1 = normalize_answer(answer1)
    norm2 = normalize_answer(answer2)

    # Exact match
    if norm1 == norm2:
        return True

    # Try numeric comparison
    try:
        num1 = float(norm1.replace(",", ""))
        num2 = float(norm2.replace(",", ""))
        return abs(num1 - num2) < 1e-6
    except ValueError:
        pass

    # Check if one contains the other (for longer answers)
    if len(norm1) > 5 and len(norm2) > 5:
        if norm1 in norm2 or norm2 in norm1:
            return True

    return False


def accuracy(results: List[Dict], ground_truth: Dict[str, str]) -> float:
    """
    Calculate accuracy of system answers against ground truth.

    Args:
        results: List of debate results
        ground_truth: Dict mapping problem_id -> correct_answer

    Returns:
        Accuracy as float between 0 and 1
    """
    if not results:
        return 0.0

    correct = 0
    for result in results:
        problem_id = result.get("problem_id")
        system_answer = result.get("final_answer", "")
        true_answer = ground_truth.get(problem_id, "")

        if answers_match(system_answer, true_answer):
            correct += 1

    return correct / len(results)


def improvement_rate(results: List[Dict], ground_truth: Dict[str, str]) -> float:
    """
    Calculate rate of problems where refinement improved the answer.

    Compares initial solutions to refined solutions.
    """
    if not results:
        return 0.0

    improvements = 0
    applicable = 0

    for result in results:
        problem_id = result.get("problem_id")
        true_answer = ground_truth.get(problem_id, "")

        # Get winner's initial and refined answers
        winner_id = result.get("judgment", {}).get("winner_solver_id")

        initial_solutions = result.get("initial_solutions", [])
        refined_solutions = result.get("refined_solutions", [])

        winner_initial = next(
            (s for s in initial_solutions if s.get("solver_id") == winner_id),
            None
        )
        winner_refined = next(
            (s for s in refined_solutions if s.get("solver_id") == winner_id),
            None
        )

        if winner_initial and winner_refined:
            applicable += 1
            initial_correct = answers_match(
                winner_initial.get("final_answer", ""),
                true_answer
            )
            refined_correct = answers_match(
                winner_refined.get("refined_answer", ""),
                true_answer
            )

            # Improvement: was wrong, now correct
            if not initial_correct and refined_correct:
                improvements += 1

    return improvements / applicable if applicable > 0 else 0.0


def consensus_rate(results: List[Dict]) -> float:
    """
    Calculate rate of problems where all solvers reached the same answer.
    """
    if not results:
        return 0.0

    consensus_count = 0

    for result in results:
        refined_solutions = result.get("refined_solutions", [])
        if len(refined_solutions) < 2:
            continue

        answers = [s.get("refined_answer", "") for s in refined_solutions]

        # Check if all answers match
        all_match = all(
            answers_match(answers[0], answer)
            for answer in answers[1:]
        )

        if all_match:
            consensus_count += 1

    return consensus_count / len(results)


def judge_accuracy(results: List[Dict], ground_truth: Dict[str, str]) -> Dict:
    """
    Analyze judge accuracy when solvers disagree.

    Returns dict with:
    - total_disagreements: Number of problems where solvers disagreed
    - judge_picked_correct: When disagreeing, how often judge picked correct answer
    - judge_accuracy: Ratio of correct picks
    """
    disagreements = 0
    judge_correct = 0

    for result in results:
        problem_id = result.get("problem_id")
        true_answer = ground_truth.get(problem_id, "")

        refined_solutions = result.get("refined_solutions", [])
        if len(refined_solutions) < 2:
            continue

        answers = [s.get("refined_answer", "") for s in refined_solutions]

        # Check if there's disagreement
        has_disagreement = not all(
            answers_match(answers[0], answer)
            for answer in answers[1:]
        )

        if has_disagreement:
            disagreements += 1

            # Check if judge picked the correct one
            winner_id = result.get("judgment", {}).get("winner_solver_id")
            winner_solution = next(
                (s for s in refined_solutions if s.get("solver_id") == winner_id),
                None
            )

            if winner_solution:
                winner_answer = winner_solution.get("refined_answer", "")
                if answers_match(winner_answer, true_answer):
                    judge_correct += 1

    return {
        "total_disagreements": disagreements,
        "judge_picked_correct": judge_correct,
        "judge_accuracy": judge_correct / disagreements if disagreements > 0 else 1.0
    }


def per_solver_accuracy(results: List[Dict], ground_truth: Dict[str, str]) -> Dict[str, float]:
    """Calculate accuracy for each solver's refined solutions."""
    solver_correct = {}
    solver_total = {}

    for result in results:
        problem_id = result.get("problem_id")
        true_answer = ground_truth.get(problem_id, "")

        for solution in result.get("refined_solutions", []):
            solver_id = solution.get("solver_id")
            answer = solution.get("refined_answer", "")

            if solver_id not in solver_correct:
                solver_correct[solver_id] = 0
                solver_total[solver_id] = 0

            solver_total[solver_id] += 1
            if answers_match(answer, true_answer):
                solver_correct[solver_id] += 1

    return {
        solver_id: solver_correct[solver_id] / solver_total[solver_id]
        for solver_id in solver_total
    }


def calculate_all_metrics(
        results: List[Dict],
        ground_truth: Dict[str, str]
) -> Dict[str, Any]:
    """Calculate all metrics for the system."""
    return {
        "overall_accuracy": accuracy(results, ground_truth),
        "improvement_rate": improvement_rate(results, ground_truth),
        "consensus_rate": consensus_rate(results),
        "judge_metrics": judge_accuracy(results, ground_truth),
        "per_solver_accuracy": per_solver_accuracy(results, ground_truth)
    }