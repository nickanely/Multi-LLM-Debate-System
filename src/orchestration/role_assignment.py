from typing import Dict, List, Optional
from ..schemas.role_assessment import RoleAssessment


def assign_roles_deterministic(
        model_ids: List[str],
        problem_index: int = 0,
        strategy: str = "round_robin"
) -> Dict:
    """
    Deterministic role assignment WITHOUT API calls.

    Strategies:
    - "round_robin": Rotate judge role based on problem index
    - "fixed": First model is always judge
    - "last_judge": Last model is always judge

    Input: List of model IDs, problem index
    Output: {
        "judge": model_id,
        "solvers": [model_id, model_id, model_id]
    }
    """
    if len(model_ids) < 2:
        raise ValueError("Need at least 2 models")

    if strategy == "round_robin":
        # Rotate which model is judge based on problem index
        judge_index = problem_index % len(model_ids)
    elif strategy == "fixed":
        # First model is always judge
        judge_index = 0
    elif strategy == "last_judge":
        # Last model is always judge
        judge_index = len(model_ids) - 1
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    judge_id = model_ids[judge_index]
    solver_ids = [m for i, m in enumerate(model_ids) if i != judge_index]

    return {
        "judge": judge_id,
        "solvers": solver_ids
    }


def assign_roles_from_assessments(role_assessments: List[RoleAssessment]) -> Dict:
    """
    Original API-based role assignment (kept for compatibility).
    Uses model self-assessments to assign roles.

    Input: List of RoleAssessment objects from each model
    Output: {
        "judge": model_id,
        "solvers": [model_id, model_id, model_id]
    }
    """
    judge_scores = {}

    for assessment in role_assessments:
        model_id = assessment.model_id
        judge_conf = assessment.confidence_by_role.get("Judge", 0.5)
        solver_conf = assessment.confidence_by_role.get("Solver", 0.5)
        pref_bonus = 0.1 if assessment.role_preferences[0] == "Judge" else 0
        judge_scores[model_id] = judge_conf - solver_conf + pref_bonus

    sorted_models = sorted(judge_scores.items(), key=lambda x: x[1], reverse=True)
    judge_id = sorted_models[0][0]
    solver_ids = [model_id for model_id, _ in sorted_models[1:]]

    return {
        "judge": judge_id,
        "solvers": solver_ids
    }


# Alias for backward compatibility
def assign_roles(role_assessments: List[RoleAssessment]) -> Dict:
    return assign_roles_from_assessments(role_assessments)

# from ..schemas.role_assessment import RoleAssessment
# from typing import Dict, List
#
#
# def assign_roles(role_assessments: List[RoleAssessment]) -> Dict:
#     """
#     Deterministic algorithm to assign roles based on self-assessments.
#
#     Input: List of RoleAssessment objects from each model
#     Output: {
#         "judge": model_id,
#         "solvers": [model_id, model_id, model_id]
#     }
#     """
#     # Calculate judge score for each model
#     # Higher score = better suited as judge
#     judge_scores = {}
#
#     for assessment in role_assessments:
#         model_id = assessment.model_id
#
#         # Get judge confidence (default 0.5 if not specified)
#         judge_conf = assessment.confidence_by_role.get("Judge", 0.5)
#         solver_conf = assessment.confidence_by_role.get("Solver", 0.5)
#
#         # Preference bonus: if judge is first preference, add bonus
#         pref_bonus = 0.1 if assessment.role_preferences[0] == "Judge" else 0
#
#         # Score combines confidence and preference
#         judge_scores[model_id] = judge_conf - solver_conf + pref_bonus
#
#     # Sort by judge score (descending)
#     sorted_models = sorted(judge_scores.items(), key=lambda x: x[1], reverse=True)
#
#     # Assign judge to highest scoring model
#     judge_id = sorted_models[0][0]
#
#     # Remaining models become solvers
#     solver_ids = [model_id for model_id, _ in sorted_models[1:]]
#
#     return {
#         "judge": judge_id,
#         "solvers": solver_ids
#     }
#
# #
# def assign_roles_round_robin(problem_index: int, model_ids: List[str]) -> Dict:
#     """
#     Alternative: Simple round-robin assignment for baseline comparison.
#     Each model takes turns being the judge.
#     """
#     judge_index = problem_index % len(model_ids)
#     judge_id = model_ids[judge_index]
#     solver_ids = [m for i, m in enumerate(model_ids) if i != judge_index]
#
#     return {
#         "judge": judge_id,
#         "solvers": solver_ids
#     }