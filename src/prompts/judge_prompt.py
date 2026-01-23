from ..schemas.solution import Solution
from ..schemas.refinement import RefinedSolution
from ..schemas.review import PeerReview
from typing import Union


def build_judge_prompt(
        original_solutions: list[Solution],
        reviews: list[PeerReview],
        refined_solutions: list[RefinedSolution]
) -> list[dict]:
    # Format original solutions
    original_text = ""
    for sol in original_solutions:
        original_text += f"""
=== Original Solution from {sol.solver_id} ===
Reasoning: {sol.reasoning}
Final Answer: {sol.final_answer}
Confidence: {sol.confidence}
"""

    # Format reviews
    reviews_text = ""
    for rev in reviews:
        reviews_text += f"""
--- Review by {rev.reviewer_id} of {rev.solution_id} ---
Assessment: {rev.overall_assessment}
Strengths: {', '.join(rev.strengths)}
Weaknesses: {', '.join(rev.weaknesses)}
Errors: {len(rev.errors)} found (severities: {', '.join(e.severity for e in rev.errors) if rev.errors else 'none'})
"""

    # Format refined solutions
    refined_text = ""
    for sol in refined_solutions:
        changes_summary = []
        for change in sol.changes_made:
            status = "ACCEPTED" if change.accepted else "REJECTED"
            changes_summary.append(f"  - {change.critique}: {status}")

        refined_text += f"""
=== Refined Solution from {sol.solver_id} ===
Changes Made:
{chr(10).join(changes_summary) if changes_summary else '  None'}

Refined Reasoning: {sol.refined_reasoning}
Refined Answer: {sol.refined_answer}
Final Confidence: {sol.confidence}
"""

    return [
        {
            "role": "system",
            "content": """You are the final judge in a multi-LLM debate system. Your task is to:

1. Analyze all original solutions and their reasoning
2. Consider the peer reviews and identified issues
3. Evaluate how each solver responded to criticism in their refined solutions
4. Determine which final answer is most likely correct

Focus on:
- Correctness of reasoning and methodology
- How well errors were addressed during refinement
- Consistency and completeness of solutions
- Appropriate confidence calibration

Select the winner based on the REFINED solutions, but use the full context to inform your decision."""
        },
        {
            "role": "user",
            "content": f"""ORIGINAL SOLUTIONS:
{original_text}

PEER REVIEWS:
{reviews_text}

REFINED SOLUTIONS:
{refined_text}

---

Based on all the evidence above, determine which solver provided the best final answer. Explain your reasoning thoroughly."""
        }
    ]