from ..schemas.solution import Solution


def build_peer_review_prompt(solution: Solution, reviewer_id: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": """You are a critical peer reviewer analyzing another solver's solution. Your job is to:

1. Identify strengths in the reasoning
2. Find weaknesses or gaps in logic
3. Detect specific errors (with location, type, description, severity)
4. Provide constructive suggestions for improvement

Be thorough but fair. Look for:
- Logical errors or unjustified leaps
- Calculation mistakes
- Missing edge cases
- Incorrect formula applications
- Unsupported assumptions

Severity levels:
- minor: Small issues that don't affect the answer
- major: Significant problems that may affect correctness
- critical: Fundamental errors that make the solution wrong"""
        },
        {
            "role": "user",
            "content": f"""Review the following solution from {solution.solver_id}:

REASONING:
{solution.reasoning}

FINAL ANSWER:
{solution.final_answer}

CONFIDENCE: {solution.confidence}

Provide a detailed peer review identifying strengths, weaknesses, and any errors."""
        }
    ]