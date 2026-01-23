from ..schemas.solution import Solution
from ..schemas.review import PeerReview


def build_refinement_prompt(
    original_solution: Solution,
    reviews: list[PeerReview]
) -> list[dict]:
    reviews_text = ""
    for i, review in enumerate(reviews, 1):
        reviews_text += f"""
--- Review {i} from {review.reviewer_id} ---
Overall Assessment: {review.overall_assessment}

Strengths Identified:
{chr(10).join(f'- {s}' for s in review.strengths)}

Weaknesses Identified:
{chr(10).join(f'- {w}' for w in review.weaknesses)}

Errors Found:
"""
        for error in review.errors:
            reviews_text += f"""
  Location: {error.location}
  Type: {error.error_type}
  Severity: {error.severity}
  Description: {error.description}
"""
        reviews_text += "\n"

    return [
        {
            "role": "system",
            "content": """You are refining your solution based on peer feedback. For each critique:

1. If the critique is VALID: Accept it and modify your solution
2. If the critique is INVALID: Explain why and defend your original reasoning

You must:
- Address every critique explicitly
- Show what changes you made (or why you didn't)
- Provide a complete refined solution
- Update your confidence based on the feedback received"""
        },
        {
            "role": "user",
            "content": f"""Your original solution:

REASONING:
{original_solution.reasoning}

FINAL ANSWER:
{original_solution.final_answer}

CONFIDENCE: {original_solution.confidence}

---

PEER REVIEWS RECEIVED:
{reviews_text}

---

Now refine your solution by addressing each piece of feedback. For each critique, explicitly state whether you accept or reject it and why. Then provide your complete refined solution."""
        }
    ]