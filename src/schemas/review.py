from pydantic import BaseModel
from typing import List, Literal

class ReviewError(BaseModel):
    location: str
    error_type: str
    description: str
    severity: Literal["minor", "major", "critical"]

class PeerReview(BaseModel):
    reviewer_id: str
    solution_id: str
    strengths: List[str]
    weaknesses: List[str]
    errors: List[ReviewError]
    overall_assessment: Literal[
        "correct", "minor_issues", "flawed", "incorrect"
    ]