from pydantic import BaseModel
from typing import List

class ChangeLog(BaseModel):
    critique: str
    response: str
    accepted: bool

class RefinedSolution(BaseModel):
    solver_id: str
    changes_made: List[ChangeLog]
    refined_reasoning: str
    refined_answer: str
    confidence: float
