from pydantic import BaseModel

class Judgment(BaseModel):
    winner_solver_id: str
    reasoning: str
    confidence: float
