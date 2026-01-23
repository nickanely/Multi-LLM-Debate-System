from pydantic import BaseModel

class Solution(BaseModel):
    solver_id: str
    reasoning: str
    final_answer: str
    confidence: float