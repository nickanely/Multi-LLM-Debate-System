from pydantic import BaseModel

class Problem(BaseModel):
    problem_id: str
    category: str
    question: str