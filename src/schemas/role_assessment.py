from pydantic import BaseModel
from typing import Dict, List


class RoleAssessment(BaseModel):
    model_id: str
    role_preferences: List[str]
    confidence_by_role: Dict[str, float]
    reasoning: str