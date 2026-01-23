import json
from pathlib import Path
from typing import List, Dict
from src.schemas.problem import Problem


def load_problems(filepath: str = None) -> List[Problem]:
    """Load problems from JSON file."""
    if filepath is None:
        filepath = Path(__file__).parent / "problems.json"

    with open(filepath, 'r') as f:
        data = json.load(f)

    return [Problem(**p) for p in data]


def load_ground_truth(filepath: str = None) -> Dict[str, str]:
    """Load ground truth answers from JSON file."""
    if filepath is None:
        filepath = Path(__file__).parent / "problems.json"

    with open(filepath, 'r') as f:
        data = json.load(f)

    return {p["problem_id"]: p["correct_answer"] for p in data}