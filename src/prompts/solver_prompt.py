from ..schemas.problem import Problem


def build_solver_prompt(problem: Problem) -> list[dict]:
    return [
        {
            "role": "system",
            "content": """You are an expert problem solver. Analyze problems methodically and provide clear, step-by-step reasoning.

Your response must include:
1. A detailed reasoning process showing all your work
2. A clear final answer
3. A confidence score (0.0-1.0) indicating how certain you are

Be thorough in your analysis and show all intermediate steps."""
        },
        {
            "role": "user",
            "content": f"""Solve the following problem step by step.

Category: {problem.category}

Problem:
{problem.question}

Provide your complete reasoning and final answer."""
        }
    ]


def build_role_assessment_prompt(problem: Problem) -> list[dict]:
    return [
        {
            "role": "system",
            "content": """You are assessing which role you would be best suited for in a collaborative problem-solving system.

Roles available:
- Solver: Independently solve the problem with detailed reasoning
- Judge: Evaluate multiple solutions and determine the best one

Consider your strengths in:
- Mathematical/logical reasoning
- Critical analysis and evaluation
- Domain expertise for this problem type"""
        },
        {
            "role": "user",
            "content": f"""Given this problem, assess your suitability for each role.

Problem Category: {problem.category}

Problem:
{problem.question}

Provide your role preferences and confidence levels for being a Solver vs Judge for this specific problem."""
        }
    ]