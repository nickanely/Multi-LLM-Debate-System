# Multi-LLM Debate System

A modular Python framework that orchestrates structured debates between multiple large language models to synthesise consensus answers. Four models from different providers solve, critique and refine each other's answers to reasoning problems, and one model acts as judge. The framework includes experiment pipelines that run the debate alongside single-model and majority-vote baselines, and tracks the results of every run.

## Debate pipeline

For each problem, one model is assigned the judge role and the others act as solvers:

| Stage | What happens | API calls |
|---|---|---|
| 0. Role assignment | Judge and solvers are assigned deterministically, with no API call | 0 |
| 1. Solve | Each solver answers independently | 3 (parallel) |
| 2. Peer review | Each solver reviews the other two solutions | 6 (parallel) |
| 3. Refine | Each solver revises its answer using the reviews | 3 (parallel) |
| 4. Judge | The judge selects the winning refined solution | 1 |

- **Parallel execution:** the 13 API calls per problem run as four sequential batches with `asyncio`, so stages run in parallel wherever possible.
- **Structured outputs:** every stage returns data validated by Pydantic schemas: problem, solution, peer review, refinement and judgment.
- **Role strategies:** `round_robin` rotates the judge across problems, `fixed` always uses the first model, and `last_judge` always uses the last. An alternative assignment based on the models' own role self-assessments is also included.

### Models

| ID | Provider | Model |
|---|---|---|
| `gpt` | OpenAI | `gpt-5-nano` |
| `claude` | Anthropic | `claude-haiku-4-5` |
| `gemini` | Google | `gemini-2.5-flash-lite` |
| `grok` | xAI | `grok-4-fast-non-reasoning` |

New providers can be added through the common model interface in `src/models/base_model.py`.

## Experiment pipeline and result tracking

The benchmark has 25 problems with known answers across four categories: mathematical/logical reasoning, logic puzzles and constraint satisfaction, strategic game theory, and physics and scientific reasoning.

Each run of `experiments/run_full_system.py`:
1. **runs the baselines:** each model answering alone (single-shot) and majority voting across models, in parallel;
2. **runs the debate pipeline** on every problem;
3. **computes metrics:** overall accuracy, per-solver accuracy, consensus rate, how often refinement corrects an answer, and judge accuracy when solvers disagree;
4. **generates plots:** accuracy comparison, per-category accuracy, solver performance, improvement analysis, consensus distribution and timing breakdown;
5. **saves everything** (baseline results, full debate transcripts, metrics, summary and plots) to `results/experiment_<timestamp>/`.

## Repository structure

```
├── data/
│   ├── problems.json          # Problems with ground-truth answers
│   └── loader.py
├── src/
│   ├── agents/                # SolverAgent, JudgeAgent
│   ├── models/                # Provider wrappers: OpenAI, Anthropic, Google, xAI
│   ├── orchestration/         # DebateRunner, role assignment
│   ├── prompts/               # Solver, review, refinement and judge prompts
│   ├── schemas/               # Pydantic schemas for each stage
│   └── evaluation/            # Metrics, baselines, plots
├── experiments/
│   └── run_full_system.py     # Full experiment: baselines, debate, metrics, plots
├── results/                   # One folder per experiment run
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/nickanely/Multi-LLM-Debate-System.git
cd Multi-LLM-Debate-System
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the project root. At least two providers are required; models without a key are skipped.

```
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...
XAI_API_KEY=...
```

## Usage

```bash
python experiments/run_full_system.py
```

To add your own problems, add entries to `data/problems.json` with a `problem_id`, `category`, `question` and `correct_answer`.

## Authors

Ana Avkopashvili (avkopashvili.ana@kiu.edu.ge) · Nikoloz Aneli (aneli.nikoloz@kiu.edu.ge)
