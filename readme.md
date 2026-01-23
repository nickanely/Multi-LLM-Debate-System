# Multi-LLM Debate System

A modular Python framework to orchestrate debates between **multiple large language models (LLMs)** and synthesize agreed results. The system runs structured multi-agent debates and produces consolidated outputs, enabling enhanced reasoning and answer quality compared to single LLM inference.

This repository provides tools for experiment pipelines, data handling, model orchestration, and result tracking.

## 🚀 Features

- 🧠 **Multiple LLM Debate Support** – Run debates between two or more LLMs.
- ⚙️ **Modular Architecture** – Clean separation of orchestration, data, and experiments.
- 📊 **Experiment & Results Tracking** – Scripts to run and collect debate outputs.
- 💡 **Extensible Agents** – Plug in new LLM backends or debate strategies.

> Multi-LLM debate systems allow models to exchange arguments and refine responses by leveraging diverse perspectives across agents.

## 📦 Installation

### Requirements

- Python 3.9+
- A virtual environment (recommended)

```bash
# Clone the repo
git clone https://github.com/nickanely/Multi-LLM-Debate-System.git
cd Multi-LLM-Debate-System

# Create and activate venv
python -m venv venv
# macOS / Linux
source venv/bin/activate
# Windows
venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

🔑 API Keys & Environment Setup

This system interacts with one or more LLM providers. Create a .env file in project root:


OPENAI_API_KEY=your_openai_api_key
OTHER_LLM_API_KEY=your_other_provider_key


├── data/               # Input datasets for debates
├── src/                # Core source code (agents, debate logic)
├── experiments/        # Scripts to run and track experimental runs
├── results/            # Output from experiments
├── requirements.txt    # Python dependencies
└── README.md           # This file



```
## 📘 Usage

### 1. Prepare Input Data

Inputs should be formatted as **JSONL** or **CSV**, with at least a **topic** or **prompt** for the debate system:

```jsonl
{"id": "ex1", "prompt": "Is AI beneficial for education?"}
{"id": "ex2", "prompt": "Should social media be regulated?"}
```

Place your file under data/.

### 2. Run a Debate

Example command (customize based on your pipeline):

```bash
python src/run_debate.py \
  --input data/debate_prompts.jsonl \
  --output results/debate_out.jsonl \
  --agents gpt-4,claude-3
```

### 3. Review Results

Results will be saved in `results/`. Each entry includes:

- Agent responses
- Debate rounds
- Final consensus or judged answer

---

## 🧠 Agent & Debate Configuration

Edit or extend `src/config.py` to register:

- Agent names & model identifiers
- Token limits & temperature settings
- Judge agent behavior

---

### For additional questions please contact:

- Avkopashvili.ana@kiu.edu.ge  
- aneli.nikoloz@kiu.edu.ge