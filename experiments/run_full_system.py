import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.openai_model import OpenAIModel
from src.models.anthropic_model import AnthropicModel
from src.models.google_model import GoogleModel
from src.models.xAI_model import XAIModel
from src.orchestration.debate_runner import DebateRunner
from src.evaluation.metrics import calculate_all_metrics
from src.evaluation.baselines import run_all_baselines_async
from src.evaluation.visualization import generate_all_plots
from data.loader import load_problems, load_ground_truth

def setup_models():
    """Initialize all LLM models."""
    models = {}

    # OpenAI GPT
    if os.getenv("OPENAI_API_KEY"):
        models["gpt"] = OpenAIModel(
            model_name="gpt-5-nano",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        print("✓ OpenAI GPT initialized")
    else:
        print("✗ OpenAI API key not found")

    # Anthropic Claude
    if os.getenv("ANTHROPIC_API_KEY"):
        models["claude"] = AnthropicModel(
            model_name="claude-haiku-4-5",
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        print("✓ Anthropic Claude initialized")
    else:
        print("✗ Anthropic API key not found")

    # Google Gemini
    if os.getenv("GOOGLE_API_KEY"):
        models["gemini"] = GoogleModel(
            model_name="gemini-2.5-flash-lite",
            api_key=os.getenv("GOOGLE_API_KEY")
        )
        print("✓ Google Gemini initialized")
    else:
        print("✗ Google API key not found")

    # xAI Grok
    if os.getenv("XAI_API_KEY"):
        models["grok"] = XAIModel(
            model_name="grok-4-fast-non-reasoning", #or grok-4-fast-reasoning
            api_key=os.getenv("XAI_API_KEY")
        )
        print("✓ xAI Grok initialized")
    else:
        print("✗ xAI API key not found")

    return models





async def main_async():
    """Main async entry point."""
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/experiment_{timestamp}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(f"{output_dir}/plots").mkdir(exist_ok=True)

    # Load problems and ground truth
    print("Loading problems...")
    problems = load_problems()
    ground_truth = load_ground_truth()
    print(f"Loaded {len(problems)} problems")

    # Setup models
    print("\nInitializing models...")
    models = setup_models()

    if len(models) < 2:
        print("ERROR: Need at least 2 models. Please set OPENAI_API_KEY.")
        return

    print(f"\nModels available: {list(models.keys())}")

    # ===== RUN BASELINES =====
    print("\n" + "=" * 60)
    print("RUNNING BASELINES (Async Parallel)")
    print("=" * 60)

    baseline_start = datetime.now()
    baseline_results = await run_all_baselines_async(
        models, problems, ground_truth
    )
    baseline_time = (datetime.now() - baseline_start).total_seconds()

    print(f"\nBaseline completed in {baseline_time:.1f}s")
    print("\nBaseline Accuracies:")
    for model_id, acc in baseline_results["accuracy"]["single_shot"].items():
        print(f"  {model_id} (single shot): {acc:.1%}")
    print(f"  Majority Vote: {baseline_results['accuracy']['majority_vote']:.1%}")

    with open(f"{output_dir}/baseline_results.json", 'w') as f:
        json.dump(baseline_results, f, indent=2, default=str)

    # ===== RUN DEBATE SYSTEM =====
    print("\n" + "=" * 60)
    print("RUNNING DEBATE SYSTEM (Async Parallel)")
    print("=" * 60)

    runner = DebateRunner(
        models
    )

    debate_start = datetime.now()
    debate_results = await runner.run_batch_async(problems)
    debate_time = (datetime.now() - debate_start).total_seconds()

    print(f"\nDebate system completed in {debate_time:.1f}s")

    with open(f"{output_dir}/debate_results.json", 'w') as f:
        json.dump(debate_results, f, indent=2, default=str)

    # ===== CALCULATE METRICS =====
    print("\n" + "=" * 60)
    print("CALCULATING METRICS")
    print("=" * 60)

    metrics = calculate_all_metrics(debate_results, ground_truth)

    print(f"\nOverall Accuracy: {metrics['overall_accuracy']:.1%}")
    print(f"Improvement Rate: {metrics['improvement_rate']:.1%}")
    print(f"Consensus Rate: {metrics['consensus_rate']:.1%}")
    print(f"Judge Accuracy (disagreements): {metrics['judge_metrics']['judge_accuracy']:.1%}")

    print("\nPer-Solver Accuracy:")
    for solver_id, acc in metrics['per_solver_accuracy'].items():
        print(f"  {solver_id}: {acc:.1%}")

    with open(f"{output_dir}/metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    # ===== GENERATE PLOTS =====
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)

    generate_all_plots(
        debate_results,
        baseline_results,
        ground_truth,
        metrics,
        f"{output_dir}/plots"
    )

    # ===== SUMMARY =====
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    print(f"\nTotal Problems: {len(problems)}")
    print(f"Total Models: {len(models)}")

    print(f"\nTiming:")
    print(f"  Baselines: {baseline_time:.1f}s")
    print(f"  Debate System: {debate_time:.1f}s")

    print(f"\nAccuracy Comparison:")
    print(f"  Debate System: {metrics['overall_accuracy']:.1%}")
    best_single = max(baseline_results["accuracy"]["single_shot"].values())
    for model_id, acc in baseline_results["accuracy"]["single_shot"].items():
        marker = " (best)" if acc == best_single else ""
        print(f"  {model_id} (single): {acc:.1%}{marker}")
    print(f"  Majority Vote: {baseline_results['accuracy']['majority_vote']:.1%}")

    improvement = metrics['overall_accuracy'] - best_single
    print(f"\nImprovement over best single model: {improvement:+.1%}")

    print(f"\nResults saved to: {output_dir}/")

    # Save summary
    summary = {
        "timestamp": timestamp,
        "num_problems": len(problems),
        "num_models": len(models),
        "models": list(models.keys()),
        "timing": {
            "baselines_seconds": baseline_time,
            "debate_seconds": debate_time
        },
        "accuracy": {
            "debate_system": metrics['overall_accuracy'],
            "best_single_shot": best_single,
            "majority_vote": baseline_results['accuracy']['majority_vote']
        },
        "improvement_over_best_single": improvement
    }

    with open(f"{output_dir}/summary.json", 'w') as f:
        json.dump(summary, f, indent=2)


def main():
    """Entry point."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()