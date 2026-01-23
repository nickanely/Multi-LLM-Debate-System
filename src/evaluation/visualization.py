import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any
from pathlib import Path


def plot_accuracy_comparison(
        system_accuracy: float,
        baseline_accuracies: Dict[str, float],
        save_path: str = None
):
    """Plot accuracy comparison between system and baselines."""
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = ["Debate System"] + list(baseline_accuracies.keys())
    accuracies = [system_accuracy] + list(baseline_accuracies.values())
    colors = ['#2ecc71'] + ['#3498db'] * len(baseline_accuracies)

    bars = ax.bar(methods, accuracies, color=colors, edgecolor='black')

    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('System Accuracy vs Baselines', fontsize=14)
    ax.set_ylim(0, 1.0)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{acc:.1%}', ha='center', va='bottom', fontsize=10)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_per_category_accuracy(
        results: List[Dict],
        ground_truth: Dict[str, str],
        save_path: str = None
):
    """Plot accuracy breakdown by problem category."""
    from .metrics import answers_match

    # Group by category
    category_results = {}
    for result in results:
        problem = result.get("problem", {})
        category = problem.get("category", "Unknown")
        problem_id = result.get("problem_id")
        answer = result.get("final_answer", "")
        correct = ground_truth.get(problem_id, "")

        if category not in category_results:
            category_results[category] = {"correct": 0, "total": 0}

        category_results[category]["total"] += 1
        if answers_match(answer, correct):
            category_results[category]["correct"] += 1

    categories = list(category_results.keys())
    accuracies = [
        category_results[c]["correct"] / category_results[c]["total"]
        for c in categories
    ]

    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(categories, accuracies, color='#9b59b6', edgecolor='black')

    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy by Problem Category', fontsize=14)
    ax.set_ylim(0, 1.0)

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{acc:.1%}', ha='center', va='bottom', fontsize=10)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_solver_performance(
        per_solver_accuracy: Dict[str, float],
        save_path: str = None
):
    """Plot individual solver performance."""
    fig, ax = plt.subplots(figsize=(10, 6))

    solvers = list(per_solver_accuracy.keys())
    accuracies = list(per_solver_accuracy.values())
    colors = plt.cm.Set3(np.linspace(0, 1, len(solvers)))

    bars = ax.bar(solvers, accuracies, color=colors, edgecolor='black')

    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Individual Solver Performance (Refined Solutions)', fontsize=14)
    ax.set_ylim(0, 1.0)

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{acc:.1%}', ha='center', va='bottom', fontsize=10)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_improvement_analysis(
        results: List[Dict],
        ground_truth: Dict[str, str],
        save_path: str = None
):
    """Plot analysis of how refinement affected solutions."""
    from .metrics import answers_match

    # Analyze each result
    stayed_correct = 0
    stayed_wrong = 0
    improved = 0
    regressed = 0

    for result in results:
        problem_id = result.get("problem_id")
        true_answer = ground_truth.get(problem_id, "")

        winner_id = result.get("judgment", {}).get("winner_solver_id")

        initial = next(
            (s for s in result.get("initial_solutions", [])
             if s.get("solver_id") == winner_id),
            None
        )
        refined = next(
            (s for s in result.get("refined_solutions", [])
             if s.get("solver_id") == winner_id),
            None
        )

        if initial and refined:
            init_correct = answers_match(initial.get("final_answer", ""), true_answer)
            ref_correct = answers_match(refined.get("refined_answer", ""), true_answer)

            if init_correct and ref_correct:
                stayed_correct += 1
            elif not init_correct and not ref_correct:
                stayed_wrong += 1
            elif not init_correct and ref_correct:
                improved += 1
            else:
                regressed += 1

    fig, ax = plt.subplots(figsize=(8, 8))

    sizes = [stayed_correct, stayed_wrong, improved, regressed]
    labels = [
        f'Stayed Correct\n({stayed_correct})',
        f'Stayed Wrong\n({stayed_wrong})',
        f'Improved\n({improved})',
        f'Regressed\n({regressed})'
    ]
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12']

    # Filter out zero values
    non_zero = [(s, l, c) for s, l, c in zip(sizes, labels, colors) if s > 0]
    if non_zero:
        sizes, labels, colors = zip(*non_zero)
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
               startangle=90, explode=[0.02] * len(sizes))

    ax.set_title('Effect of Refinement on Solutions', fontsize=14)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_consensus_distribution(
        results: List[Dict],
        save_path: str = None
):
    """Plot distribution of solver agreement."""
    from .metrics import answers_match

    agreement_counts = {1: 0, 2: 0, 3: 0}  # Number of solvers agreeing

    for result in results:
        refined = result.get("refined_solutions", [])
        if len(refined) < 2:
            continue

        answers = [s.get("refined_answer", "") for s in refined]

        # Count agreements
        max_agree = 1
        for i, ans1 in enumerate(answers):
            count = 1
            for j, ans2 in enumerate(answers):
                if i != j and answers_match(ans1, ans2):
                    count += 1
            max_agree = max(max_agree, count)

        agreement_counts[max_agree] += 1

    fig, ax = plt.subplots(figsize=(8, 6))

    levels = ['No Consensus (1)', 'Partial (2)', 'Full Consensus (3)']
    counts = [agreement_counts[1], agreement_counts[2], agreement_counts[3]]
    colors = ['#e74c3c', '#f39c12', '#2ecc71']

    bars = ax.bar(levels, counts, color=colors, edgecolor='black')

    ax.set_ylabel('Number of Problems', fontsize=12)
    ax.set_title('Solver Agreement Distribution', fontsize=14)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_timing_breakdown(
        results: List[Dict],
        save_path: str = None
):
    """Plot timing breakdown across stages."""
    stages = ["role_assessment", "solution_generation", "peer_review",
              "refinement", "judgment"]

    # Aggregate timing data
    stage_times = {stage: [] for stage in stages}

    for result in results:
        timing = result.get("timing", {})
        for stage in stages:
            if stage in timing:
                stage_times[stage].append(timing[stage])

    fig, ax = plt.subplots(figsize=(10, 6))

    avg_times = [np.mean(stage_times[s]) if stage_times[s] else 0 for s in stages]

    bars = ax.bar(stages, avg_times, color='#3498db', edgecolor='black')

    ax.set_ylabel('Average Time (seconds)', fontsize=12)
    ax.set_title('Average Time per Stage', fontsize=14)

    for bar, time in zip(bars, avg_times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{time:.1f}s', ha='center', va='bottom', fontsize=10)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def generate_all_plots(
        results: List[Dict],
        baseline_results: Dict,
        ground_truth: Dict[str, str],
        metrics: Dict,
        output_dir: str = "results/plots"
):
    """Generate all evaluation plots."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 1. Accuracy comparison
    baseline_accuracies = {
        **{f"Single ({k})": v for k, v in baseline_results["accuracy"]["single_shot"].items()},
        "Majority Vote": baseline_results["accuracy"]["majority_vote"]
    }
    plot_accuracy_comparison(
        metrics["overall_accuracy"],
        baseline_accuracies,
        f"{output_dir}/accuracy_comparison.png"
    )

    # 2. Per-category accuracy
    plot_per_category_accuracy(
        results,
        ground_truth,
        f"{output_dir}/category_accuracy.png"
    )

    # 3. Solver performance
    plot_solver_performance(
        metrics["per_solver_accuracy"],
        f"{output_dir}/solver_performance.png"
    )

    # 4. Improvement analysis
    plot_improvement_analysis(
        results,
        ground_truth,
        f"{output_dir}/improvement_analysis.png"
    )

    # 5. Consensus distribution
    plot_consensus_distribution(
        results,
        f"{output_dir}/consensus_distribution.png"
    )

    # 6. Timing breakdown
    plot_timing_breakdown(
        results,
        f"{output_dir}/timing_breakdown.png"
    )

    print(f"All plots saved to {output_dir}/")