#!/usr/bin/env python3
"""
Generate a comprehensive MMLU benchmark report with visualizations.
"""

# Standard
import glob
import json
import os
import re

try:
    # Third Party
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Visualization dependencies not available: {e}")
    print("📝 Falling back to text-only report generation")
    VISUALIZATION_AVAILABLE = False

# Use environment variable for results directory if set, otherwise default
RESULTS_DIR = os.environ.get("RESULTS_DIR", "mmlu-results")
OUTPUT_PDF = "compare-results/mmlu_benchmark_report.pdf"
OUTPUT_JSON = "compare-results/results_summary.json"

# Set style for better-looking plots
if VISUALIZATION_AVAILABLE:
    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")


def parse_result_file(filepath):
    """Parse a single result file to extract metrics."""
    results = {
        "filename": os.path.basename(filepath),
        "config": os.path.basename(filepath).replace(".txt", ""),
        "accuracy": None,
        "latency": None,
        "subjects": {},
        "total_questions": 0,
    }

    try:
        with open(filepath, "r") as f:
            content = f.read()

        # Extract overall metrics
        for line in content.splitlines():
            # Check subject-specific results first (more specific pattern)
            if match := re.match(
                r"Average accuracy ([0-9.]+), latency ([0-9.]+), #q: ([0-9]+) - (.+)",
                line,
            ):
                # Subject-specific results
                acc, lat, nq, subject = match.groups()
                results["subjects"][subject] = {
                    "accuracy": float(acc),
                    "latency": float(lat),
                    "questions": int(nq),
                }
                results["total_questions"] += int(nq)
            elif match := re.match(r"Average accuracy:?\s*([0-9.]+)", line):
                results["accuracy"] = float(match.group(1))
            elif match := re.match(r"Total latency:?\s*([0-9.]+)", line):
                results["latency"] = float(match.group(1))

    except Exception as e:
        print(f"⚠️ Error parsing {filepath}: {e}")

    return results


def create_comparison_plots(all_results):
    """Create comparison plots for the results."""
    if not VISUALIZATION_AVAILABLE:
        return None

    if not all_results:
        return None

    # Prepare data for plotting
    configs = []
    accuracies = []
    latencies = []

    for result in all_results:
        if result["accuracy"] is not None and result["latency"] is not None:
            configs.append(result["config"].replace("_", " ").title())
            accuracies.append(result["accuracy"] * 100)  # Convert to percentage
            latencies.append(result["latency"])

    if not configs:
        return None

    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("MMLU Benchmark Results Comparison", fontsize=16, fontweight="bold")

    # 1. Accuracy comparison
    bars1 = ax1.bar(configs, accuracies, color=sns.color_palette("husl", len(configs)))
    ax1.set_title("Accuracy Comparison", fontweight="bold")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(0, max(accuracies) * 1.1)

    # Add value labels on bars
    for bar, acc in zip(bars1, accuracies, strict=False):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.5,
            f"{acc:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 2. Latency comparison
    bars2 = ax2.bar(configs, latencies, color=sns.color_palette("husl", len(configs)))
    ax2.set_title("Latency Comparison", fontweight="bold")
    ax2.set_ylabel("Latency (seconds)")
    ax2.set_ylim(0, max(latencies) * 1.1)

    # Add value labels on bars
    for bar, lat in zip(bars2, latencies, strict=False):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.1,
            f"{lat:.2f}s",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 3. Accuracy vs Latency scatter plot
    ax3.scatter(
        latencies, accuracies, s=100, alpha=0.7, c=range(len(configs)), cmap="viridis"
    )
    ax3.set_xlabel("Latency (seconds)")
    ax3.set_ylabel("Accuracy (%)")
    ax3.set_title("Accuracy vs Latency Trade-off", fontweight="bold")

    # Add labels for each point
    for i, config in enumerate(configs):
        ax3.annotate(
            config,
            (latencies[i], accuracies[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )

    # 4. Performance efficiency (accuracy/latency ratio)
    efficiency = [acc / lat for acc, lat in zip(accuracies, latencies, strict=False)]
    bars4 = ax4.bar(configs, efficiency, color=sns.color_palette("husl", len(configs)))
    ax4.set_title("Performance Efficiency (Accuracy/Latency)", fontweight="bold")
    ax4.set_ylabel("Efficiency (% per second)")

    # Add value labels on bars
    for bar, eff in zip(bars4, efficiency, strict=False):
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.1,
            f"{eff:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Rotate x-axis labels if needed
    for ax in [ax1, ax2, ax3, ax4]:
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    return fig


def create_subject_breakdown(all_results):
    """Create subject-by-subject breakdown if available."""
    if not VISUALIZATION_AVAILABLE:
        return None

    # Find the result with the most subject data
    best_result = None
    max_subjects = 0

    for result in all_results:
        if len(result["subjects"]) > max_subjects:
            max_subjects = len(result["subjects"])
            best_result = result

    if not best_result or not best_result["subjects"]:
        return None

    # Create subject breakdown plot
    subjects = list(best_result["subjects"].keys())
    accuracies = [best_result["subjects"][s]["accuracy"] * 100 for s in subjects]

    fig, ax = plt.subplots(figsize=(15, 8))
    bars = ax.bar(
        range(len(subjects)),
        accuracies,
        color=sns.color_palette("viridis", len(subjects)),
    )

    ax.set_title(
        f"Subject-wise Accuracy Breakdown ({best_result['config']})", fontweight="bold"
    )
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Subjects")
    ax.set_xticks(range(len(subjects)))
    ax.set_xticklabels(subjects, rotation=45, ha="right")

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies, strict=False):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.5,
            f"{acc:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    return fig


def create_summary_table(all_results):
    """Create a summary table figure."""
    if not VISUALIZATION_AVAILABLE:
        return None

    if not all_results:
        return None

    # Prepare table data
    table_data = []
    for result in all_results:
        row = [
            result["config"].replace("_", " ").title(),
            f"{result['accuracy'] * 100:.2f}%" if result["accuracy"] else "N/A",
            f"{result['latency']:.2f}s" if result["latency"] else "N/A",
            str(result["total_questions"]) if result["total_questions"] else "N/A",
            f"{len(result['subjects'])}" if result["subjects"] else "0",
        ]
        table_data.append(row)

    # Create table plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("tight")
    ax.axis("off")

    headers = [
        "Configuration",
        "Accuracy",
        "Latency",
        "Total Questions",
        "Subjects Tested",
    ]
    table = ax.table(
        cellText=table_data, colLabels=headers, cellLoc="center", loc="center"
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)

    # Color the header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor("#4CAF50")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor("#f0f0f0")

    ax.set_title(
        "MMLU Benchmark Results Summary", fontweight="bold", fontsize=16, pad=20
    )

    return fig


def generate_report():
    """Generate the complete PDF report."""
    print(f"🔍 Looking for results in: {os.path.abspath(RESULTS_DIR)}")

    if not os.path.exists(RESULTS_DIR):
        print(f"❌ Results directory '{RESULTS_DIR}' does not exist")
        return

    # Find all result files
    result_files = glob.glob(os.path.join(RESULTS_DIR, "*.txt"))
    print(
        f"📁 Found {len(result_files)} result files: "
        f"{[os.path.basename(f) for f in result_files]}"
    )

    if not result_files:
        print("❌ No result files found")
        return

    # Parse all results
    all_results = []
    for filepath in result_files:
        result = parse_result_file(filepath)
        all_results.append(result)
        print(
            f"📊 Parsed {result['config']}: "
            f"acc={result['accuracy']}, lat={result['latency']}"
        )

    # Create output directory
    os.makedirs("compare-results", exist_ok=True)

    if VISUALIZATION_AVAILABLE:
        # Generate PDF report with visualizations
        with PdfPages(OUTPUT_PDF) as pdf:
            # Page 1: Summary table
            fig_table = create_summary_table(all_results)
            if fig_table:
                pdf.savefig(fig_table, bbox_inches="tight")
                plt.close(fig_table)

            # Page 2: Comparison plots
            fig_comparison = create_comparison_plots(all_results)
            if fig_comparison:
                pdf.savefig(fig_comparison, bbox_inches="tight")
                plt.close(fig_comparison)

            # Page 3: Subject breakdown (if available)
            fig_subjects = create_subject_breakdown(all_results)
            if fig_subjects:
                pdf.savefig(fig_subjects, bbox_inches="tight")
                plt.close(fig_subjects)

        print(f"✅ PDF report generated: {OUTPUT_PDF}")
    else:
        # Generate text-only report
        text_report_path = "compare-results/detailed_report.txt"
        with open(text_report_path, "w") as f:
            f.write("MMLU Benchmark Results - Detailed Report\n")
            f.write("=" * 50 + "\n\n")

            for result in all_results:
                f.write(f"Configuration: {result['config']}\n")
                f.write(
                    f"  Accuracy: {result['accuracy'] * 100:.2f}%\n"
                    if result["accuracy"]
                    else "  Accuracy: N/A\n"
                )
                f.write(
                    f"  Latency: {result['latency']:.2f}s\n"
                    if result["latency"]
                    else "  Latency: N/A\n"
                )
                f.write(f"  Total Questions: {result['total_questions']}\n")
                f.write(f"  Subjects Tested: {len(result['subjects'])}\n")
                f.write("-" * 30 + "\n")

        print(f"✅ Text report generated: {text_report_path}")

    # Save JSON summary (always available)
    summary_data = {
        "timestamp": str(pd.Timestamp.now())
        if VISUALIZATION_AVAILABLE
        else str(os.path.getctime(result_files[0])),
        "total_configs": len(all_results),
        "results": all_results,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"✅ JSON summary: {OUTPUT_JSON}")


if __name__ == "__main__":
    generate_report()
