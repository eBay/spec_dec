#!/usr/bin/env python3
"""
Parse group rate experiment logs and generate LaTeX table.

This script extracts:
- Group rate: Percentage of cross-batch (same length) operations
- OH(ms): Overhead in milliseconds (Stage 1 + Stage 4)
- OH(%): Overhead as percentage of total time
- TPS: Tokens per second

Example log format:
============================================================
PROFILING RESULTS - Stage Time Breakdown:
============================================================
Stage 1 (Get Batch):           5.001s ( 10.1%)
Stage 2 (Draft Generate):     14.660s ( 29.5%)
Stage 3 (Verification):       27.741s ( 55.9%)
Stage 4 (Write Back):          2.234s (  4.5%)
------------------------------------------------------------
Total Stage Time:             49.637s
Total Pure Decoding Time:     49.674s
============================================================

============================================================
GET_BATCH BRANCH STATISTICS:
============================================================
Total get_batch calls:       952
Cross-batch (same length):      40 (  4.2%)
Fallback (realignment):        912 ( 95.8%)
============================================================

Ours-XBatch     49.67        156.5      0.229  0.0          952      952      ✓ OK
"""

import re
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple


def parse_log_file(log_file: Path) -> Optional[Dict]:
    """
    Parse a single log file to extract group rate metrics.

    Returns:
        Dictionary with extracted metrics or None if parsing failed
    """
    if not log_file.exists():
        print(f"Warning: Log file {log_file} not found")
        return None

    content = log_file.read_text()
    metrics = {}

    # Extract Total get_batch calls
    total_calls_match = re.search(
        r'Total get_batch calls:\s+(\d+)',
        content
    )
    if total_calls_match:
        metrics['total_spec_calls'] = int(total_calls_match.group(1))
    else:
        print(f"Warning: Could not find total get_batch calls in {log_file}")
        metrics['total_spec_calls'] = 952  # Default value based on your example

    # Extract Group Rate (Cross-batch percentage)
    group_rate_match = re.search(
        r'Cross-batch \(same length\):\s+\d+\s+\(\s*([\d.]+)%\)',
        content
    )
    if group_rate_match:
        metrics['group_rate'] = float(group_rate_match.group(1))
    else:
        print(f"Warning: Could not find group rate in {log_file}")
        metrics['group_rate'] = 0.0  # Default to 0 if not found

    # Extract Stage 1 (Get Batch) time
    stage1_match = re.search(r'Stage 1 \(Get Batch\):\s+([\d.]+)s', content)
    if stage1_match:
        metrics['stage1_time'] = float(stage1_match.group(1))
    else:
        print(f"Warning: Could not find Stage 1 time in {log_file}")
        return None

    # Extract Stage 4 (Write Back) time
    stage4_match = re.search(r'Stage 4 \(Write Back\):\s+([\d.]+)s', content)
    if stage4_match:
        metrics['stage4_time'] = float(stage4_match.group(1))
    else:
        print(f"Warning: Could not find Stage 4 time in {log_file}")
        return None

    # Calculate total overhead (Stage 1 + Stage 4)
    # metrics['overhead_s'] = metrics['stage1_time'] + metrics['stage4_time']
    metrics['overhead_s'] = metrics['stage1_time']

    # Calculate overhead per spec call in milliseconds
    metrics['overhead_ms'] = metrics['overhead_s'] * 1000  # Convert to milliseconds
    metrics['overhead_per_spec'] = metrics['overhead_ms'] / metrics['total_spec_calls']

    # Extract total stage time for percentage calculation
    total_stage_match = re.search(r'Total Stage Time:\s+([\d.]+)s', content)
    if total_stage_match:
        total_stage_time = float(total_stage_match.group(1))
        # metrics['overhead_percentage'] = (
        #     (metrics['stage1_time'] + metrics['stage4_time']) / total_stage_time * 100
        # )
        metrics['overhead_percentage'] = (
            (metrics['stage1_time']) / total_stage_time * 100
        )
    else:
        print(f"Warning: Could not find total stage time in {log_file}")
        return None

    # Extract tokens per second from benchmark results
    # Look for pattern: Ours-XBatch     49.67        156.5
    tps_match = re.search(r'Ours-XBatch\s+[\d.]+\s+([\d.]+)', content)
    if tps_match:
        metrics['tokens_per_second'] = float(tps_match.group(1))
    else:
        print(f"Warning: Could not find tokens per second in {log_file}")
        return None

    return metrics


def generate_latex_table(results: Dict[Tuple[int, str], Dict]) -> str:
    """
    Generate LaTeX table from results.

    Args:
        results: Dictionary mapping (batch_size, config) to metrics

    Returns:
        LaTeX table as string
    """
    latex = r"""\begin{table}[h]
\centering
\begin{tabular}{@{}llrrrrrr@{}}
\toprule
BS & Config & Group\% & |Spec| & OH(ms) & OH/Spec & OH(\%) & TPS \\
\midrule
"""

    # Process batch sizes in order
    batch_sizes = [2, 4, 8]

    for bs in batch_sizes:
        # Process Random config first, then All-Mean
        for i, config in enumerate(['random', 'all-mean']):
            key = (bs, config)
            if key not in results:
                continue

            metrics = results[key]

            # Format values
            config_display = "Random" if config == "random" else "All-Mean"
            group_pct = f"{metrics['group_rate']:.1f}\\%"
            spec_calls = f"{metrics['total_spec_calls']}"
            overhead_ms = f"{metrics['overhead_ms']:.1f}"
            overhead_per_spec = f"{metrics['overhead_per_spec']:.2f}"
            overhead_pct = f"{metrics['overhead_percentage']:.1f}"
            tps = f"{metrics['tokens_per_second']:.1f}"

            # Use multirow for the first config of each batch size
            if i == 0:
                latex += f"\\multirow{{2}}{{*}}{{{bs}}} \n"
                latex += f" & {config_display} & {group_pct} & {spec_calls} & {overhead_ms} & {overhead_per_spec} & {overhead_pct} & {tps} \\\\\n"
            else:
                latex += f" & {config_display} & {group_pct} & {spec_calls} & {overhead_ms} & {overhead_per_spec} & {overhead_pct} & {tps} \\\\\n"

        # Add horizontal line between batch sizes (except after the last one)
        if bs != batch_sizes[-1]:
            latex += r"\cmidrule{1-8}" + "\n"

    latex += r"""\bottomrule
\end{tabular}
\caption{Impact of grouping on overhead across batch and window sizes (Multi30k dataset), window size 50, data point 100, mean is 16 token. |Spec| = total speculation calls, OH/Spec = overhead per speculation call in ms.}
\end{table}"""

    return latex


def main():
    """Main function to process logs and generate table."""

    # Configuration
    log_dir = Path("logs/group_rate")
    batch_sizes = [2, 4, 8]
    configs = ["random", "all-mean"]

    print("="*60)
    print("Group Rate Analysis")
    print("="*60)
    print(f"Log directory: {log_dir}")
    print()

    # Parse all log files
    results = {}

    for bs in batch_sizes:
        print(f"Processing batch_size={bs}...")

        for config in configs:
            log_file = log_dir / f"group_rate_bs{bs}_{config}.log"
            print(f"  Reading {config} configuration...")

            metrics = parse_log_file(log_file)
            if metrics is None:
                print(f"    ✗ Failed to parse {log_file}")
                continue

            results[(bs, config)] = metrics

            # Print summary
            print(f"    ✓ Group: {metrics['group_rate']:.1f}%, "
                  f"|Spec|: {metrics['total_spec_calls']}, "
                  f"OH: {metrics['overhead_ms']:.1f}ms ({metrics['overhead_percentage']:.1f}%), "
                  f"OH/Spec: {metrics['overhead_per_spec']:.2f}ms, "
                  f"TPS: {metrics['tokens_per_second']:.1f}")

        print()

    if not results:
        print("Error: No valid results found!")
        sys.exit(1)

    # Generate LaTeX table
    latex_table = generate_latex_table(results)

    # Save to file
    output_file = Path("group_rate_table.tex")
    output_file.write_text(latex_table)
    print(f"LaTeX table saved to: {output_file}")
    print()

    # Also print to console
    print("LaTeX Table:")
    print("-"*60)
    print(latex_table)
    print("-"*60)

    # Print analysis summary
    print()
    print("Analysis Summary:")
    print("-"*60)

    for bs in batch_sizes:
        random_key = (bs, "random")
        all_mean_key = (bs, "all-mean")

        if random_key in results and all_mean_key in results:
            group_improvement = results[all_mean_key]['group_rate'] - results[random_key]['group_rate']
            overhead_reduction = results[random_key]['overhead_s'] - results[all_mean_key]['overhead_s']
            tps_improvement = results[all_mean_key]['tokens_per_second'] - results[random_key]['tokens_per_second']

            print(f"Batch Size {bs}:")
            print(f"  Group rate improvement: +{group_improvement:.1f}%")
            print(f"  Overhead reduction: {overhead_reduction:.3f}s")
            print(f"  TPS improvement: +{tps_improvement:.1f}")
            print()


if __name__ == "__main__":
    main()