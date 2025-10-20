#!/usr/bin/env python3
"""
bash scripts/run_overhead_scaling.sh

Parse overhead scaling experiment logs and generate LaTeX table.

Example log format (from batch_size=4):
============================================================
PROFILING RESULTS - Stage Time Breakdown:
============================================================
Stage 1 (Draft Generate):      22.352s ( 27.4%)
Stage 2 (Verification):        36.552s ( 44.9%)
Stage 3 (Update/Alignment):    22.527s ( 27.7%)
------------------------------------------------------------
Total Stage Time:              81.431s
Total Pure Decoding Time:      81.967s
Overhead (non-stage time):      0.536s
============================================================
Total draft calls: 1469, Total verification calls: 1469

================================================================================
BENCHMARK RESULTS SUMMARY
================================================================================
Method          Pure Time(s) Tokens/s   TAR    Latency(ms)  Draft    Verify   Status
----------------------------------------------------------------------------------------------------
Ours-Batch-Cache 81.97        95.6       0.228  55.7         1469     1469     ✓ OK
                └─ Timing: 0.06s tokenization, 0.04s post-proc, 0.1% overhead
--------------------------------------------------------------------------------
"""

import re
import sys
from pathlib import Path
from typing import Dict, Optional


def parse_log_file(log_file: Path) -> Optional[Dict]:
    """
    Parse a single log file to extract overhead metrics.

    Returns:
        Dictionary with extracted metrics or None if parsing failed
    """
    if not log_file.exists():
        print(f"Warning: Log file {log_file} not found")
        return None

    content = log_file.read_text()

    metrics = {}

    # Extract Stage 3 (Update/Alignment) time as the overhead
    alignment_match = re.search(r'Stage 3 \(Update/Alignment\):\s+([\d.]+)s', content)
    if alignment_match:
        metrics['overhead_time'] = float(alignment_match.group(1))
    else:
        print(f"Warning: Could not find Stage 3 alignment time in {log_file}")
        return None

    # Extract pure decoding time
    pure_time_match = re.search(r'Total Pure Decoding Time:\s+([\d.]+)s', content)
    if pure_time_match:
        metrics['pure_decoding_time'] = float(pure_time_match.group(1))
    else:
        print(f"Warning: Could not find pure decoding time in {log_file}")
        return None

    # Extract verification calls
    verif_calls_match = re.search(r'Total verification calls:\s+(\d+)', content)
    if verif_calls_match:
        metrics['verification_calls'] = int(verif_calls_match.group(1))
    else:
        print(f"Warning: Could not find verification calls in {log_file}")
        return None

    # Extract tokens per second from benchmark results
    tps_match = re.search(r'Ours-Batch-Cache\s+([\d.]+)\s+([\d.]+)', content)
    if tps_match:
        metrics['tokens_per_second'] = float(tps_match.group(2))
    else:
        print(f"Warning: Could not find tokens per second in {log_file}")
        return None

    return metrics


def calculate_derived_metrics(metrics: Dict, baseline_overhead_pct: Optional[float] = None) -> Dict:
    """
    Calculate derived metrics from raw extracted metrics.

    Args:
        metrics: Raw metrics from log file
        baseline_overhead_pct: Overhead percentage for batch_size=1 (for scaling calculation)

    Returns:
        Dictionary with all metrics including derived ones
    """
    result = metrics.copy()

    # Calculate overhead percentage
    result['overhead_percentage'] = (metrics['overhead_time'] / metrics['pure_decoding_time']) * 100

    # Calculate scaling factor based on overhead percentage if baseline is provided
    if baseline_overhead_pct is not None:
        result['scale'] = result['overhead_percentage'] / baseline_overhead_pct
    else:
        result['scale'] = 1.0

    return result


def generate_latex_table(results: Dict[int, Dict]) -> str:
    """
    Generate LaTeX table from results.

    Args:
        results: Dictionary mapping batch_size to metrics

    Returns:
        LaTeX table as string
    """
    latex = r"""\begin{table}[t]
\centering
\begin{tabular}{@{}rrrrrr@{}}
\toprule
BS & TPS & |Spec| & OH (s) & OH (\%) & Scale \\
\midrule
"""

    # Sort by batch size
    batch_sizes = sorted(results.keys())

    for bs in batch_sizes:
        metrics = results[bs]

        # Format values
        tps = f"{metrics['tokens_per_second']:.1f}"
        spec_count = f"{metrics['verification_calls']}"
        overhead_s = f"{metrics['overhead_time']:.3f}"
        overhead_pct = f"{metrics['overhead_percentage']:.1f}"
        scale = f"{metrics['scale']:.2f}"

        # Adjust spacing based on value width
        latex += f"{bs:2d} & {tps:>5s} & {spec_count:>5s} & {overhead_s:>6s} & {overhead_pct:>4s} & {scale:>4s} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\vspace{-0.5em}
\begin{flushleft}
\footnotesize
\textbf{Legend:} BS = Batch Size, TPS = Tokens/sec, |Spec| = Number of verification calls, OH (s) = Total alignment overhead (Stage 3), OH (\%) = Alignment overhead percentage, Scale = Overhead percentage scaling factor (OH\%(B)/OH\%(1))
\end{flushleft}
\caption{Overhead scaling analysis for Our-Batch method across batch sizes}
\label{tab:overhead-scaling}
\end{table}"""

    return latex


def main():
    """Main function to process logs and generate table."""

    # Configuration
    log_dir = Path("logs/overhead_scaling")
    batch_sizes = [1, 2, 4, 8, 16, 32]

    print("="*60)
    print("Overhead Scaling Analysis")
    print("="*60)
    print(f"Log directory: {log_dir}")
    print()

    # Parse all log files
    results = {}
    baseline_overhead_pct = None

    for bs in batch_sizes:
        log_file = log_dir / f"overhead_bs{bs}.log"
        print(f"Processing batch_size={bs}...")

        metrics = parse_log_file(log_file)
        if metrics is None:
            print(f"  ✗ Failed to parse {log_file}")
            continue

        # Calculate derived metrics
        if bs == 1:
            # First batch size becomes baseline
            derived = calculate_derived_metrics(metrics)
            baseline_overhead_pct = derived['overhead_percentage']
        else:
            derived = calculate_derived_metrics(metrics, baseline_overhead_pct)

        results[bs] = derived

        # Print summary
        print(f"  ✓ TPS: {derived['tokens_per_second']:.1f}, "
              f"|Spec|: {derived['verification_calls']}, "
              f"OH: {derived['overhead_time']:.3f}s ({derived['overhead_percentage']:.1f}%), "
              f"Scale: {derived['scale']:.2f}")

    print()

    if not results:
        print("Error: No valid results found!")
        sys.exit(1)

    # Generate LaTeX table
    latex_table = generate_latex_table(results)

    # Save to file
    output_file = Path("overhead_scaling_table.tex")
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

    if 1 in results and 32 in results:
        speedup_32 = results[32]['tokens_per_second'] / results[1]['tokens_per_second']
        scale_32 = results[32]['scale']
        print(f"Throughput scaling (BS=32 vs BS=1): {speedup_32:.1f}x")
        print(f"Overhead scaling (BS=32 vs BS=1): {scale_32:.2f}x")
        print(f"Overhead remains manageable: {'YES' if scale_32 < 10 else 'NO'}")


if __name__ == "__main__":
    main()