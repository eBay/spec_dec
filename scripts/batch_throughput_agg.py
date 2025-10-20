#!/usr/bin/env python3
"""
Aggregation script for batch throughput benchmark results.
Reads the CSV file and generates a formatted throughput table.
"""

import csv
import sys
from pathlib import Path

def main():
    csv_file = "benchmark_results/batch_vs_throughput/throughput_table.csv"
    
    if not Path(csv_file).exists():
        print(f"Error: {csv_file} not found!")
        sys.exit(1)
    
    # Read data
    data = {}
    batch_sizes = []
    methods = []
    errors = {}
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row['Method']
            batch_size = int(row['Batch_Size'])
            throughput = float(row['Throughput_TPS']) if row['Throughput_TPS'] and row['Success'] != 'SKIPPED' else 0
            error_msg = row.get('Error', '')
            
            if method not in data:
                data[method] = {}
                errors[method] = {}
                methods.append(method)
            
            data[method][batch_size] = throughput
            errors[method][batch_size] = error_msg
            
            if batch_size not in batch_sizes:
                batch_sizes.append(batch_size)
    
    batch_sizes.sort()
    methods = list(dict.fromkeys(methods))  # Preserve order, remove duplicates
    
    # Print throughput table
    print("\n" + "="*78)
    print("THROUGHPUT TABLE (tokens/second)")
    print("="*78)
    print("-" * (20 + len(batch_sizes) * 11))
    
    # Header
    header = f"{'Method':<20}"
    for bs in batch_sizes:
        header += f"{'B=' + str(bs):>11}"
    print(header)
    print("-" * (20 + len(batch_sizes) * 11))
    
    # Data rows
    for method in methods:
        row = f"{method:<20}"
        for bs in batch_sizes:
            if bs in data[method]:
                val = data[method][bs]
                if val > 0:
                    row += f"{val:>11.1f}"
                elif method == "SP-HF" and bs > 1:
                    row += f"{'SKIP':>11}"
                else:
                    row += f"{'FAIL':>11}"
            else:
                row += f"{'---':>11}"
        print(row)
    
    print("-" * (20 + len(batch_sizes) * 11))
    
    # Print errors if any
    has_errors = False
    for method in methods:
        for bs in batch_sizes:
            if bs in errors[method] and errors[method][bs] and errors[method][bs] != 'None':
                if not has_errors:
                    print("\n" + "="*78)
                    print("ERROR DETAILS")
                    print("="*78)
                    has_errors = True
                print(f"\n{method} (B={bs}):")
                error_msg = errors[method][bs]
                # Truncate very long error messages
                if len(error_msg) > 200:
                    error_msg = error_msg[:200] + "... (truncated)"
                print(f"  {error_msg}")
    
    if not has_errors:
        print("\n✓ All successful runs completed without errors")
    
    # Print statistics
    print("\n" + "="*78)
    print("STATISTICS")
    print("="*78)
    
    for bs in batch_sizes:
        successful_methods = [m for m in methods if bs in data[m] and data[m][bs] > 0]
        if successful_methods:
            best_method = max(successful_methods, key=lambda m: data[m][bs])
            best_throughput = data[best_method][bs]
            
            print(f"\nBatch Size {bs}:")
            print(f"  Best: {best_method} ({best_throughput:.1f} tokens/s)")
            print(f"  Successful: {len(successful_methods)}/{len(methods)} methods")

if __name__ == "__main__":
    main()