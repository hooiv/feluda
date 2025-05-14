#!/usr/bin/env python
"""
Benchmark Runner Script

This script runs benchmarks for Feluda and generates a report.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def run_benchmarks(
    output_dir: str = "benchmark_results",
    modules: Optional[List[str]] = None,
    json_output: bool = True,
    csv_output: bool = True,
    plot_output: bool = True,
) -> Dict[str, Any]:
    """
    Run benchmarks and generate a report.
    
    Args:
        output_dir: The directory to store the benchmark results.
        modules: The modules to benchmark. If None, all modules are benchmarked.
        json_output: Whether to generate a JSON report.
        csv_output: Whether to generate a CSV report.
        plot_output: Whether to generate plots.
        
    Returns:
        A dictionary with the benchmark results.
    """
    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine the modules to benchmark
    if modules is None:
        modules = [
            "performance",
            "resilience",
            "autonomic",
            "hardware",
            "crypto",
        ]
    
    # Run the benchmarks
    results = {}
    
    for module in modules:
        print(f"Running benchmarks for {module}...")
        
        # Run pytest-benchmark
        cmd = [
            "pytest",
            f"benchmarks/test_{module}.py",
            "--benchmark-only",
            "--benchmark-json", f"{output_dir}/{module}.json",
        ]
        
        try:
            subprocess.run(cmd, check=True)
            
            # Load the results
            with open(f"{output_dir}/{module}.json", "r") as f:
                module_results = json.load(f)
                results[module] = module_results
        except subprocess.SubprocessError as e:
            print(f"Failed to run benchmarks for {module}: {e}")
        except FileNotFoundError as e:
            print(f"Failed to load benchmark results for {module}: {e}")
    
    # Generate reports
    if json_output:
        generate_json_report(results, output_dir)
    
    if csv_output:
        generate_csv_report(results, output_dir)
    
    if plot_output:
        generate_plots(results, output_dir)
    
    return results


def generate_json_report(results: Dict[str, Any], output_dir: str) -> None:
    """
    Generate a JSON report.
    
    Args:
        results: The benchmark results.
        output_dir: The directory to store the report.
    """
    report_path = os.path.join(output_dir, "report.json")
    
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"JSON report generated: {report_path}")


def generate_csv_report(results: Dict[str, Any], output_dir: str) -> None:
    """
    Generate a CSV report.
    
    Args:
        results: The benchmark results.
        output_dir: The directory to store the report.
    """
    # Extract the benchmark data
    data = []
    
    for module, module_results in results.items():
        for benchmark in module_results.get("benchmarks", []):
            data.append({
                "module": module,
                "name": benchmark.get("name"),
                "min": benchmark.get("stats", {}).get("min"),
                "max": benchmark.get("stats", {}).get("max"),
                "mean": benchmark.get("stats", {}).get("mean"),
                "stddev": benchmark.get("stats", {}).get("stddev"),
                "median": benchmark.get("stats", {}).get("median"),
                "iqr": benchmark.get("stats", {}).get("iqr"),
                "iterations": benchmark.get("stats", {}).get("iterations"),
                "rounds": benchmark.get("stats", {}).get("rounds"),
            })
    
    # Create a DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    report_path = os.path.join(output_dir, "report.csv")
    df.to_csv(report_path, index=False)
    
    print(f"CSV report generated: {report_path}")


def generate_plots(results: Dict[str, Any], output_dir: str) -> None:
    """
    Generate plots.
    
    Args:
        results: The benchmark results.
        output_dir: The directory to store the plots.
    """
    # Create a directory for plots
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Extract the benchmark data
    data = []
    
    for module, module_results in results.items():
        for benchmark in module_results.get("benchmarks", []):
            data.append({
                "module": module,
                "name": benchmark.get("name"),
                "mean": benchmark.get("stats", {}).get("mean"),
                "stddev": benchmark.get("stats", {}).get("stddev"),
            })
    
    # Create a DataFrame
    df = pd.DataFrame(data)
    
    # Generate plots for each module
    for module in df["module"].unique():
        module_df = df[df["module"] == module]
        
        # Sort by mean time
        module_df = module_df.sort_values("mean")
        
        # Create a bar plot
        plt.figure(figsize=(12, 8))
        plt.barh(module_df["name"], module_df["mean"])
        plt.xlabel("Mean Time (seconds)")
        plt.ylabel("Benchmark")
        plt.title(f"Benchmark Results for {module}")
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(plots_dir, f"{module}.png")
        plt.savefig(plot_path)
        plt.close()
        
        print(f"Plot generated: {plot_path}")
    
    # Generate a summary plot
    summary_df = df.groupby("module")["mean"].mean().reset_index()
    summary_df = summary_df.sort_values("mean")
    
    plt.figure(figsize=(10, 6))
    plt.barh(summary_df["module"], summary_df["mean"])
    plt.xlabel("Mean Time (seconds)")
    plt.ylabel("Module")
    plt.title("Average Benchmark Results by Module")
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(plots_dir, "summary.png")
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Summary plot generated: {plot_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run benchmarks for Feluda")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="The directory to store the benchmark results",
    )
    parser.add_argument(
        "--modules",
        type=str,
        nargs="+",
        choices=["performance", "resilience", "autonomic", "hardware", "crypto", "all"],
        default=["all"],
        help="The modules to benchmark",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Disable JSON report generation",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Disable CSV report generation",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable plot generation",
    )
    
    args = parser.parse_args()
    
    # Determine the modules to benchmark
    modules = args.modules
    if "all" in modules:
        modules = None
    
    # Run the benchmarks
    run_benchmarks(
        output_dir=args.output_dir,
        modules=modules,
        json_output=not args.no_json,
        csv_output=not args.no_csv,
        plot_output=not args.no_plot,
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
