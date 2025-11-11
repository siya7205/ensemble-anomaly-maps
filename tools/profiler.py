"""
Performance profiling utilities for the ML pipeline.
Provides timing, memory tracking, and performance comparison tools.
"""
import time
import functools
import json
from pathlib import Path
from contextlib import contextmanager
import psutil
import numpy as np


class PerformanceMonitor:
    """Monitor and record performance metrics for pipeline stages."""
    
    def __init__(self):
        self.metrics = {}
        self.current_stage = None
        self.stage_start_time = None
        self.stage_start_memory = None
    
    @contextmanager
    def stage(self, stage_name):
        """
        Context manager for timing and monitoring a pipeline stage.
        
        Usage:
            with monitor.stage("feature_extraction"):
                # ... code ...
        """
        self.current_stage = stage_name
        self.stage_start_time = time.time()
        
        # Get current memory usage
        process = psutil.Process()
        self.stage_start_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        try:
            yield
        finally:
            elapsed = time.time() - self.stage_start_time
            end_memory = process.memory_info().rss / (1024 * 1024)  # MB
            memory_delta = end_memory - self.stage_start_memory
            
            self.metrics[stage_name] = {
                'duration_sec': elapsed,
                'memory_delta_mb': memory_delta,
                'peak_memory_mb': end_memory
            }
            
            print(f"  [{stage_name}] completed in {elapsed:.2f}s "
                  f"(memory: {memory_delta:+.1f} MB, peak: {end_memory:.1f} MB)")
    
    def save_report(self, output_path):
        """Save performance report to JSON file."""
        report = {
            'total_duration_sec': sum(m['duration_sec'] for m in self.metrics.values()),
            'peak_memory_mb': max(m['peak_memory_mb'] for m in self.metrics.values()),
            'stages': self.metrics
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n[Performance Report]")
        print(f"  Total duration: {report['total_duration_sec']:.2f}s")
        print(f"  Peak memory: {report['peak_memory_mb']:.1f} MB")
        print(f"  Report saved to: {output_path}")
    
    def print_summary(self):
        """Print performance summary to console."""
        if not self.metrics:
            print("No performance metrics recorded")
            return
        
        print("\n" + "="*70)
        print("PERFORMANCE SUMMARY")
        print("="*70)
        
        total_time = sum(m['duration_sec'] for m in self.metrics.values())
        
        for stage_name, metrics in self.metrics.items():
            pct = 100 * metrics['duration_sec'] / total_time if total_time > 0 else 0
            print(f"{stage_name:30s} {metrics['duration_sec']:8.2f}s ({pct:5.1f}%)  "
                  f"Mem: {metrics['memory_delta_mb']:+7.1f} MB")
        
        print("-"*70)
        print(f"{'TOTAL':30s} {total_time:8.2f}s (100.0%)")
        print("="*70)


def timer(func):
    """
    Decorator to time function execution.
    
    Usage:
        @timer
        def my_function():
            # ... code ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"[timer] {func.__name__} completed in {elapsed:.3f}s")
        return result
    return wrapper


class PerformanceComparator:
    """Compare performance between different optimization versions."""
    
    def __init__(self, baseline_path=None):
        self.baseline = None
        if baseline_path and Path(baseline_path).exists():
            with open(baseline_path) as f:
                self.baseline = json.load(f)
    
    def compare(self, current_report_path):
        """
        Compare current performance with baseline.
        
        Args:
            current_report_path: Path to current performance report
        """
        if not self.baseline:
            print("No baseline available for comparison")
            return
        
        with open(current_report_path) as f:
            current = json.load(f)
        
        print("\n" + "="*70)
        print("PERFORMANCE COMPARISON (vs baseline)")
        print("="*70)
        
        # Overall comparison
        baseline_total = self.baseline['total_duration_sec']
        current_total = current['total_duration_sec']
        speedup = baseline_total / current_total if current_total > 0 else 0
        improvement = 100 * (baseline_total - current_total) / baseline_total
        
        print(f"\nOverall Performance:")
        print(f"  Baseline:  {baseline_total:.2f}s")
        print(f"  Current:   {current_total:.2f}s")
        print(f"  Speedup:   {speedup:.2f}x")
        print(f"  Improvement: {improvement:+.1f}%")
        
        # Memory comparison
        baseline_mem = self.baseline['peak_memory_mb']
        current_mem = current['peak_memory_mb']
        mem_change = current_mem - baseline_mem
        mem_pct = 100 * mem_change / baseline_mem
        
        print(f"\nMemory Usage:")
        print(f"  Baseline:  {baseline_mem:.1f} MB")
        print(f"  Current:   {current_mem:.1f} MB")
        print(f"  Change:    {mem_change:+.1f} MB ({mem_pct:+.1f}%)")
        
        # Stage-by-stage comparison
        print(f"\nStage-by-Stage Comparison:")
        print(f"{'Stage':30s} {'Baseline':>10s} {'Current':>10s} {'Speedup':>10s}")
        print("-"*70)
        
        baseline_stages = self.baseline.get('stages', {})
        current_stages = current.get('stages', {})
        
        for stage_name in set(baseline_stages.keys()) | set(current_stages.keys()):
            baseline_t = baseline_stages.get(stage_name, {}).get('duration_sec', 0)
            current_t = current_stages.get(stage_name, {}).get('duration_sec', 0)
            
            if baseline_t > 0 and current_t > 0:
                stage_speedup = baseline_t / current_t
                print(f"{stage_name:30s} {baseline_t:10.2f}s {current_t:10.2f}s "
                      f"{stage_speedup:9.2f}x")
            elif current_t > 0:
                print(f"{stage_name:30s} {'N/A':>10s} {current_t:10.2f}s {'NEW':>10s}")
            else:
                print(f"{stage_name:30s} {baseline_t:10.2f}s {'N/A':>10s} {'REMOVED':>10s}")
        
        print("="*70)


def benchmark_function(func, *args, n_runs=5, **kwargs):
    """
    Benchmark a function over multiple runs.
    
    Args:
        func: Function to benchmark
        *args: Function arguments
        n_runs: Number of runs to average over
        **kwargs: Function keyword arguments
    
    Returns:
        dict with timing statistics
    """
    times = []
    
    print(f"Benchmarking {func.__name__} over {n_runs} runs...")
    
    for i in range(n_runs):
        start = time.time()
        func(*args, **kwargs)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Run {i+1}/{n_runs}: {elapsed:.3f}s")
    
    times = np.array(times)
    
    stats = {
        'mean': float(times.mean()),
        'std': float(times.std()),
        'min': float(times.min()),
        'max': float(times.max()),
        'median': float(np.median(times))
    }
    
    print(f"\nBenchmark Results:")
    print(f"  Mean:   {stats['mean']:.3f}s ± {stats['std']:.3f}s")
    print(f"  Median: {stats['median']:.3f}s")
    print(f"  Range:  [{stats['min']:.3f}s, {stats['max']:.3f}s]")
    
    return stats


if __name__ == "__main__":
    # Example usage
    monitor = PerformanceMonitor()
    
    with monitor.stage("example_stage_1"):
        time.sleep(0.1)
        data = [i**2 for i in range(1000000)]
    
    with monitor.stage("example_stage_2"):
        time.sleep(0.2)
        result = sum(data)
    
    monitor.print_summary()
    monitor.save_report("/tmp/performance_report.json")
