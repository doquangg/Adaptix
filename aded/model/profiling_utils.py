import cProfile
import pstats
import time
import functools
import torch
from typing import Dict, Optional
import os
from pathlib import Path
import json
from datetime import datetime

class DetailedProfiler:
    def __init__(self, output_dir: str = "profiling_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.function_stats = {}
        self.current_context = []

    def __call__(self, func):
        """Decorator to profile a function"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            context = '.'.join(self.current_context + [func.__qualname__])
            start_time = time.perf_counter()
            start_memory = self._get_memory_stats()

            try:
                self.current_context.append(func.__qualname__)
                result = func(*args, **kwargs)
                return result
            finally:
                self.current_context.pop()
                end_time = time.perf_counter()
                end_memory = self._get_memory_stats()
                
                self._update_stats(
                    context,
                    end_time - start_time,
                    {k: end_memory[k] - start_memory[k] for k in start_memory}
                )

        return wrapper

    def _get_memory_stats(self) -> Dict[str, float]:
        """Get current memory usage"""
        return {
            'gpu_allocated': torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0,
            'gpu_reserved': torch.cuda.memory_reserved() / 1024**2 if torch.cuda.is_available() else 0,
            'gpu_max_allocated': torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        }

    def _update_stats(self, context: str, duration: float, memory_delta: Dict[str, float]):
        """Update statistics for a function"""
        if context not in self.function_stats:
            self.function_stats[context] = {
                'calls': 0,
                'total_time': 0,
                'min_time': float('inf'),
                'max_time': 0,
                'memory_deltas': []
            }
        
        stats = self.function_stats[context]
        stats['calls'] += 1
        stats['total_time'] += duration
        stats['min_time'] = min(stats['min_time'], duration)
        stats['max_time'] = max(stats['max_time'], duration)
        stats['memory_deltas'].append(memory_delta)

    def save_stats(self):
        """Save profiling results"""
        # Save detailed stats
        stats_file = self.output_dir / f"detailed_profile_{self.timestamp}.json"
        
        # Process memory stats
        for func_stats in self.function_stats.values():
            memory_deltas = func_stats['memory_deltas']
            avg_deltas = {}
            for key in memory_deltas[0].keys():
                avg_deltas[key] = sum(d[key] for d in memory_deltas) / len(memory_deltas)
            func_stats['avg_memory_delta'] = avg_deltas
            del func_stats['memory_deltas']

        with open(stats_file, 'w') as f:
            json.dump(self.function_stats, f, indent=2)

# Global profiler instance
profiler = DetailedProfiler()