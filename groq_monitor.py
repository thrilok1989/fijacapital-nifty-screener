# groq_monitor.py
import time
from collections import defaultdict

class GroqUsageMonitor:
    def __init__(self):
        self.usage_stats = defaultdict(lambda: {
            'count': 0,
            'total_tokens': 0,
            'total_time': 0,
            'errors': 0
        })
    
    def record_usage(self, model: str, tokens: int, duration: float, success: bool = True):
        stats = self.usage_stats[model]
        stats['count'] += 1
        stats['total_tokens'] += tokens
        stats['total_time'] += duration
        if not success:
            stats['errors'] += 1
    
    def get_stats(self):
        return dict(self.usage_stats)
    
    def print_summary(self):
        print("\n🤖 GROQ USAGE SUMMARY")
        print("="*40)
        for model, stats in self.usage_stats.items():
            avg_time = stats['total_time'] / stats['count'] if stats['count'] > 0 else 0
            avg_tokens = stats['total_tokens'] / stats['count'] if stats['count'] > 0 else 0
            print(f"\n{model}:")
            print(f"  Requests: {stats['count']}")
            print(f"  Avg Tokens: {avg_tokens:.0f}")
            print(f"  Avg Time: {avg_time:.2f}s")
            print(f"  Errors: {stats['errors']}")
