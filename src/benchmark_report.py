from typing import List, Dict, Any

def calculate_format_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    format_names = list(set(r['format'] for r in results))
    format_results = []

    for format_name in format_names:
        format_results_list = [r for r in results if r['format'] == format_name]
        correct_count = sum(1 for r in format_results_list if r['isCorrect'])
        total_count = len(format_results_list)
        accuracy = correct_count / total_count if total_count > 0 else 0

        total_tokens = sum(r.get('prompt_tokens', 0) for r in format_results_list)
        avg_tokens = total_tokens / total_count if total_count > 0 else 0

        average_latency = sum(r['latencyMs'] for r in format_results_list) / total_count if total_count > 0 else 0

        format_results.append({
            "format": format_name,
            "accuracy": accuracy,
            "totalTokens": round(avg_tokens),
            "averageLatency": round(average_latency),
            "correctCount": correct_count,
            "totalCount": total_count,
        })

    return sorted(format_results, key=lambda x: x['accuracy'], reverse=True)


def generate_toon_report(
    results: List[Dict[str, Any]],
    format_results: List[Dict[str, Any]],
) -> str:
    # This is a simplified version of the report generation.
    report = "# Toon Benchmark Report\n\n"
    for fr in format_results:
        report += f"## {fr['format']}\n"
        report += f"- Accuracy: {fr['accuracy']:.2%}\n"
        report += f"- Average Tokens: {fr['totalTokens']}\n"
        report += f"- Average Latency: {fr['averageLatency']}ms\n"
        report += "\n"
    return report
