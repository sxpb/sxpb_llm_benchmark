from typing import Dict, Any
from src.benchmark_questions import Question
from src.answer_normalization import compare_answers
from src.llm_api import LlmApi
import time

PRIMERS: Dict[str, str] = {
    "sxpb": "SxPB: S-expression based format.",
    "json": "JSON: Strict JSON objects/arrays with repeated keys per row.",
    "yaml": "YAML: Indentation-based key/value and lists (- items).",
    "xml": "XML: Tag-based tree structure with nested elements.",
}

FENCE: Dict[str, str] = {
    "sxpb": "sxpb",
    "json": "json",
    "yaml": "yaml",
    "xml": "xml",
}


def evaluate_question(
    question: Question,
    format_name: str,
    formatted_data: str,
    llm_api: LlmApi,
) -> Dict[str, Any]:
    primer = PRIMERS.get(format_name, "")
    fence = FENCE.get(format_name, "")

    prompt = f"""
{primer}

Given the following data in {format_name} format:

```{fence}
{formatted_data}
```

Question: {question["prompt"]}

Answer format requirements:
- Provide only the value itself, no explanation
- For numbers: output digits only (no commas, currency symbols, or units)
- For dates/field names: use the exact string from the data
- For lists: output comma-separated values with no spaces

Answer:
""".strip()

    start_time = time.time()

    response = llm_api.call_llm(prompt)

    latency_ms = (time.time() - start_time) * 1000

    actual = response["answer"].strip()

    is_correct, _ = compare_answers(
        actual,
        question["groundTruth"],
        question.get("answerType", "string"),
        question.get("normalizationOptions"),
    )

    return {
        "questionId": question["id"],
        "format": format_name,
        "model": llm_api.model if hasattr(llm_api, "model") else "gguf-local",
        "expected": question["groundTruth"],
        "actual": actual,
        "isCorrect": is_correct,
        "latencyMs": latency_ms,
        "prompt_tokens": response.get("prompt_tokens", 0),
    }
