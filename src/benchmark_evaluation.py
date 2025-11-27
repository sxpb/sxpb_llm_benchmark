from typing import Dict, Any, cast, Optional
from src.benchmark_questions import Question
from src.answer_normalization import compare_answers
from src.llm_api import LlmApi
import time
import json
import re

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


def clean_llm_answer(answer: str) -> str:
    """
    Cleans the LLM answer by removing common prefixes and markdown formatting.
    """
    answer = answer.strip()

    # Remove markdown bolding (e.g., **Answer**)
    answer = answer.replace("**", "")

    # Remove common prefixes
    lower_answer = answer.lower()
    if lower_answer.startswith("final answer:"):
        answer = answer[len("Final Answer:") :].strip()
    elif lower_answer.startswith("answer:"):
        answer = answer[len("Answer:") :].strip()

    return answer


def extract_json_answer(text: str) -> Optional[str]:
    """
    Attempts to extract the answer from a JSON object in the text.
    Returns the answer string if found and valid, otherwise None.
    """
    # Try finding JSON code block first
    json_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_block:
        text_to_parse = json_block.group(1)
    else:
        # Try finding just a JSON object
        json_obj = re.search(r"(\{.*\})", text, re.DOTALL)
        if json_obj:
            text_to_parse = json_obj.group(1)
        else:
            text_to_parse = text

    try:
        data = json.loads(text_to_parse)
        if isinstance(data, dict) and "answer" in data:
            return str(data["answer"])
    except json.JSONDecodeError:
        pass
    return None


def evaluate_question(
    question: Question,
    format_name: str,
    formatted_data: str,
    llm_api: LlmApi,
    use_json_output: bool = False,
) -> Dict[str, Any]:
    primer = PRIMERS.get(format_name, "")
    fence = FENCE.get(format_name, "")

    base_prompt = f"""
{primer}

Given the following data in {format_name} format:

```{fence}
{formatted_data}
```

Question: {question["prompt"]}
""".strip()

    if use_json_output:
        prompt = (
            base_prompt
            + "\n\n"
            + """
Answer format requirements:
- Format your answer as a JSON object with a single key "answer".
- Example: {"answer": "Lansing"}
""".strip()
        )
    else:
        prompt = (
            base_prompt
            + "\n\n"
            + """
Answer format requirements:
- Provide only the value itself, no explanation
- For numbers: output digits only (no commas, currency symbols, or units)
- For dates/field names: use the exact string from the data
- For lists: output comma-separated values with no spaces

Answer:
""".strip()
        )

    start_time = time.time()

    try:
        response = llm_api.call_llm(prompt)
    except Exception as e:
        print(f"Error calling LLM: {e}")
        response = {"answer": f"ERROR_LLM_TIMEOUT: {e}", "prompt_tokens": 0}

    latency_ms = (time.time() - start_time) * 1000

    raw_actual = response["answer"].strip()
    actual = raw_actual

    if use_json_output:
        extracted = extract_json_answer(raw_actual)
        if extracted is not None:
            actual = extracted
        else:
            # Fallback to cleaning if JSON extraction failed
            actual = clean_llm_answer(raw_actual)
    else:
        actual = clean_llm_answer(raw_actual)

    is_correct, _ = compare_answers(
        actual,
        question["groundTruth"],
        question.get("answerType", "string"),
        cast(Optional[Dict[str, Any]], question.get("normalizationOptions")),
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
