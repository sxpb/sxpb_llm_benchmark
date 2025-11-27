import argparse
import os
import sxpb
import sxpb.jsonutil
import time
from src.prompt_creation import create_prompt
from src.generate_data import (
    generate_json,
    generate_jsonl,
    generate_txtpb,
    generate_xml,
    generate_yaml,
    generate_sxpb,
    generate_toon,
)
from typing import Any, Dict, List, Optional
from src.llm_api import LlmApi, LlamaCppApi, OpenAiApi

# Global variable to hold the LLM instance
llm: Optional[LlmApi] = None

SUPPORTED_FORMATS = sorted(
    [
        "json",
        "json.compact",
        "json.oneline",
        "jsonl",
        "jsonl.compact",
        "sxpb",
        "sxpb.compact",
        "sxpb.oneline",
        "txtpb",
        "txtpb.compact",
        "txtpb.oneline",
        "toon",
        "toon.compact",
        "xml",
        "xml.compact",
        "yaml",
    ]
)


def call_llm(prompt: str, log_context_file: Optional[str] = None) -> Dict[str, Any]:
    """Calls the local LLM to get a response and returns answer and token usage."""
    if llm is None:
        raise Exception("LLM not initialized. Please call initialize_llm() first.")

    try:
        llm_response = llm.call_llm(prompt)
    except Exception as e:
        print(f"Error calling LLM: {e}")
        llm_response = {"answer": f"ERROR_LLM_TIMEOUT: {e}", "prompt_tokens": 0}

    llm_answer = llm_response["answer"]

    if log_context_file:
        with open(log_context_file, "a", encoding="utf-8") as f:
            f.write("--- Input ---\n")
            f.write(prompt)
            f.write("\n\n--- Output ---\n")
            f.write(llm_answer)
            f.write("\n\n")

    return llm_response


def format_for_display(format_name: str) -> str:
    """Formats the given format name for display."""
    if "sxpb" in format_name:
        return "SxPB"
    return format_name.upper()


def run_benchmark(
    data_format: str,
    qa_data: List[Dict[str, Any]],
    raw_file_content: str,
    log_context_file: Optional[str] = None,
) -> Dict[str, Any]:
    print(f"\n--- Running benchmark for {format_for_display(data_format)} ---")

    if not qa_data:
        print("No QA data found, skipping benchmark.")
        return {
            "accuracy": 0,
            "average_time": 0,
            "qa_results": [],
            "total_prompt_tokens": 0,
        }

    qa_results: List[Dict[str, Any]] = []
    total_time = 0.0
    correct_predictions = 0
    total_prompt_tokens = 0
    total_prompt_bytes = 0

    for item in qa_data:
        question = item["question"]
        expected_answer = item["answer"]

        prompt = create_prompt(data_format, raw_file_content, question)
        prompt_bytes = len(prompt.encode("utf-8"))
        total_prompt_bytes += prompt_bytes

        start_time = time.time()
        llm_response = call_llm(prompt, log_context_file=log_context_file)
        end_time = time.time()

        llm_answer = llm_response["answer"]
        total_prompt_tokens += llm_response["prompt_tokens"]

        time_taken = end_time - start_time
        total_time += time_taken

        # The final answer should be on the last line, starting with "Final Answer:"
        llm_answer_lines = llm_answer.strip().split("\n")
        final_answer_line = llm_answer_lines[-1].strip()

        # Clean potential markdown bolding
        final_answer_line = final_answer_line.replace("**", "")

        is_correct = False
        if final_answer_line.lower().startswith("final answer:"):
            final_answer = final_answer_line[len("Final Answer:") :].strip()
            is_correct = expected_answer in final_answer

        if is_correct:
            correct_predictions += 1

        print(f"LLM Answer: {final_answer_line}")
        print(f"Expected Answer: {expected_answer}")
        print(f"Result: {'CORRECT' if is_correct else 'INCORRECT'}")

        qa_results.append(
            {
                "question": question,
                "expected_answer": expected_answer,
                "llm_answer": llm_answer,
                "time_taken": time_taken,
                "is_correct": is_correct,
                "prompt_tokens": llm_response["prompt_tokens"],
                "prompt_bytes": prompt_bytes,
            }
        )

    accuracy = (correct_predictions / len(qa_data)) * 100 if qa_data else 0
    avg_time = total_time / len(qa_data) if qa_data else 0

    print(f"\nAccuracy: {accuracy:.2f}%")
    print(f"Average time per question: {avg_time:.4f} seconds")
    print(f"Total input tokens: {total_prompt_tokens}")

    return {
        "accuracy": accuracy,
        "average_time": avg_time,
        "qa_results": qa_results,
        "correct_predictions": correct_predictions,
        "total_questions": len(qa_data),
        "total_prompt_tokens": total_prompt_tokens,
        "total_prompt_bytes": total_prompt_bytes,
    }


def _check_format(value: str) -> str:
    """Checks if the given format is supported, case-insensitively."""
    lower_value = value.lower()
    if lower_value in SUPPORTED_FORMATS:
        return lower_value
    raise argparse.ArgumentTypeError(
        f"invalid choice: '{value}' (choose from {', '.join(SUPPORTED_FORMATS)})"
    )


def main() -> None:
    global llm
    parser = argparse.ArgumentParser(description="Run a benchmark for a given dataset.")
    parser.add_argument(
        "benchmark_name",
        type=str,
        nargs="?",
        default="all",
        help="The name of the benchmark to run (e.g., exoplanet or chronocrystal). Defaults to 'all'.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ggml-org/gemma-3-270m-it-GGUF/gemma-3-270m-it-Q8_0.gguf",
        help="The model to use. For llama-cpp, this can be a local file path or a Hugging Face repo ID. For OpenAI/OpenRouter, this is the model name.",
    )
    parser.add_argument(
        "--api-key",
        "--api_key",
        type=str,
        default=None,
        help="API key for OpenAI or OpenRouter. Required if --api-url is set.",
    )
    parser.add_argument(
        "--api-url",
        "--api_url",
        type=str,
        default=None,
        help="If specified, runs the benchmark against an OpenAI-compatible API at this URL. Otherwise, runs locally using llama-cpp-python.",
    )
    parser.add_argument(
        "--format",
        type=_check_format,
        default=None,
        help="If specified, runs the benchmark only for this data format.",
    )
    parser.add_argument(
        "--log-dir",
        "--log_dir",
        type=str,
        default=None,
        help="If specified, saves the benchmark results and context logs to this directory.",
    )
    parser.add_argument(
        "--completion-token-limit",
        "--completion_token_limit",
        type=int,
        default=4000,
        help="The maximum number of tokens to generate for each completion. Set to 0 for no limit.",
    )
    parser.add_argument(
        "--ollama-compatibility-on",
        "--ollama_compatibility_on",
        action="store_true",
        help="Enable Ollama compatibility mode. This will cause a completion_token_limit of 0 to be sent as -1.",
    )
    args = parser.parse_args()
    selected_benchmark: str = args.benchmark_name

    log_context_file: Optional[str] = None
    if args.log_dir:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        log_context_file = os.path.join(args.log_dir, "context_log.txt")
        # Clear the log file at the beginning of the run
        with open(log_context_file, "w", encoding="utf-8") as f:
            pass

    if args.api_url:
        if not args.api_key:
            raise ValueError("--api-key is required when using --api-url.")
        llm = OpenAiApi(
            model=args.model,
            api_key=args.api_key,
            base_url=args.api_url,
            completion_token_limit=args.completion_token_limit,
            ollama_compatibility_on=args.ollama_compatibility_on,
        )
    else:
        llm = LlamaCppApi(
            args.model,
            completion_token_limit=args.completion_token_limit,
            ollama_compatibility_on=args.ollama_compatibility_on,
        )

    script_dir: str = os.path.dirname(os.path.abspath(__file__))
    data_dir: str = os.path.join(script_dir, "../data")

    benchmark_names: List[str]
    if selected_benchmark == "all":
        benchmark_names = [
            d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
        ]
    else:
        benchmark_names = [selected_benchmark]

    overall_results: Dict[str, Dict[str, int]] = {
        fmt: {
            "correct_predictions": 0,
            "total_questions": 0,
            "total_prompt_tokens": 0,
            "total_prompt_bytes": 0,
        }
        for fmt in SUPPORTED_FORMATS
    }

    # Determine the maximum length of the format names for alignment
    max_format_len = 0
    if SUPPORTED_FORMATS:
        max_format_len = len(max(SUPPORTED_FORMATS, key=len))

    for benchmark_name in benchmark_names:
        print(f"\n===== Running Benchmark: {benchmark_name} =====")
        benchmark_dir: str = os.path.join(data_dir, benchmark_name)

        if not os.path.isdir(benchmark_dir):
            print(f"Error: Benchmark directory not found at {benchmark_dir}")
            continue

        # Load QA data
        qa_sxpb_file_path: str = os.path.join(benchmark_dir, "qa.sxpb")
        if not os.path.isfile(qa_sxpb_file_path):
            print(f"Error: QA file not found at {qa_sxpb_file_path}")
            continue
        with open(qa_sxpb_file_path, "r", encoding="utf-8") as f:
            raw_qa_sxpb_content: str = f.read()
        qa_data = sxpb.loads(raw_qa_sxpb_content)
        if qa_data is None:
            print(f"Error: Could not parse QA data from {qa_sxpb_file_path}")
            continue

        # Load and parse the source of truth: data.sxpb
        sxpb_file_path: str = os.path.join(benchmark_dir, "data.sxpb")
        if not os.path.isfile(sxpb_file_path):
            print(f"Error: data.sxpb not found in {benchmark_dir}")
            continue
        with open(sxpb_file_path, "r", encoding="utf-8") as f:
            raw_sxpb_content: str = f.read()

        try:
            native_data = sxpb.loads(raw_sxpb_content, precise=True)
        except Exception as e:
            print(f"Error parsing {sxpb_file_path}: {e}")
            continue
        if native_data is None:
            print(f"Error: Could not parse data from {sxpb_file_path}")
            continue

        plain_data = sxpb.jsonutil.to_plain_types(native_data)

        data_list: List[Dict[str, Any]]
        root_element_name = "item"
        plain_data_dict: Dict[str, Any]

        if not isinstance(plain_data, dict):
            print(f"Error: data.sxpb root is not a dictionary in {sxpb_file_path}")
            continue
        plain_data_dict = plain_data
        root_element_name = list(plain_data_dict.keys())[0]
        data_list = plain_data_dict[root_element_name]

        generated_data: Dict[str, str] = {
            "json": generate_json(plain_data_dict),
            "json.compact": generate_json(plain_data_dict, mode="compact"),
            "json.oneline": generate_json(plain_data_dict, mode="oneline"),
            "jsonl": generate_jsonl(data_list),
            "jsonl.compact": generate_jsonl(data_list, mode="compact"),
            "sxpb": generate_sxpb(native_data),
            "sxpb.compact": generate_sxpb(native_data, mode="compact"),
            "sxpb.oneline": generate_sxpb(native_data, mode="oneline"),
            "txtpb": generate_txtpb(data_list, root_element_name),
            "txtpb.compact": generate_txtpb(
                data_list, root_element_name, mode="compact"
            ),
            "txtpb.oneline": generate_txtpb(
                data_list, root_element_name, mode="oneline"
            ),
            "toon": generate_toon(plain_data_dict),
            "toon.compact": generate_toon(plain_data_dict, mode="compact"),
            "yaml": generate_yaml(plain_data_dict),
            "xml": generate_xml(data_list, root_element_name),
            "xml.compact": generate_xml(data_list, root_element_name, mode="compact"),
        }

        results: Dict[str, Any] = {}
        formats_to_run = [args.format] if args.format else SUPPORTED_FORMATS

        for data_format in formats_to_run:
            raw_content = generated_data[data_format]
            if not qa_data or not isinstance(qa_data, list):
                print("Warning: qa_data is not a list, skipping benchmark.")
                continue
            result = run_benchmark(
                data_format, qa_data, raw_content, log_context_file=log_context_file
            )
            results[data_format] = result
            overall_results[data_format]["correct_predictions"] += result[
                "correct_predictions"
            ]
            overall_results[data_format]["total_questions"] += result["total_questions"]
            overall_results[data_format]["total_prompt_tokens"] += result[
                "total_prompt_tokens"
            ]
            overall_results[data_format]["total_prompt_bytes"] += result[
                "total_prompt_bytes"
            ]

        print(f"\n--- Benchmark Summary for {benchmark_name} ---")
        for format_name in sorted(results.keys()):
            result_data = results[format_name]
            accuracy: float = result_data["accuracy"]
            avg_time: float = result_data["average_time"]
            print(
                f"{format_for_display(format_name):<{max_format_len}}: Accuracy: {accuracy:.2f}%, Avg Time: {avg_time:.4f}s"
            )

        if args.log_dir:
            results_file: str = os.path.join(
                args.log_dir, f"benchmark_results_{benchmark_name}.sxpb"
            )
            with open(results_file, "w", encoding="utf-8") as f:
                sxpb_output = sxpb.dumps(results, indent=1)
                f.write(sxpb_output)
            print(f"\nBenchmark results for {benchmark_name} saved to {results_file}")

    if selected_benchmark == "all":
        print("\n--- Overall Benchmark Summary ---")
        for fmt in sorted(overall_results.keys()):
            data = overall_results[fmt]
            correct = data["correct_predictions"]
            total = data["total_questions"]
            prompt_tokens = data["total_prompt_tokens"]
            prompt_bytes = data["total_prompt_bytes"]
            overall_accuracy = (correct / total) * 100 if total > 0 else 0
            print(
                f"{format_for_display(fmt):<{max_format_len}}: "
                f"Accuracy: {overall_accuracy:.2f}% ({correct}/{total}) | "
                f"Input Tokens: {prompt_tokens} | Input Bytes: {prompt_bytes}"
            )


if __name__ == "__main__":
    main()
