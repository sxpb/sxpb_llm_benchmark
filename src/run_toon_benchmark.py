import argparse
from src.benchmark_datasets import get_toon_datasets
from src.benchmark_questions import generate_questions
from src.benchmark_evaluation import evaluate_question
from src.benchmark_storage import (
    save_model_results,
    has_model_results,
    get_all_model_results,
)
from src.benchmark_report import calculate_format_results, generate_toon_report
from src.generate_data import generate_json, generate_yaml, generate_xml, generate_toon
from src.llm_api import LlamaCppApi, OpenAiApi
import sxpb


def main():
    parser = argparse.ArgumentParser(description="Run retrieval toon benchmark.")
    parser.add_argument(
        "--model", type=str, required=True, help="Model ID to use for the benchmark."
    )
    parser.add_argument(
        "--api-key", "--api_key", type=str, help="API key for OpenAI API."
    )
    parser.add_argument(
        "--api-url", "--api_url", type=str, help="Base URL for OpenAI-compatible API."
    )
    parser.add_argument(
        "--fullsize-ratio",
        "--fullsize_ratio",
        type=float,
        default=1.0,
        help="Ratio to scale the dataset sizes.",
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

    if args.api_key:
        llm_api = OpenAiApi(
            model=args.model,
            api_key=args.api_key,
            base_url=args.api_url,
            completion_token_limit=args.completion_token_limit,
            ollama_compatibility_on=args.ollama_compatibility_on,
        )
    else:
        llm_api = LlamaCppApi(
            model_identifier=args.model,
            completion_token_limit=args.completion_token_limit,
            ollama_compatibility_on=args.ollama_compatibility_on,
        )

    model_id = args.model.replace("/", "_")

    toon_datasets = get_toon_datasets(args.fullsize_ratio)

    questions = generate_questions(toon_datasets)

    formatters = {
        "sxpb": lambda data: sxpb.dumps(data, indent=1),
        "json": lambda data: generate_json(data, mode="pretty"),
        "yaml": lambda data: generate_yaml(data),
        "toon": lambda data: generate_toon(data, mode="pretty"),
        "xml": lambda data: generate_xml(
            list(data.values())[0], root_element_name=list(data.keys())[0]
        ),
    }

    if has_model_results(model_id):
        print(f"Results for model {model_id} already exist. Skipping.")
    else:
        results = []
        questions_by_dataset = {}
        for q in questions:
            dataset_name = q["dataset"]
            if dataset_name not in questions_by_dataset:
                questions_by_dataset[dataset_name] = []
            questions_by_dataset[dataset_name].append(q)

        total_evaluations = len(questions) * len(formatters)
        evaluations_processed = 0
        for dataset in toon_datasets:
            dataset_name = dataset["name"]
            print(f"Running benchmark for dataset: {dataset_name}")
            dataset_questions = questions_by_dataset.get(dataset_name, [])
            if not dataset_questions:
                continue

            for format_name, formatter in formatters.items():
                print(f"  Running benchmark for format: {format_name}")
                formatted_data = formatter(dataset["data"])
                for question in dataset_questions:
                    evaluations_processed += 1
                    print(
                        f"    Running {evaluations_processed}/{total_evaluations}: {question['prompt']}",
                        end="",
                        flush=True,
                    )
                    result = evaluate_question(
                        question, format_name, formatted_data, llm_api
                    )
                    if result["isCorrect"]:
                        print(f" -- {question['id']} PASS")
                    else:
                        print(f" -- {question['id']} FAIL")
                    results.append(result)

        save_model_results(model_id, results)

    all_results = get_all_model_results()
    flat_results = [item for sublist in all_results.values() for item in sublist]

    format_results = calculate_format_results(flat_results)
    report = generate_toon_report(flat_results, format_results)

    with open("results/toon-benchmark.md", "w") as f:
        f.write(report)


if __name__ == "__main__":
    main()
