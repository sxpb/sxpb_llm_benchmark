import argparse
from src.benchmark_datasets import get_accuracy_datasets
from src.benchmark_questions import generate_questions
from src.benchmark_evaluation import evaluate_question
from src.benchmark_storage import save_model_results, has_model_results, get_all_model_results
from src.benchmark_report import calculate_format_results, generate_accuracy_report
from src.generate_data import generate_json, generate_yaml, generate_xml
from src.llm_api import LlamaCppApi, OpenAiApi
import sxpb

def main():
    parser = argparse.ArgumentParser(description="Run retrieval accuracy benchmark.")
    parser.add_argument("--model", type=str, required=True, help="Model ID to use for the benchmark.")
    parser.add_argument("--api-key", type=str, help="API key for OpenAI API.")
    parser.add_argument("--api-url", type=str, help="Base URL for OpenAI-compatible API.")
    parser.add_argument("--fullsize-ratio", type=float, default=1.0, help="Ratio to scale the dataset sizes.")
    args = parser.parse_args()

    if args.api_key:
        llm_api = OpenAiApi(model=args.model, api_key=args.api_key, base_url=args.api_url)
    else:
        llm_api = LlamaCppApi(model_identifier=args.model)

    model_id = args.model.replace("/", "_")

    accuracy_datasets = get_accuracy_datasets(args.fullsize_ratio)

    questions = generate_questions(accuracy_datasets)

    formatters = {
        "sxpb": lambda data: sxpb.dumps(data, indent=1),
        "json": lambda data: generate_json(data, mode="pretty"),
        "yaml": lambda data: generate_yaml(data),
        "xml": lambda data: generate_xml(list(data.values())[0], root_element_name=list(data.keys())[0]),
    }

    if has_model_results(model_id):
        print(f"Results for model {model_id} already exist. Skipping.")
    else:
        results = []
        total_questions = len(questions)
        for format_name, formatter in formatters.items():
            print(f"Running benchmark for format: {format_name}")
            for i, question in enumerate(questions):
                print(f"  Running {i+1}/{total_questions}: {question['prompt']}")
                dataset = next((d for d in accuracy_datasets if d["name"] == question["dataset"]), None)
                if not dataset:
                    continue

                formatted_data = formatter(dataset["data"])
                result = evaluate_question(question, format_name, formatted_data, llm_api)
                results.append(result)

        save_model_results(model_id, results)

    all_results = get_all_model_results()
    flat_results = [item for sublist in all_results.values() for item in sublist]

    format_results = calculate_format_results(flat_results)
    report = generate_accuracy_report(flat_results, format_results)

    with open("results/retrieval-accuracy.md", "w") as f:
        f.write(report)

if __name__ == "__main__":
    main()
