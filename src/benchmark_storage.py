import json
import os
from typing import List, Dict, Any

RESULTS_DIR = "results/toon/models"


def load_model_results(model_id: str) -> List[Dict[str, Any]]:
    filepath = os.path.join(RESULTS_DIR, f"{model_id}.json")
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return []


def save_model_results(model_id: str, results: List[Dict[str, Any]]):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    filepath = os.path.join(RESULTS_DIR, f"{model_id}.json")
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)


def get_all_model_results() -> Dict[str, List[Dict[str, Any]]]:
    results = {}
    if os.path.exists(RESULTS_DIR):
        for filename in os.listdir(RESULTS_DIR):
            if filename.endswith(".json"):
                model_id = filename[:-5]
                results[model_id] = load_model_results(model_id)
    return results


def has_model_results(model_id: str) -> bool:
    filepath = os.path.join(RESULTS_DIR, f"{model_id}.json")
    return os.path.exists(filepath)
