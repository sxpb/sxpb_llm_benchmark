import unittest
from typing import Dict, Any, cast
from src.bench.toon.datasets import get_toon_datasets
from src.bench.toon.questions import generate_questions, Question
from src.bench.toon.evaluation import evaluate_question, clean_llm_answer
from src.llm_api import LlmApi

class MockLlmApi(LlmApi):
    def __init__(self, answer: str):
        self.answer = answer

    def call_llm(self, prompt: str) -> Dict[str, Any]:
        return {"answer": self.answer, "prompt_tokens": 10}

class TestToonBenchmark(unittest.TestCase):

    def test_get_toon_datasets(self):
        datasets = get_toon_datasets()
        self.assertTrue(len(datasets) > 0)
        names = [d["name"] for d in datasets]
        self.assertIn("tabular", names)
        self.assertIn("nested", names)
        self.assertIn("analytics", names)
        self.assertIn("github", names)
        self.assertIn("event-logs", names)
        self.assertIn("nested-config", names)

    def test_generate_questions(self):
        datasets = get_toon_datasets(fullsize_ratio=0.1) # Small ratio for speed
        questions = generate_questions(datasets)
        self.assertTrue(len(questions) > 0)

        # Check basic structure of a question
        q = questions[0]
        self.assertIn("id", q)
        self.assertIn("prompt", q)
        self.assertIn("groundTruth", q)
        self.assertIn("type", q)
        self.assertIn("dataset", q)
        self.assertIn("answerType", q)

    def test_clean_llm_answer(self):
        self.assertEqual(clean_llm_answer("Final Answer: 42"), "42")
        self.assertEqual(clean_llm_answer("**42**"), "42")
        self.assertEqual(clean_llm_answer("Answer: 42"), "42")
        self.assertEqual(clean_llm_answer("  42  "), "42")

    def test_evaluate_question_correct(self):
        question = cast(Question, {
            "id": "q1",
            "prompt": "What is 2+2?",
            "groundTruth": "4",
            "type": "retrieval",
            "dataset": "test",
            "answerType": "integer"
        })
        llm_api = MockLlmApi("Final Answer: 4")
        result = evaluate_question(question, "json", "{}", llm_api)
        self.assertTrue(result["isCorrect"])
        self.assertEqual(result["actual"], "4")

    def test_evaluate_question_incorrect(self):
        question = cast(Question, {
            "id": "q1",
            "prompt": "What is 2+2?",
            "groundTruth": "4",
            "type": "retrieval",
            "dataset": "test",
            "answerType": "integer"
        })
        llm_api = MockLlmApi("Final Answer: 5")
        result = evaluate_question(question, "json", "{}", llm_api)
        self.assertFalse(result["isCorrect"])
        self.assertEqual(result["actual"], "5")

    def test_evaluate_question_json_output(self):
        question = cast(Question, {
            "id": "q1",
            "prompt": "What is 2+2?",
            "groundTruth": "4",
            "type": "retrieval",
            "dataset": "test",
            "answerType": "integer"
        })
        # LLM returns JSON as requested
        llm_api = MockLlmApi('```json\n{"answer": "4"}\n```')
        result = evaluate_question(question, "json", "{}", llm_api, use_json_output=True)
        self.assertTrue(result["isCorrect"])
        self.assertEqual(result["actual"], "4")

    def test_evaluate_question_json_output_fallback(self):
        question = cast(Question, {
            "id": "q1",
            "prompt": "What is 2+2?",
            "groundTruth": "4",
            "type": "retrieval",
            "dataset": "test",
            "answerType": "integer"
        })
        # LLM fails to return JSON, but returns correct answer text
        llm_api = MockLlmApi('Final Answer: 4')
        result = evaluate_question(question, "json", "{}", llm_api, use_json_output=True)
        self.assertTrue(result["isCorrect"])
        self.assertEqual(result["actual"], "4")

if __name__ == '__main__':
    unittest.main()
