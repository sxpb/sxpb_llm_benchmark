from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import os
import openai
import argparse


class LlmApi(ABC):
    def __init__(
        self,
        completion_token_limit: Optional[int] = 4000,
        ollama_compatibility_on: bool = False,
    ):
        self.max_tokens = completion_token_limit
        if self.max_tokens == 0:
            if ollama_compatibility_on:
                self.max_tokens = -1
            else:
                self.max_tokens = None

    @abstractmethod
    def _raw_call_llm(self, prompt: str) -> Dict[str, Any]:
        """Internal method to implement the raw LLM call."""
        pass

    def call_llm(self, prompt: str) -> Dict[str, Any]:
        """Calls the LLM with error handling."""
        try:
            return self._raw_call_llm(prompt)
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return {"answer": f"ERROR_LLM_TIMEOUT: {e}", "prompt_tokens": 0}


class OpenAiApi(LlmApi):
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        completion_token_limit: Optional[int] = 4000,
        ollama_compatibility_on: bool = False,
    ):
        super().__init__(
            completion_token_limit=completion_token_limit,
            ollama_compatibility_on=ollama_compatibility_on,
        )
        # Some local APIs don't require an API key, so we provide a placeholder.
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or "placeholder"
        self.model = model
        self.client = openai.OpenAI(api_key=self.api_key, base_url=base_url)

    def _raw_call_llm(self, prompt: str) -> Dict[str, Any]:
        messages: List[Any] = [{"role": "user", "content": prompt}]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
        )
        content = response.choices[0].message.content
        llm_answer = content.strip() if content else ""
        prompt_tokens = response.usage.prompt_tokens if response.usage else 0
        return {"answer": llm_answer, "prompt_tokens": prompt_tokens}


def get_llm_api(
    model: str,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    completion_token_limit: int = 4000,
    ollama_compatibility_on: bool = False,
) -> LlmApi:
    """
    Factory function to get the appropriate LLM API instance.
    """
    return OpenAiApi(
        model=model,
        api_key=api_key,
        base_url=api_url,
        completion_token_limit=completion_token_limit,
        ollama_compatibility_on=ollama_compatibility_on,
    )


def add_llm_args(parser: argparse.ArgumentParser) -> None:
    """Adds standard LLM arguments to an ArgumentParser."""
    parser.add_argument(
        "--model",
        type=str,
        default="gemma3:4b",
        help="The model name to use on the API.",
    )
    parser.add_argument(
        "--api-key",
        "--api_key",
        type=str,
        default=None,
        help="API key for OpenAI or OpenRouter. Optional for local APIs.",
    )
    parser.add_argument(
        "--api-url",
        "--api_url",
        type=str,
        default="http://localhost:11434/v1",
        help="The URL of the OpenAI-compatible API. Defaults to local Ollama.",
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
