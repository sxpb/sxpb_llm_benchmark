from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, cast
import psutil
from huggingface_hub import hf_hub_download
import os
import openai
import argparse


DEFAULT_N_THREADS = 4


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


class LlamaCppApi(LlmApi):
    def __init__(
        self,
        model_identifier: str,
        completion_token_limit: Optional[int] = 4000,
        ollama_compatibility_on: bool = False,
    ):
        super().__init__(
            completion_token_limit=completion_token_limit,
            ollama_compatibility_on=ollama_compatibility_on,
        )
        self.llm: Optional["Llama"] = None
        self._initialize_llm(model_identifier)

    def _initialize_llm(self, model_identifier: str) -> None:
        """Initializes the LLM instance."""
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python is required for local LLM execution. "
                "Please install it with: pip install llama-cpp-python"
            )

        model_path: str

        if os.path.exists(model_identifier):
            print(f"Loading model from local path: {model_identifier}")
            model_path = model_identifier
        else:
            print(
                f"Model identifier is not a local path, treating as Hugging Face repo: {model_identifier}"
            )
            try:
                repo_id, filename = model_identifier.rsplit("/", 1)
            except ValueError:
                raise ValueError(
                    "Invalid Hugging Face model identifier. Expected format: 'repo_id/file_name'"
                )

            print(f"Downloading model: {repo_id}/{filename}")
            model_path = hf_hub_download(repo_id=repo_id, filename=filename)
            print(f"Model downloaded to: {model_path}")

        try:
            physical_cores = psutil.cpu_count(logical=False)
            n_threads = (
                physical_cores if physical_cores is not None else DEFAULT_N_THREADS
            )
            logical_cores = psutil.cpu_count(logical=True)
            n_threads_batch = (
                logical_cores if logical_cores is not None else DEFAULT_N_THREADS
            )
        except Exception:
            n_threads = DEFAULT_N_THREADS
            n_threads_batch = DEFAULT_N_THREADS

        self.llm = Llama(
            model_path=model_path,
            n_ctx=8192,
            n_threads=n_threads,
            n_threads_batch=n_threads_batch,
            n_gpu_layers=0,
            verbose=False,
        )

    def _raw_call_llm(self, prompt: str) -> Dict[str, Any]:
        if self.llm is None:
            raise Exception("LLM not initialized.")

        # Local import to avoid top-level dependency
        from llama_cpp.llama_types import ChatCompletionRequestMessage

        messages: List[Dict[str, str]] = [{"role": "user", "content": prompt}]
        output: Any = self.llm.create_chat_completion(
            cast(List[ChatCompletionRequestMessage], messages),
            max_tokens=self.max_tokens,
        )
        assert isinstance(output, dict)
        content = output["choices"][0]["message"]["content"]
        assert isinstance(content, str)
        llm_answer: str = content.strip()
        usage = output["usage"]
        prompt_tokens = usage["prompt_tokens"]
        return {"answer": llm_answer, "prompt_tokens": prompt_tokens}


class OpenAiApi(LlmApi):
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: Optional[str] = None,
        completion_token_limit: Optional[int] = 4000,
        ollama_compatibility_on: bool = False,
    ):
        super().__init__(
            completion_token_limit=completion_token_limit,
            ollama_compatibility_on=ollama_compatibility_on,
        )
        if not api_key:
            raise ValueError("API key is required for OpenAI API.")
        self.model = model
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)

    def _raw_call_llm(self, prompt: str) -> Dict[str, Any]:
        messages: List[Dict[str, str]] = [{"role": "user", "content": prompt}]
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
    if api_key:
        return OpenAiApi(
            model=model,
            api_key=api_key,
            base_url=api_url,
            completion_token_limit=completion_token_limit,
            ollama_compatibility_on=ollama_compatibility_on,
        )
    else:
        return LlamaCppApi(
            model_identifier=model,
            completion_token_limit=completion_token_limit,
            ollama_compatibility_on=ollama_compatibility_on,
        )


def add_llm_args(parser: argparse.ArgumentParser) -> None:
    """Adds standard LLM arguments to an ArgumentParser."""
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
