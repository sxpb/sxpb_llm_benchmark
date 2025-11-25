from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, cast
import psutil
from llama_cpp import Llama
from llama_cpp.llama_types import ChatCompletionRequestMessage
from huggingface_hub import hf_hub_download
import os
import openai


DEFAULT_N_THREADS = 4


class LlmApi(ABC):
    @abstractmethod
    def call_llm(self, prompt: str) -> Dict[str, Any]:
        pass


class LlamaCppApi(LlmApi):
    def __init__(
        self, model_identifier: str, completion_token_limit: Optional[int] = 4000
    ):
        self.llm: Optional[Llama] = None
        self.completion_token_limit = completion_token_limit
        self._initialize_llm(model_identifier)

    def _initialize_llm(self, model_identifier: str) -> None:
        """Initializes the LLM instance."""
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

    def call_llm(self, prompt: str) -> Dict[str, Any]:
        if self.llm is None:
            raise Exception("LLM not initialized.")

        max_tokens = self.completion_token_limit
        if max_tokens == 0:
            max_tokens = None

        messages: List[Dict[str, str]] = [{"role": "user", "content": prompt}]
        output: Any = self.llm.create_chat_completion(
            cast(List[ChatCompletionRequestMessage], messages),
            max_tokens=max_tokens,
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
    ):
        if not api_key:
            raise ValueError("API key is required for OpenAI API.")
        self.model = model
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.completion_token_limit = completion_token_limit

    def call_llm(self, prompt: str) -> Dict[str, Any]:
        max_tokens = self.completion_token_limit
        if max_tokens == 0:
            max_tokens = None
        messages: List[Dict[str, str]] = [{"role": "user", "content": prompt}]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content
        llm_answer = content.strip() if content else ""
        prompt_tokens = response.usage.prompt_tokens if response.usage else 0
        return {"answer": llm_answer, "prompt_tokens": prompt_tokens}
