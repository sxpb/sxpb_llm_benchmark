# SxPB LLM Benchmark

This benchmark is designed to evaluate how well a Large Language Model (LLM) understands structured data formatted in SxPB vs others like JSON, YAML, XML, and TXTPB.

## Setup

### PDM Prereq

Install `pdm` on your machine via `apt install pdm`.
Then install package dependencies for this project, omitting the ones needed for development (see [CONTRIBUTING.md](CONTRIBUTING.md) for that).

```shell
pdm install --prod
```

If `pdm` is not available directly, then try installing `uv` like `apk add uv` and set up the environment.
You'll have to run `pdm` as `uv tool run pdm`.

```shell
uv venv .venv
uv pip install pdm
uv tool run pdm install --prod
```

## Run

Now run the benchmark!
This will download and run with a [quantized Gemma 3 270M model](https://huggingface.co/ggml-org/gemma-3-270m-GGUF).
To save the benchmark results, use the `--log-dir` flag.

```shell
pdm run benchmark -- --log-dir result
```

### Local GGUF

Other small models can be used as well, as long as they're GGUF files with `chat_template` metadata. You can specify a model from Hugging Face Hub or a local file path using the `--model` flag.

To use a model from Hugging Face, provide the repository and file name in the format `repo_id/file_name`:
```shell
pdm run benchmark --model ggml-org/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_0.gguf
pdm run benchmark --model unsloth/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q4_K_M.gguf
```

To use a model from a local file:
```shell
pdm run benchmark --model /path/to/your/model.gguf
```

### OpenAI API

You can also run benchmarks through an OpenAI-compatible API by providing `--api-url` and `--api-key` flags.
For example, if you're running Ollama on localhost with the `gpt-oss:20b` model installed, run:
```shell
pdm run benchmark --model gpt-oss:20b --api-url http://localhost:11434/v1 --api-key ignored
```
