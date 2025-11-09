import textwrap


def create_prompt(data_format: str, raw_file_content: str, question: str) -> str:
    """Creates a formatted prompt for the LLM."""
    # Use the base file extension in the prompt (e.g., "sxpb" instead of "sxpb.compact")
    file_extension = data_format.split(".")[0]

    prompt = textwrap.dedent(f"""
        You are an AI fact lookup engine who answers questions about given reference data.
        Provide the answer directly in the last line of your response, starting with `Final Answer:`.

        ### Example
        Question: What is the capital of Michigan?
        ... optional scratch space ...
        Final Answer: Lansing

        #### Reference Data
        ```{file_extension}
    """).strip()
    prompt += "\n" + raw_file_content.strip() + "\n"
    prompt += textwrap.dedent(f"""
        ```

        #### Question
        Question: {question}
    """).strip()
    return prompt
