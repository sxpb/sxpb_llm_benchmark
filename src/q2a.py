#!/usr/bin/env python
import argparse
import sxpb
from llm_api import OpenAiApi
import re
import sys


def parse_sxpb_block(text):
    """Parses the last ```sxpb markdown code block from the given text."""
    sxpb_blocks = re.findall(r"```sxpb\n(.*?)\n```", text, re.DOTALL)
    if not sxpb_blocks:
        return None
    return sxpb_blocks[-1]


def main():
    parser = argparse.ArgumentParser(description="Interactive SxPB parser.")
    parser.add_argument("--api-key", required=True, help="OpenAI API key.")
    parser.add_argument("--api-url", help="OpenAI API base URL.")
    parser.add_argument("--model", required=True, help="OpenAI model to use.")
    args = parser.parse_args()

    api = OpenAiApi(model=args.model, api_key=args.api_key, base_url=args.api_url)
    prompt = sys.stdin.read()

    max_retries = 10
    retries = 0
    while retries < max_retries:
        print("Sending prompt to LLM...", file=sys.stderr)
        response = api.call_llm(prompt)
        llm_answer = response["answer"]
        print(f"LLM response: {llm_answer}", file=sys.stderr)

        sxpb_content = parse_sxpb_block(llm_answer)

        if sxpb_content is None:
            prompt = "Please include a SxPB code block like: `(foo: (bar: 1))`"
            continue

        try:
            parsed_data = sxpb.loads(sxpb_content)
            print("Successfully parsed SxPB data:", file=sys.stderr)
            print(parsed_data)
            break
        except Exception as e:
            prompt = f"Syntax error in SxPB block: {e}. Please fix it."

        retries += 1
    else:
        print(f"Max retries ({max_retries}) reached. Exiting.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
