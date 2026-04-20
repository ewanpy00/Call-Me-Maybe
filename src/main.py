from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Union
import torch
import time

from llm_sdk.llm_sdk import Small_LLM_Model
from pydantic import ValidationError
import argparse

from src.validators import FunctionCallAnswer
from src.validators import FunctionDefinition, PromptItem, validate_answer
from src.json_helpers import jsonl_to_json_array, load_json
from src.json_helpers import extract_first_json_object

MAX_NEW_TOKENS = 256
MAX_RETRIES = 3



def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Project Runner")
    parser.add_argument(
                        "-function_definition",
                        type=str,
                        default='data/input/functions_definition.json',
                        help="Path to function's definitions")
    parser.add_argument(
                        "-input",
                        type=str,
                        default='data/input/function_calling_tests.json',
                        help="Path to input file")
    parser.add_argument(
                        "-output",
                        type=str,
                        default='data/output/result.json',
                        help="Path to output file")
    args = parser.parse_args()

    return args


def build_system_prompt(functions: list[FunctionDefinition]) -> str:
    return f"""You are a function calling assistant.
Return one strict JSON object only, no markdown and no extra text.
The JSON must have this shape:
{{"name":"<function_name>", "parameters":{{...}}}}

Available functions:
{json.dumps([f.model_dump() for f in functions], indent=2)}

If none of the functions match or parameters are invalid just set 'None' as 'name' value
"""


def generate_answer(
    model: Small_LLM_Model,
    system_prompt: str,
    user_query: str,
) -> Union[str, None]:
    user_query = user_query.replace("'", '"')
    final_prompt = f"{system_prompt}\nUser query: {user_query}\nResult: "
    ids = model.encode(final_prompt)[0].tolist()
    generated_text = ""
    for _ in range(MAX_NEW_TOKENS):
        logits = model.get_logits_from_input_ids(ids)
        next_token_id = int(torch.tensor(logits).argmax().item())
        ids.append(next_token_id)

        token_text = model.decode([next_token_id])
        generated_text += token_text
        if "None" in token_text:
            return None
        if "}" in token_text:
            maybe_json = extract_first_json_object(generated_text)
            if maybe_json is not None:
                return maybe_json

    full_generated = model.decode(ids).split(
        "Result: ", maxsplit=1)[-1].strip()
    maybe_json = extract_first_json_object(full_generated)
    return maybe_json or full_generated


def parse_function_call_dict(raw: str) -> dict[str, Any]:
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError(
            "Top-level JSON must be an object with name and parameters")
    if "name" not in parsed or "parameters" not in parsed:
        raise ValueError('Expected keys "name" and "parameters"')
    if not isinstance(parsed["parameters"], dict):
        raise ValueError('"parameters" must be a JSON object')
    return parsed


def iter_prompt_results(
    model: Small_LLM_Model,
    system_prompt: str,
    prompts: list[PromptItem],
    functions_by_name: dict[str, FunctionDefinition],
) -> Iterator[dict[str, Any]]:

    for prompt in prompts:
        user_query = prompt.prompt.strip()

        if not user_query:
            print("Skipping empty prompt")
            continue

        print(f"Processing Query: {user_query}")

        validated_answer: FunctionCallAnswer | None = None
        raw_answer = ""

        for att in range(MAX_RETRIES):
            retry_prompt = system_prompt if att == 0 else system_prompt + "\nONLY VALID JSON."

            raw_answer = generate_answer(model, retry_prompt, user_query)

            if not raw_answer:
                print(f"Retry {att+1}: empty model output")
                continue

            try:
                parsed = parse_function_call_dict(raw_answer)
                validated_answer = validate_answer(parsed, functions_by_name)
                break

            except Exception as exc:
                print(f"Retry {att+1}: {exc}")

        if validated_answer is None:
            print("Skipping prompt due to invalid model output")
            continue

        yield {
            "prompt": user_query,
            **validated_answer.model_dump(),
        }

        print("Done.")
        print("-" * 20)


def main() -> None:
    start_time = time.perf_counter()
    try:
        args = parse()
        OUTPUT_PATH = Path(args.output)
        PROMPTS_PATH = Path(args.input)
        FUNCTIONS_PATH = Path(args.function_definition)
    except SystemExit:
        return

    try:
        prompts_raw = load_json(PROMPTS_PATH)
        functions_raw = load_json(FUNCTIONS_PATH)

        prompts = [PromptItem.model_validate(p) for p in prompts_raw]
        functions = [
            FunctionDefinition.model_validate(f) for f in functions_raw]
        functions_by_name = {f.name: f for f in functions}

    except FileNotFoundError as e:
        print(f"File not found - {e.filename}")
        return
    except json.JSONDecodeError as e:
        print(f"Invalid JSON format in input files - {e}")
        return
    except ValidationError as e:
        print(f"CData validation failed (Pydantic):\n{e}")
        return
    except Exception as e:
        print(f"Unexpected error during initialization: {e}")
        return

    model = Small_LLM_Model(
        model_name="Qwen/Qwen3-0.6B",
        )
    system_prompt = build_system_prompt(functions)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    try:
        with OUTPUT_PATH.with_suffix('.jsonl').open("w",
                                                    encoding="utf-8") as f:
            for row in iter_prompt_results(model,
                                           system_prompt,
                                           prompts,
                                           functions_by_name):
                if row is None:
                    print("Skipping due to invalid input")
                    continue
                if row['name'].strip() == "None":
                    print("Skipping invalid prompt")
                    continue
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()

        jsonl_to_json_array(OUTPUT_PATH.with_suffix('.jsonl'), OUTPUT_PATH)
        end_time = time.perf_counter()
        final_time = end_time - start_time
        print(f"{int(final_time//60)} mins {int(final_time % 60)} seconds")
        print(f"Success! Results saved to {OUTPUT_PATH}")

    except Exception as e:
        print(f"An error occurred during generation: {e}")
