import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from llm_sdk.llm_sdk import Small_LLM_Model
from pydantic import ValidationError

from src.schemas import FunctionCallAnswer, FunctionDefinition, PromptItem

FUNCTIONS_PATH = "data/input/functions_definition.json"
PROMPTS_PATH = "data/input/function_calling_tests.json"
OUTPUT_JSON_PATH = "data/output/results.json"
OUTPUT_JSONL_PATH = "data/output/results.jsonl"
MAX_NEW_TOKENS = 256
MAX_RETRIES = 3
MIN_FREE_DISK_BYTES_FOR_MPS = 2 * 1024 * 1024 * 1024
RAW_ANSWER_SNIP_LEN = 2000


def load_json(file_path: str) -> Any:
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def extract_first_json_object(text: str) -> str | None:
    start_idx = -1
    depth = 0
    in_string = False
    escaped = False

    for idx, char in enumerate(text):
        if start_idx == -1:
            if char == "{":
                start_idx = idx
                depth = 1
            continue

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start_idx: idx + 1]

    return None


def build_system_prompt(functions: list[FunctionDefinition]) -> str:
    return f"""You are a function calling assistant.
Return one strict JSON object only, no markdown and no extra text.
The JSON must have this shape:
{{"name":"<function_name>", "parameters":{{...}}}}

Available functions:
{json.dumps([f.model_dump() for f in functions], indent=2)}
"""


def choose_device() -> str | None:
    free_bytes = shutil.disk_usage("/").free
    if free_bytes < MIN_FREE_DISK_BYTES_FOR_MPS:
        print(
            f"Low disk space ({free_bytes / (1024 ** 3):.2f} GB free). "
            "Using CPU to avoid MPS graph cache failures."
        )
        return "cpu"
    return None


def generate_answer(
    model: Small_LLM_Model,
    system_prompt: str,
    user_query: str,
) -> str:
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


def validate_answer(
    answer_obj: dict[str, Any],
    functions_by_name: dict[str, FunctionDefinition],
) -> FunctionCallAnswer:
    answer = FunctionCallAnswer.model_validate(answer_obj)
    function = functions_by_name.get(answer.name)
    if function is None:
        raise ValueError(f"Unknown function: {answer.name}")

    expected_parameters = function.parameters
    provided_keys = set(answer.parameters.keys())
    expected_keys = set(expected_parameters.keys())
    if provided_keys != expected_keys:
        raise ValueError(
            f"Invalid parameter keys for {answer.name}:"
            " expected {sorted(expected_keys)}, got {sorted(provided_keys)}"
        )

    for param_name, param_def in expected_parameters.items():
        value = answer.parameters[param_name]
        p_type = param_def.type

        if p_type == "string" and not isinstance(value, str):
            raise ValueError(f"Parameter '{param_name}' must be string")
        if p_type == "number" and not isinstance(value, (int, float)):
            raise ValueError(f"Parameter '{param_name}' must be number")
        if p_type == "integer" and not isinstance(value, int):
            raise ValueError(f"Parameter '{param_name}' must be integer")
        if p_type == "boolean" and not isinstance(value, bool):
            raise ValueError(f"Parameter '{param_name}' must be boolean")

    return answer


def truncate_for_log(text: str, limit: int = RAW_ANSWER_SNIP_LEN) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit]}… (truncated, {len(text)} chars)"


def iter_prompt_results(
    model: Small_LLM_Model,
    system_prompt: str,
    prompts: list[PromptItem],
    functions_by_name: dict[str, FunctionDefinition],
) -> Iterator[dict[str, Any]]:
    for prompt in prompts:
        user_query = prompt.prompt
        print(f"Processing Query: {user_query}")

        validated_answer: FunctionCallAnswer | None = None
        raw_answer = ""
        for attempt in range(MAX_RETRIES):
            raw_answer = generate_answer(model, system_prompt, user_query)
            try:
                parsed = parse_function_call_dict(raw_answer)
                validated_answer = validate_answer(parsed, functions_by_name)
                break
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                print(
                    f"Retry {attempt + 1}/{MAX_RETRIES}"
                    f" because answer is invalid: {exc!s}. "
                    f"raw_snip={truncate_for_log(raw_answer)!r}"
                )

        if validated_answer is None:
            answer_payload: Any = {
                "error": "invalid_model_output",
                "raw_answer_snip": truncate_for_log(raw_answer),
            }
        else:
            answer_payload = validated_answer.model_dump()

        yield {"prompt": user_query, "answer": answer_payload}
        print("Done.")
        print("-" * 20)


def jsonl_to_json_array(jsonl_path: Path, json_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(rows, file, indent=4, ensure_ascii=False)


def main() -> None:
    prompts_raw = load_json(PROMPTS_PATH)
    functions_raw = load_json(FUNCTIONS_PATH)

    prompts = [PromptItem.model_validate(prompt) for prompt in prompts_raw]
    functions = [FunctionDefinition.model_validate(f) for f in functions_raw]
    functions_by_name = {function.name: function for function in functions}

    model = Small_LLM_Model(
        model_name="Qwen/Qwen3-0.6B", device=choose_device())
    system_prompt = build_system_prompt(functions)

    output_jsonl = Path(OUTPUT_JSONL_PATH)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with output_jsonl.open("w", encoding="utf-8") as jsonl_file:
        for row in iter_prompt_results(model,
                                       system_prompt,
                                       prompts,
                                       functions_by_name):
            jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            jsonl_file.flush()

    jsonl_to_json_array(output_jsonl, Path(OUTPUT_JSON_PATH))
    print(f"Saved line-by-line results to {output_jsonl}")
    print(f"Wrote combined JSON array to {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    main()
