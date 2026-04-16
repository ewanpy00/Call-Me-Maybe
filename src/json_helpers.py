from pathlib import Path
from typing import Union, Any
import json


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


def load_json(file_path: Union[str, Path]) -> Any:
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)
