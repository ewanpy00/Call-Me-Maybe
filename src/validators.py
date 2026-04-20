from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


class ParameterDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["string", "number", "integer", "boolean"]


class ReturnDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal[
        "string", "number", "integer", "boolean", "object", "array", "null"]


class FunctionDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: dict[str, ParameterDefinition] = Field(min_length=1)
    returns: ReturnDefinition


class PromptItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str


class FunctionCallAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    parameters: dict[str, Any]


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
