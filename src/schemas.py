from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


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
    parameters: dict[str, ParameterDefinition]
    returns: ReturnDefinition


class PromptItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str


class FunctionCallAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    parameters: dict[str, Any]
