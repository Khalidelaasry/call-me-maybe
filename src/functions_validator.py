"""Pydantic models and validation helpers for function-call data."""

from typing import Any
from pydantic import BaseModel, Field


class ParameterModel(BaseModel):
    """Describe one parameter type declared by an input function."""

    type: str


class FunctionDefinition(BaseModel):
    """Describe a callable function supplied in the definitions file."""

    name: str = Field(min_length=1)
    description: str
    parameters: dict[str, ParameterModel]
    returns: ParameterModel


class FunctionCallingTest(BaseModel):
    """Represent one natural-language prompt from the input file."""

    prompt: str


class FunctionCallResult(BaseModel):
    """Represent the required top-level shape of one output entry."""

    prompt: str
    name: str
    parameters: dict[str, Any]


def validate_function_call(
        result: FunctionCallResult,
        definition: FunctionDefinition) -> FunctionCallResult:
    """Ensure a generated result exactly matches its function definition."""
    if result.name != definition.name:
        raise ValueError("Generated function name does not match its schema.")

    expected_names = set(definition.parameters)
    actual_names = set(result.parameters)
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise ValueError(
            f"Invalid parameter keys; missing={missing}, extra={extra}.")

    for name, schema in definition.parameters.items():
        if not _has_declared_type(result.parameters[name], schema.type):
            raise ValueError(
                f"Parameter '{name}' must have type '{schema.type}'.")
    return result


def _has_declared_type(value: Any, declared_type: str) -> bool:
    """Return whether a JSON value has the requested schema type."""
    if declared_type == "string":
        return isinstance(value, str)
    if declared_type == "boolean":
        return isinstance(value, bool)
    if declared_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if declared_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if declared_type == "null":
        return value is None
    raise ValueError(f"Unsupported parameter type '{declared_type}'.")
