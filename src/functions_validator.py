from typing import Any
from pydantic import BaseModel, Field


class ParameterModel(BaseModel):
    """Schema of a single function parameter (or return value).

    Attributes:
        type: JSON type name expected for the parameter value.
    """

    type: str


class FunctionDefinition(BaseModel):
    """A callable function exposed to the model, and its declared contract.

    Attributes:
        name: Function identifier expected in the generated output.
        description: Natural-language explanation of the function purpose.
        parameters: Mapping of parameter names to their schema definitions.
        returns: Declared return type schema for the function.
    """

    name: str = Field(min_length=1)
    description: str
    parameters: dict[str, ParameterModel]
    returns: ParameterModel


class FunctionCallingTest(BaseModel):
    """A single user prompt used in evaluation.

    Attributes:
        prompt: Raw user request that should map to a function call.
    """

    prompt: str


class FunctionCallResult(BaseModel):
    """A validated function call produced from one prompt.

    Attributes:
        prompt: Original user prompt.
        name: Selected function name.
        parameters: Extracted arguments prepared for function execution.
    """

    prompt: str
    name: str
    parameters: dict[str, Any]