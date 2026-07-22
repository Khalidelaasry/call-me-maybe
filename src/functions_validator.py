from typing import Any
from pydantic import BaseModel, Field


class ParameterModel(BaseModel):

    type: str


class FunctionDefinition(BaseModel):

    name: str = Field(min_length=1)
    description: str
    parameters: dict[str, ParameterModel]
    returns: ParameterModel


class FunctionCallingTest(BaseModel):

    prompt: str


class FunctionCallResult(BaseModel):

    prompt: str
    name: str
    parameters: dict[str, Any]
