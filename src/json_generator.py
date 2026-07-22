"""Generate schema-constrained function calls with the supplied LLM."""

import json
from typing import Any
from pydantic import BaseModel, ConfigDict

from src.constrained_decoder import ConstrainedDecoder
from src.functions_validator import FunctionDefinition, ParameterModel
from src.state_machine import (
    State,
    StateTerminal,
    StateExpectLiteral,
    StateBranch,
    StateParseString,
    StateParseNumber,
)

_NUMERIC_PARAMETER_TYPES = ("number", "integer")


class GenerationJsonError(Exception):
    """Report JSON generation or schema-construction failures."""

    pass


class TwoStepJsonGenerator(BaseModel):
    """Select a function, then generate exactly its JSON arguments."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    user_prompt: str
    functions_definition: list[FunctionDefinition]
    assistant: ConstrainedDecoder

    def generate(self) -> dict[str, Any]:
        """Generate one complete output object for the user prompt."""
        function_name = self._select_function_name()
        function_schema = self._lookup_function(function_name)
        arguments = self._extract_arguments(function_schema)

        return {
            "prompt": self.user_prompt,
            "name": function_name,
            "parameters": arguments,
        }

    def _select_function_name(self) -> str:
        """Generate a function name restricted to the available choices."""
        return self.assistant.generate(
            prompt=self._function_selection_prompt(),
            state=self._function_selection_state(),
            max_tokens=150,
        )

    def _function_selection_prompt(self) -> str:
        """Build the LLM prompt used to choose the function."""
        header = "<|im_start|>system\nChoose the exact function name.\n" \
                 "Functions:\n"
        listing = "".join(
            f"- {fn.name}: {fn.description}\n"
            for fn in self.functions_definition
        )
        footer = (
            f"<|im_end|>\n<|im_start|>user\n{self.user_prompt}"
            "<|im_end|>\n<|im_start|>assistant\n"
        )
        return header + listing + footer

    def _function_selection_state(self) -> State:
        """Create a state that permits only declared function names."""
        return StateBranch(choices={
            fn.name: StateTerminal() for fn in self.functions_definition
        })

    def _extract_arguments(
            self, function_schema: FunctionDefinition) -> dict[str, Any]:
        """Generate, parse, and normalize the selected function arguments."""
        raw_json_text = self.assistant.generate(
            prompt=self._argument_extraction_prompt(function_schema),
            state=self._argument_extraction_state(function_schema),
            max_tokens=150,
        )

        arguments = self._parse_json_object(raw_json_text)
        return self._coerce_numeric_arguments(arguments, function_schema)

    def _argument_extraction_prompt(
            self, function_schema: FunctionDefinition) -> str:
        """Build the extraction prompt for one known function schema."""
        params_info = ", ".join(
            f"'{param_name}' ({param_schema.type})"
            for param_name, param_schema in function_schema.parameters.items()
        )

        segments = [
            "<|im_start|>system\n",
            "Extract the specific parameters for the function",
            f" '{function_schema.name}'.\n",
            f"You must find these parameters: {params_info}\n",
            "CRITICAL: Do NOT execute the command.",
            "Do NOT calculate or reverse anything.",
            "ONLY extract the exact literal values from the text.\n",
            "For string parameters, preserve the EXACT case from the input.\n",
            "For an explicitly quoted empty string, emit an empty JSON "
            "string (\"\"), never a placeholder.\n",
            "For numbers, preserve a leading '-' and every decimal digit. "
            "A leading '+' means a positive JSON number (without '+'). "
            "Keep values in their original parameter order.\n",
            "<|im_end|>\n",
            f"<|im_start|>user\n{self.user_prompt}<|im_end|>\n",
            "<|im_start|>assistant\n",
        ]
        return "".join(segments)

    def _argument_extraction_state(
            self, function_schema: FunctionDefinition) -> State:
        """Create the state machine for a JSON arguments object."""
        param_items = list(function_schema.parameters.items())

        if not param_items:
            return StateExpectLiteral(
                expected='{}', next_state=StateTerminal())

        closing_brace = StateExpectLiteral(
            expected='\n}', next_state=StateTerminal())

        return self._build_argument_chain(param_items, 0, closing_brace)

    def _build_argument_chain(
            self,
            param_items: list[tuple[str, ParameterModel]],
            index: int,
            tail_state: State) -> State:
        """Recursively create states for parameters in definition order."""
        param_name, param_schema = param_items[index]

        is_last = index + 1 == len(param_items)
        if is_last:
            next_literal = tail_state
        else:
            next_literal = self._build_argument_chain(
                param_items, index + 1, tail_state)

        value_state = self._value_state_for(param_schema, next_literal)
        prefix = (
            f'{{\n  "{param_name}": ' if index == 0
            else f',\n  "{param_name}": '
        )
        return StateExpectLiteral(expected=prefix, next_state=value_state)

    @staticmethod
    def _value_state_for(
            param_schema: ParameterModel, next_state: State) -> State:
        """Return the constrained value state for one declared JSON type."""
        if param_schema.type in _NUMERIC_PARAMETER_TYPES:
            return StateParseNumber(next_state=next_state)
        if param_schema.type == "string":
            return StateParseString(next_state=next_state)
        if param_schema.type == "boolean":
            return StateBranch(choices={
                "true": next_state,
                "false": next_state,
            })
        if param_schema.type == "null":
            return StateBranch(choices={"null": next_state})
        raise GenerationJsonError(
            f"Unsupported parameter type '{param_schema.type}'.")

    def _lookup_function(self, function_name: str) -> FunctionDefinition:
        """Find the schema for a model-selected function name."""
        for function_schema in self.functions_definition:
            if function_schema.name == function_name:
                return function_schema
        raise GenerationJsonError(f"Unknown function: {function_name}")

    @staticmethod
    def _parse_json_object(json_text: str) -> dict[str, Any]:
        """Parse generated text and require a JSON object."""
        if not json_text.strip():
            return {}

        try:
            parsed = json.loads(json_text)
        except json.JSONDecodeError as exc:
            raise GenerationJsonError(f"Invalid JSON: {json_text}") from exc

        if not isinstance(parsed, dict):
            raise GenerationJsonError(
                "Generated arguments are not a JSON object.")
        return parsed

    @staticmethod
    def _coerce_numeric_arguments(
            arguments: dict[str, Any],
            function_schema: FunctionDefinition) -> dict[str, Any]:
        """Convert JSON number values to the declared Python number types."""
        for param_name, param_schema in function_schema.parameters.items():
            if param_name not in arguments:
                continue

            raw_value = arguments[param_name]
            try:
                if param_schema.type == "number":
                    arguments[param_name] = float(raw_value)
                elif param_schema.type == "integer":
                    arguments[param_name] = int(float(raw_value))
            except (ValueError, TypeError) as exc:
                message = f"Parameter '{param_name}' is not a valid number."
                raise GenerationJsonError(message) from exc

        return arguments
