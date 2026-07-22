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
    StateParseNumber
)

_NUMERIC_PARAMETER_TYPES = ("number", "integer")


class GenerationJsonError(Exception):
    pass


class TwoStepJsonGenerator(BaseModel):

    model_config = ConfigDict(arbitrary_types_allowed=True)
    user_prompt: str
    functions_definition: list[FunctionDefinition]
    assistant: ConstrainedDecoder

    def generate(self) -> dict[str, Any]:
        function_name = self._select_function_name()
        function_schema = self._lookup_function(function_name)
        arguments = self._extract_arguments(function_schema)

        return {
            "prompt": self.user_prompt,
            "name": function_name,
            "parameters": arguments,
        }


    def _select_function_name(self) -> str:
        return self.assistant.generate(
            prompt=self._function_selection_prompt(),
            state=self._function_selection_state(),
            max_tokens=150,
        )

    def _function_selection_prompt(self) -> str:
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
        return StateBranch(choices={
            fn.name: StateTerminal() for fn in self.functions_definition
        })


    def _extract_arguments(
            self, function_schema: FunctionDefinition) -> dict[str, Any]:
        raw_json_text = self.assistant.generate(
            prompt=self._argument_extraction_prompt(function_schema),
            state=self._argument_extraction_state(function_schema),
            max_tokens=150,
        )

        arguments = self._parse_json_object(raw_json_text)
        return self._coerce_numeric_arguments(arguments, function_schema)

    def _argument_extraction_prompt(
            self, function_schema: FunctionDefinition) -> str:
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
        if param_schema.type in _NUMERIC_PARAMETER_TYPES:
            return StateParseNumber(next_state=next_state)
        return StateParseString(next_state=next_state)


    def _lookup_function(self, function_name: str) -> FunctionDefinition:
        for function_schema in self.functions_definition:
            if function_schema.name == function_name:
                return function_schema
        raise GenerationJsonError(f"Unknown function: {function_name}")

    @staticmethod
    def _parse_json_object(json_text: str) -> dict[str, Any]:
        if not json_text.strip():
            return {}

        try:
            parsed = json.loads(json_text)
        except json.JSONDecodeError:
            raise GenerationJsonError(f"Invalid JSON: {json_text}")

        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _coerce_numeric_arguments(
            arguments: dict[str, Any],
            function_schema: FunctionDefinition) -> dict[str, Any]:
        for param_name, param_schema in function_schema.parameters.items():
            if param_name not in arguments:
                continue

            raw_value = arguments[param_name]
            try:
                if param_schema.type == "number":
                    arguments[param_name] = float(raw_value)
                elif param_schema.type == "integer":
                    arguments[param_name] = int(float(raw_value))
            except (ValueError, TypeError):
                pass

        return arguments
