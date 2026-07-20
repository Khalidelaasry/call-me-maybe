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
    """Raised when constrained generation cannot produce valid JSON output."""


class TwoStepJsonGenerator(BaseModel):
    """Generate function-calling JSON in two constrained decoding phases.

    Attributes:
        user_prompt: Natural-language user request to transform.
        functions_definition: Available callable function schemas.
        assistant: Constrained decoder used for token-by-token generation.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    user_prompt: str
    functions_definition: list[FunctionDefinition]
    assistant: ConstrainedDecoder

    def generate(self) -> dict[str, Any]:
        """Generate the final function call object for the user prompt.

        Returns:
            dict[str, Any]: Dictionary with prompt, name, and parameters
            fields.

        Raises:
            GenerationJsonError: If generated function selection or parameters
                cannot be mapped to a known, valid JSON output.
        """
        function_name = self._select_function_name()
        function_schema = self._lookup_function(function_name)
        arguments = self._extract_arguments(function_schema)

        return {
            "prompt": self.user_prompt,
            "name": function_name,
            "parameters": arguments,
        }

    # -- Phase 1: choosing which function to call -----------------------

    def _select_function_name(self) -> str:
        """Generate the function name that best matches the prompt.

        Returns:
            str: Selected function name constrained to known choices.
        """
        return self.assistant.generate(
            prompt=self._function_selection_prompt(),
            state=self._function_selection_state(),
            max_tokens=150,
        )

    def _function_selection_prompt(self) -> str:
        """Build the system prompt used to select one function name.

        Returns:
            str: Prompt that lists all candidate functions and user input.
        """
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
        """Create a branching state machine constrained to known names.

        Returns:
            State: Branch state where each key is a possible function name.
        """
        return StateBranch(choices={
            fn.name: StateTerminal() for fn in self.functions_definition
        })

    # -- Phase 2: extracting the arguments for that function -------------

    def _extract_arguments(
            self, function_schema: FunctionDefinition) -> dict[str, Any]:
        """Generate and normalize JSON parameters for a target function.

        Args:
            function_schema: Function schema whose parameters must be
                extracted.

        Returns:
            dict[str, Any]: Parsed parameters converted to expected numeric
            types where applicable.

        Raises:
            GenerationJsonError: If the generated parameter text is not valid
                JSON object content.
        """
        raw_json_text = self.assistant.generate(
            prompt=self._argument_extraction_prompt(function_schema),
            state=self._argument_extraction_state(function_schema),
            max_tokens=150,
        )

        arguments = self._parse_json_object(raw_json_text)
        return self._coerce_numeric_arguments(arguments, function_schema)

    def _argument_extraction_prompt(
            self, function_schema: FunctionDefinition) -> str:
        """Build a prompt that asks for literal parameter extraction only.

        Args:
            function_schema: Function schema describing required parameters.

        Returns:
            str: Instructional prompt for constrained parameter decoding.
        """
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
            "<|im_end|>\n",
            f"<|im_start|>user\n{self.user_prompt}<|im_end|>\n",
            "<|im_start|>assistant\n",
        ]
        return "".join(segments)

    def _argument_extraction_state(
            self, function_schema: FunctionDefinition) -> State:
        """Build a deterministic state chain for parameter JSON emission.

        Args:
            function_schema: Function schema used to derive parameter
                structure.

        Returns:
            State: Root state that enforces exact JSON layout and value
            types.
        """
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
        """Recursively assemble the literal/value chain for one parameter
        and everything that follows it.

        Args:
            param_items: Ordered (name, schema) pairs for every parameter.
            index: Position of the parameter currently being linked.
            tail_state: State to reach once the whole object has closed.

        Returns:
            State: The literal state expecting this parameter's key prefix.
        """
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
        """Pick the value-parsing state that matches a parameter's type.

        Args:
            param_schema: Schema describing the expected value type.
            next_state: State to continue with once the value is complete.

        Returns:
            State: A number or string parsing state, as appropriate.
        """
        if param_schema.type in _NUMERIC_PARAMETER_TYPES:
            return StateParseNumber(next_state=next_state)
        return StateParseString(next_state=next_state)

    # -- Shared helpers ----------------------------------------------------

    def _lookup_function(self, function_name: str) -> FunctionDefinition:
        """Find a function schema by exact name.

        Args:
            function_name: Generated function identifier to resolve.

        Returns:
            FunctionDefinition: Matching function schema.

        Raises:
            GenerationJsonError: If the name is not present in definitions.
        """
        for function_schema in self.functions_definition:
            if function_schema.name == function_name:
                return function_schema
        raise GenerationJsonError(f"Unknown function: {function_name}")

    @staticmethod
    def _parse_json_object(json_text: str) -> dict[str, Any]:
        """Parse generated parameter text into a JSON object.

        Args:
            json_text: Raw generated text expected to encode a JSON object.

        Returns:
            dict[str, Any]: Parsed JSON object dictionary, or an empty
            dictionary if the text is empty or the payload is not a
            dictionary.

        Raises:
            GenerationJsonError: If text is non-empty and not valid JSON.
        """
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
        """Convert numeric argument values to schema-expected Python types.

        Args:
            arguments: Parsed argument dictionary to normalize.
            function_schema: Function schema containing expected parameter
                types.

        Returns:
            dict[str, Any]: The same dictionary, with numeric-typed entries
            converted in place.
        """
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