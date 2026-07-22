"""Unit tests for strict generated-call schema validation."""

import unittest

from src.functions_validator import (
    FunctionCallResult,
    FunctionDefinition,
    validate_function_call,
)


class FunctionCallValidationTests(unittest.TestCase):
    """Check required keys and JSON parameter types."""

    def setUp(self) -> None:
        """Create a representative schema for each test."""
        self.definition = FunctionDefinition.model_validate({
            "name": "fn_configure",
            "description": "Configure a feature.",
            "parameters": {
                "name": {"type": "string"},
                "retries": {"type": "integer"},
                "enabled": {"type": "boolean"},
            },
            "returns": {"type": "null"},
        })

    def test_accepts_exact_schema(self) -> None:
        """Accept a result containing every expected key and type."""
        result = FunctionCallResult(
            prompt="Configure alpha",
            name="fn_configure",
            parameters={"name": "alpha", "retries": 2, "enabled": True},
        )
        validated = validate_function_call(result, self.definition)
        self.assertEqual(validated, result)

    def test_rejects_missing_or_extra_parameter(self) -> None:
        """Reject schemas whose parameter key set is not exact."""
        result = FunctionCallResult(
            prompt="Configure alpha",
            name="fn_configure",
            parameters={"name": "alpha", "unexpected": 2},
        )
        with self.assertRaises(ValueError):
            validate_function_call(result, self.definition)

    def test_rejects_boolean_as_integer(self) -> None:
        """Reject Python booleans where an integer is required."""
        result = FunctionCallResult(
            prompt="Configure alpha",
            name="fn_configure",
            parameters={"name": "alpha", "retries": True, "enabled": True},
        )
        with self.assertRaises(ValueError):
            validate_function_call(result, self.definition)
