import sys
import json
import argparse
from pathlib import Path
from typing import Any, TypeVar
from pydantic import BaseModel, ValidationError
from src.functions_validator import FunctionDefinition, FunctionCallingTest

ModelT = TypeVar("ModelT", bound=BaseModel)

_DEFAULT_FUNCTIONS_FILE = Path("data/input/functions_definition.json")
_DEFAULT_TESTS_FILE = Path("data/input/function_calling_tests.json")
_DEFAULT_OUTPUT_FILE = Path("data/output/function_calling_results.json")


def parse_arguments_and_load_data() -> (
            tuple[Path, list[FunctionDefinition],
                  list[FunctionCallingTest]]):
    args = _parse_cli_arguments()
    _ensure_input_files_exist(args.functions_definition, args.input)

    functions_def = load_json_array(
        args.functions_definition, FunctionDefinition)
    calling_tests = load_json_array(args.input, FunctionCallingTest)

    _ensure_output_directory_exists(args.output)

    return args.output, functions_def, calling_tests


def _parse_cli_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--functions_definition", type=Path,
                        default=_DEFAULT_FUNCTIONS_FILE)
    parser.add_argument("--input", type=Path,
                        default=_DEFAULT_TESTS_FILE)
    parser.add_argument("--output", type=Path,
                        default=_DEFAULT_OUTPUT_FILE)
    return parser.parse_args()


def _ensure_input_files_exist(*paths: Path) -> None:
    for file_path in paths:
        if not file_path.exists():
            sys.exit(f"Critical Error: File '{file_path}' not found.")


def _ensure_output_directory_exists(output_path: Path) -> None:
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        sys.exit(f"Error: Could not create output dir. {output_path.parent}"
                 f"\nDetails: {exc}")


def load_json_array(
        file_path: Path, model_class: type[ModelT]) -> list[ModelT]:
    try:
        raw_items = _read_json_array(file_path)
        return [model_class.model_validate(item) for item in raw_items]
    except (FileNotFoundError, PermissionError):
        sys.exit(f"Error accessing file '{file_path}'.")
    except json.JSONDecodeError as exc:
        sys.exit(f"Error: '{file_path}' is not valid JSON. {exc.msg}")
    except ValidationError:
        sys.exit("Error: Data validation failed.")
    except Exception as exc:
        sys.exit(f"Unexpected error with '{file_path}': {exc}")


def _read_json_array(file_path: Path) -> list[Any]:
    with file_path.open('r') as raw_file:
        payload = json.load(raw_file)

    if not isinstance(payload, list):
        sys.exit(f"Error: File '{file_path}' must contain a JSON.")

    return payload
