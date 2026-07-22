"""Optional interactive front end for manually testing function calls."""

import argparse
import json
import sys
from pathlib import Path

from src.__main__ import init_ai
from src.constrained_decoder import ConstrainedDecoder
from src.data_loader import load_json_array
from src.functions_validator import FunctionCallResult, FunctionDefinition
from src.json_generator import GenerationJsonError, TwoStepJsonGenerator

_EXIT_WORDS = {"exit", "quit"}


def _parse_args() -> argparse.Namespace:
    """Parse the optional function-definition path."""
    parser = argparse.ArgumentParser(
        description="Interactively test function-calling prompt by prompt."
    )
    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=Path("data/input/functions_definition.json"),
        help="Path to the JSON file describing the callable functions.",
    )
    return parser.parse_args()


def _load_functions(path: Path) -> list[FunctionDefinition]:
    """Load function definitions or exit with a clear path error."""
    if not path.exists():
        sys.exit(f"Critical Error: File '{path}' not found.")
    return load_json_array(path, FunctionDefinition)


def _generate_for_prompt(
        user_prompt: str,
        functions_def: list[FunctionDefinition],
        assistant: ConstrainedDecoder) -> None:
    """Generate, validate, and print one interactive function call."""
    try:
        generator = TwoStepJsonGenerator(
            user_prompt=user_prompt,
            functions_definition=functions_def,
            assistant=assistant,
        )
        generated = generator.generate()
        validated = FunctionCallResult.model_validate(generated)
        print(json.dumps(validated.model_dump(), indent=2,
                         ensure_ascii=False))
    except (ValueError, GenerationJsonError) as exc:
        print(f"  ✗ Generation error: {exc}")
    except Exception as exc:
        print(f"  ✗ Unexpected error: {exc}")


def _run_repl(
        functions_def: list[FunctionDefinition],
        assistant: ConstrainedDecoder) -> None:
    """Read prompts until the user exits or closes standard input."""
    print("Type a prompt and press Enter.")
    print("Type 'exit', 'quit', or press Ctrl+D to stop.\n")

    while True:
        try:
            user_prompt = input("prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return

        if not user_prompt:
            continue
        if user_prompt.lower() in _EXIT_WORDS:
            print("Goodbye.")
            return

        _generate_for_prompt(user_prompt, functions_def, assistant)
        print()


def main() -> None:
    """Start the interactive command-line interface."""
    try:
        args = _parse_args()
        functions_def = _load_functions(args.functions_definition)
        assistant = init_ai()
        _run_repl(functions_def, assistant)
    except KeyboardInterrupt:
        print("\nInterrupted. Goodbye.")


if __name__ == "__main__":
    main()
