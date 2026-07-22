"""Command-line entry point for the constrained function-calling tool."""

import sys
import json
import time
from pathlib import Path
from typing import Any

from llm_sdk import Small_LLM_Model
from src.data_loader import parse_arguments_and_load_data
from src.functions_validator import (
    FunctionCallResult,
    FunctionDefinition,
    FunctionCallingTest,
    validate_function_call,
)
from src.vocabulary import VocabIndex
from src.constrained_decoder import ConstrainedDecoder
from src.json_generator import TwoStepJsonGenerator, GenerationJsonError


def init_ai() -> ConstrainedDecoder:
    """Load the LLM and prepare its vocabulary index."""
    print("Initializing the LLM model and vocabulary...")
    try:
        model = Small_LLM_Model()
        print("Qwen/Qwen3-0.6B model loaded successfully.")
        vocab_index = VocabIndex.from_model(model)
        return ConstrainedDecoder(llm=model, vocab_index=vocab_index)
    except Exception as exc:
        sys.exit(f"CRITICAL ERROR during initialization: {exc}")


def process_all_prompts(
        calling_tests: list[FunctionCallingTest],
        functions_def: list[FunctionDefinition],
        assistant: ConstrainedDecoder) -> list[dict[str, Any]]:
    """Generate one validated result for every supplied prompt."""
    results: list[dict[str, Any]] = []

    for test_case in calling_tests:
        outcome = _run_one_prompt(test_case, functions_def, assistant)
        results.append(outcome)

    return results


def _run_one_prompt(
        test_case: FunctionCallingTest,
        functions_def: list[FunctionDefinition],
        assistant: ConstrainedDecoder) -> dict[str, Any]:
    """Generate and validate a result without silently dropping failures."""
    print(f"Processing: '{test_case.prompt}'...")

    try:
        generator = TwoStepJsonGenerator(
            user_prompt=test_case.prompt,
            functions_definition=functions_def,
            assistant=assistant
        )
        generated = generator.generate()
        selected = next(
            fn for fn in functions_def if fn.name == generated["name"])
        validated = validate_function_call(
            FunctionCallResult.model_validate(generated), selected)

        print(f"  ✓ Success: {generated.get('name')} "
              f"{generated.get('parameters')}")
        return validated.model_dump()

    except (ValueError, GenerationJsonError) as exc:
        raise GenerationJsonError(f"Generation failed: {exc}") from exc


def save_results(results: list[dict[str, Any]], output_path: Path) -> None:
    """Write all validated results as one JSON array, including empties."""

    try:
        with output_path.open('w') as output_file:
            json.dump(results, output_file, indent=2, ensure_ascii=False)
        print(f"\n✓ All results successfully saved to {output_path}")
    except OSError as exc:
        sys.exit(f"  ✗ Error saving the JSON file: {exc}")


def main() -> None:
    """Run the command-line application and report user-friendly failures."""
    try:
        started_at = time.time()

        output_path, functions_def, tests = parse_arguments_and_load_data()
        assistant = init_ai()
        results = process_all_prompts(tests, functions_def, assistant)
        save_results(results, output_path)

        elapsed = time.time() - started_at
        print(f"\nTotal execution time: {elapsed:.2f} seconds.")
    except KeyboardInterrupt:
        print("\nInterrupted. Goodbye.")
    except (GenerationJsonError, ValueError) as exc:
        sys.exit(f"ERROR: {exc}")


if __name__ == "__main__":
    main()
