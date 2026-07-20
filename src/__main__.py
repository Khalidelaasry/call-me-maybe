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
    FunctionCallingTest)
from src.vocabulary import VocabIndex
from src.constrained_decoder import ConstrainedDecoder
from src.json_generator import TwoStepJsonGenerator, GenerationJsonError


def init_ai() -> ConstrainedDecoder:
    """Initialize the language model and build its vocabulary index.

    Returns:
        ConstrainedDecoder: Ready-to-use constrained decoding assistant.

    Raises:
        SystemExit: If model or vocabulary initialization fails.
    """
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
    """Generate validated function-call outputs for all provided prompts.

    Args:
        calling_tests: Prompt test cases to process.
        functions_def: Available function schema definitions.
        assistant: Constrained decoder used for JSON generation.

    Returns:
        list[dict[str, Any]]: Successfully validated generation results.
    """
    results: list[dict[str, Any]] = []

    for test_case in calling_tests:
        outcome = _run_one_prompt(test_case, functions_def, assistant)
        if outcome is not None:
            results.append(outcome)

    return results


def _run_one_prompt(
        test_case: FunctionCallingTest,
        functions_def: list[FunctionDefinition],
        assistant: ConstrainedDecoder) -> dict[str, Any] | None:
    """Run generation for a single prompt and report the outcome.

    Args:
        test_case: Prompt test case to process.
        functions_def: Available function schema definitions.
        assistant: Constrained decoder used for JSON generation.

    Returns:
        dict[str, Any] | None: The validated result payload, or None if
        generation failed for this prompt.
    """
    print(f"Processing: '{test_case.prompt}'...")

    try:
        generator = TwoStepJsonGenerator(
            user_prompt=test_case.prompt,
            functions_definition=functions_def,
            assistant=assistant
        )
        generated = generator.generate()
        validated = FunctionCallResult.model_validate(generated)

        print(f"  ✓ Success: {generated.get('name')} "
              f"{generated.get('parameters')}")
        return validated.model_dump()

    except (ValueError, GenerationJsonError) as exc:
        print(f"  ✗ Generation error: {exc}")
    except Exception as exc:
        print(f"  ✗ Unexpected error: {exc}")

    return None


def save_results(results: list[dict[str, Any]], output_path: Path) -> None:
    """Persist generated results to disk as formatted JSON.

    Args:
        results: Validated result payloads to serialize.
        output_path: Destination file where output JSON is written.

    Raises:
        SystemExit: If writing the output file fails.
    """
    if not results:
        print("\nNo results generated. File not saved.")
        return

    try:
        with output_path.open('w') as output_file:
            json.dump(results, output_file, indent=2, ensure_ascii=False)
        print(f"\n✓ All results successfully saved to {output_path}")
    except OSError as exc:
        sys.exit(f"  ✗ Error saving the JSON file: {exc}")


def main() -> None:
    started_at = time.time()

    output_path, functions_def, tests = parse_arguments_and_load_data()
    assistant = init_ai()
    results = process_all_prompts(tests, functions_def, assistant)
    save_results(results, output_path)

    elapsed = time.time() - started_at
    print(f"\nTotal execution time: {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()