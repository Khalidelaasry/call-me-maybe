*This project has been created as part of the 42 curriculum by khelaasr.*

# call me maybe

## Description

A function-calling system that translates natural-language prompts into structured JSON function calls using constrained decoding on a small LLM (Qwen/Qwen3-0.6B).

Given "What is the sum of 2 and 3?", instead of answering "5", it produces:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": { "a": 2.0, "b": 3.0 }
}
```

## Instructions

```bash
# Install dependencies
uv sync

# Run with default paths
uv run python -m src

# Run with custom paths
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

> Make sure to copy the real `llm_sdk/` folder from your school into the project root before running.

## Algorithm Explanation

The program generates JSON one token at a time. At each step:

1. Get logits from the LLM for all ~32,000 vocabulary tokens
2. Set all invalid tokens to −∞ (constrained decoding)
3. Pick the highest remaining token

A state machine tracks where we are in the JSON structure and decides which tokens are valid at each position. For the function name, only tokens that are valid prefixes of a known function name are allowed. For number parameters, only digit tokens are allowed.

This guarantees 100% valid JSON output every time.

## Design Decisions

- All data models use **Pydantic** for validation
- The state machine is in its own file (`state_machine.py`) to keep concerns separate
- All errors are caught gracefully — the program never crashes unexpectedly

## Performance Analysis

- **JSON validity**: 100% — guaranteed by constrained decoding
- **Function accuracy**: 90%+ — depends on model and prompt quality
- **Speed**: ~5–15 seconds per prompt on CPU

## Challenges Faced

- The Qwen tokenizer uses `Ġ` to represent a leading space — had to strip it before token matching
- Multi-character tokens (e.g. `fn_add` as one token) required prefix matching at the token level, not character level
- Knowing when to stop generating a number value required checking if the next unconstrained token would be `,` or `}`

## Testing Strategy

```bash
uv run pytest tests/ -v
```

Tests cover: Pydantic model validation, file loading errors, prefix narrowing logic, and prompt building.

## Example Usage

```bash
uv run python -m src
cat data/output/function_calling_results.json
```

## Resources

- [3Blue1Brown — But what is a GPT?](https://www.youtube.com/watch?v=wjZofJX0v4M)
- [Pydantic v2 docs](https://docs.pydantic.dev)
- [uv docs](https://docs.astral.sh/uv/)
- [JSON specification](https://www.json.org/json-en.html)

**AI usage:** Claude (Anthropic) was used to help explain constrained decoding concepts, review the state machine design, and draft docstrings. All code was reviewed and understood before use.