*This project has been created as part of the 42 curriculum by khelaasr.*

# Call Me Maybe

## Description

Call Me Maybe translates natural-language requests into JSON function calls. It uses
Qwen/Qwen3-0.6B through the supplied `llm_sdk`, selecting a declared function and
extracting its arguments without executing that function. The output always has the
required `prompt`, `name`, and `parameters` keys.

## Instructions

Install the dependencies and run the default demonstration:

```sh
make install
make run
```

Use alternative input and output paths with `ARGS`:

```sh
make run ARGS="--functions_definition data/input/functions_definition.json \
--input data/input/function_calling_tests.json \
--output /tmp/function_calling_results.json"
```

The equivalent required command is:

```sh
uv run python -m src [--functions_definition FILE] [--input FILE] [--output FILE]
```

Useful development commands are `make test`, `make lint`, `make lint-strict`,
`make debug`, and `make clean`. `uv sync` is sufficient for a reviewer to set up the
project.

## Constrained decoding algorithm

The generator works in two LLM passes. The first pass constrains each generated token
to prefixes of the available function names. The selected name is therefore always one
of the input definitions. The second pass constructs the argument JSON object in the
definition's exact parameter order.

During the second pass, fixed JSON punctuation and parameter names are written
directly. For values, a finite-state machine filters the vocabulary before selecting the
highest-logit token. `StateParseString` permits only valid JSON escaping and a valid
closing quote; `StateParseNumber` permits only valid JSON numbers; fixed branches emit
only `true`/`false` or `null` for those types. Invalid logits are effectively excluded
because only state-approved token IDs are considered. Finally, Python parses the JSON
and validates the exact parameter-key set and declared types.

## Design decisions

- Pydantic validates input files, internal data models, and output shape.
- The supplied SDK is accessed only through its public methods.
- Errors from missing files, malformed JSON, invalid schemas, model setup, and failed
  generation are reported clearly. A failed prompt stops the program instead of quietly
  omitting an output entry.
- Generated output is written only after every prompt has produced a valid result.

## Performance and reliability

The decoder performs greedy selection over only valid tokens, so every completed
argument object is parseable JSON and conforms to its declared supported schema. The
two-pass approach keeps the selection prompt small and avoids unrestricted prose.
Runtime depends on model loading and prompt count; the included demonstration is
intended to complete well within the five-minute project target on standard hardware.

## Testing strategy

`make test` runs unit tests for exact parameter keys, string/integer/boolean handling,
and Python's boolean-versus-integer edge case. Manual tests should also cover missing
or malformed input files, empty strings, escaped special characters, negative and
decimal numbers, zero-argument functions, and custom function definitions. `make lint`
runs the required flake8 and mypy checks.

## Challenges faced

Tokenizer tokens can contain multiple characters, partial JSON, or punctuation. The
state machine therefore validates the complete candidate token against the text already
generated, and passes any token overflow into the following state. This makes the
decoder robust even when one token includes a value terminator and the next JSON
literal.

## Resources and AI use

- [JSON specification](https://www.rfc-editor.org/rfc/rfc8259)
- [Pydantic documentation](https://docs.pydantic.dev/)
- [Python typing documentation](https://docs.python.org/3/library/typing.html)
- Project-provided `llm_sdk` documentation and source code

AI was used as a review and pair-programming aid: to inspect the subject requirements,
identify missing validation and documentation, and propose tests and refactors. The
implementation was reviewed, adapted, and tested by the project author.
