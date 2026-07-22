import json
import re
import sys
from typing import Any, Iterable
from pydantic import BaseModel, Field

_NUMERIC_TOKEN_PATTERN = re.compile(r'^[ \n\r\t]*-?[\d.eE+-]*[ \n\r\t,}\]]*$')
_QUOTE_CHAR = '"'


class StrictVocabFilter(BaseModel):

    numeric_tokens: set[int] = Field(default_factory=set)
    string_content_tokens: set[int] = Field(default_factory=set)
    string_closer_tokens: set[int] = Field(default_factory=set)
    exact_quote_tokens: set[int] = Field(default_factory=set)

    @classmethod
    def from_clean_vocab(
            cls, clean_vocab: dict[int, str]) -> "StrictVocabFilter":
        buckets = cls()

        for token_id, token_str in clean_vocab.items():
            if not token_str:
                continue
            buckets._classify(token_id, token_str)

        return buckets

    def _classify(self, token_id: int, token_str: str) -> None:
        if _NUMERIC_TOKEN_PATTERN.match(token_str):
            self.numeric_tokens.add(token_id)

        if token_str == _QUOTE_CHAR:
            self.exact_quote_tokens.add(token_id)

        if _QUOTE_CHAR in token_str:
            self.string_closer_tokens.add(token_id)
        else:
            self.string_content_tokens.add(token_id)


class VocabIndex(BaseModel):

    clean_vocab: dict[int, str]
    filter_vocab: StrictVocabFilter
    literal_cache: dict[str, set[int]] = Field(default_factory=dict)

    @classmethod
    def from_model(cls, model: Any) -> "VocabIndex":
        vocab_path = model.get_path_to_vocab_file()
        raw_vocabulary = cls._read_raw_vocab(vocab_path)

        try:
            clean_vocab = cls._decode_all_tokens(model, raw_vocabulary)
            return cls(
                clean_vocab=clean_vocab,
                filter_vocab=StrictVocabFilter.from_clean_vocab(clean_vocab),
            )
        except Exception as exc:
            sys.exit(f"Error: Failed to process vocabulary data: {exc}")

    @staticmethod
    def _read_raw_vocab(vocab_path: str) -> dict[str, int]:
        try:
            with open(vocab_path, 'r', encoding='utf-8') as vocab_file:
                raw_vocabulary: dict[str, int] = json.load(vocab_file)
            return raw_vocabulary
        except (FileNotFoundError, PermissionError) as exc:
            sys.exit(f"Error accessing vocabulary file '{vocab_path}': {exc}")
        except Exception as exc:
            sys.exit(
                f"Unexpected error loading vocabulary file "
                f"'{vocab_path}': {exc}")

    @staticmethod
    def _decode_all_tokens(
            model: Any, raw_vocabulary: dict[str, int]) -> dict[int, str]:
        return {
            token_id: model.decode([token_id])
            for token_id in raw_vocabulary.values()
        }

    def get_literal_matches(self, remainder: str) -> set[int]:
        cached = self.literal_cache.get(remainder)
        if cached is None:
            cached = set(self._find_literal_matches(remainder))
            self.literal_cache[remainder] = cached
        return cached

    def _find_literal_matches(self, remainder: str) -> Iterable[int]:
        for token_id, token_str in self.clean_vocab.items():
            if remainder.startswith(token_str) or token_str.startswith(
                    remainder):
                yield token_id
