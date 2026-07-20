import json
import re
import sys
from typing import Any, Iterable
from pydantic import BaseModel, Field

# A token qualifies as "numeric" if it could legally appear somewhere in a
# JSON number literal, possibly followed by whitespace and/or one of the
# delimiters that can terminate a value (comma, closing brace/bracket).
_NUMERIC_TOKEN_PATTERN = re.compile(r'^[ \n\r\t]*-?[\d.eE+-]*[ \n\r\t,}\]]*$')
_QUOTE_CHAR = '"'


class StrictVocabFilter(BaseModel):
    """Pre-computed token groups used by constrained JSON parsing.

    Attributes:
        numeric_tokens: Tokens that can appear in JSON number contexts.
        string_content_tokens: Tokens safe inside open JSON strings.
        string_closer_tokens: Tokens containing quote characters.
        exact_quote_tokens: Tokens that are exactly one double-quote.
    """

    numeric_tokens: set[int] = Field(default_factory=set)
    string_content_tokens: set[int] = Field(default_factory=set)
    string_closer_tokens: set[int] = Field(default_factory=set)
    exact_quote_tokens: set[int] = Field(default_factory=set)

    @classmethod
    def from_clean_vocab(
            cls, clean_vocab: dict[int, str]) -> "StrictVocabFilter":
        """Build token filter sets from a decoded token vocabulary.

        Args:
            clean_vocab: Mapping from token IDs to decoded token strings.

        Returns:
            Filter object with token IDs grouped by usage.
        """
        buckets = cls()

        for token_id, token_str in clean_vocab.items():
            if not token_str:
                continue
            buckets._classify(token_id, token_str)

        return buckets

    def _classify(self, token_id: int, token_str: str) -> None:
        """Sort a single decoded token into every bucket it belongs to.

        Args:
            token_id: Vocabulary ID of the token being classified.
            token_str: Decoded text of the token being classified.
        """
        if _NUMERIC_TOKEN_PATTERN.match(token_str):
            self.numeric_tokens.add(token_id)

        if token_str == _QUOTE_CHAR:
            self.exact_quote_tokens.add(token_id)

        if _QUOTE_CHAR in token_str:
            self.string_closer_tokens.add(token_id)
        else:
            self.string_content_tokens.add(token_id)


class VocabIndex(BaseModel):
    """Store decoded vocabulary and derived lookup accelerators.

    Attributes:
        clean_vocab: Mapping of token ID to decoded text.
        filter_vocab: Pre-computed token groups for parser states.
        literal_cache: Cached literal-prefix token matches by remainder string.
    """

    clean_vocab: dict[int, str]
    filter_vocab: StrictVocabFilter
    literal_cache: dict[str, set[int]] = Field(default_factory=dict)

    @classmethod
    def from_model(cls, model: Any) -> "VocabIndex":
        """Load, decode, and index vocabulary from a model backend.

        Args:
            model: Model object exposing vocab path and decode capabilities.

        Returns:
            Fully initialized vocabulary index and filter sets.

        Raises:
            SystemExit: If vocabulary loading or processing fails.
        """
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
        """Load the raw token-string-to-ID mapping from disk.

        Args:
            vocab_path: Filesystem path to the vocabulary JSON file.

        Returns:
            dict[str, int]: Raw vocabulary as stored on disk.

        Raises:
            SystemExit: If the file cannot be opened or parsed.
        """
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
        """Decode every token ID in the raw vocabulary via the model.

        Args:
            model: Model object exposing a `decode` method.
            raw_vocabulary: Mapping of raw token strings to their IDs.

        Returns:
            dict[int, str]: Token ID to decoded text mapping.
        """
        return {
            token_id: model.decode([token_id])
            for token_id in raw_vocabulary.values()
        }

    def get_literal_matches(self, remainder: str) -> set[int]:
        """Get tokens that match the given remainder string.

        Args:
            remainder: Remaining literal text expected by the state machine.

        Returns:
            set[int]: Token IDs where either the token starts with the
            remainder or the remainder starts with the token. Results are
            cached for performance.
        """
        cached = self.literal_cache.get(remainder)
        if cached is None:
            cached = set(self._find_literal_matches(remainder))
            self.literal_cache[remainder] = cached
        return cached

    def _find_literal_matches(self, remainder: str) -> Iterable[int]:
        """Scan the vocabulary for tokens compatible with a remainder.

        Args:
            remainder: Remaining literal text expected by the state machine.

        Yields:
            int: Token IDs that either extend the remainder, or are
            themselves fully covered by it.
        """
        for token_id, token_str in self.clean_vocab.items():
            if remainder.startswith(token_str) or token_str.startswith(
                    remainder):
                yield token_id