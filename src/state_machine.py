"""Finite-state JSON parser used to constrain each generated token."""

import re
from abc import ABC, abstractmethod
from pydantic import BaseModel, ConfigDict, Field
from src.vocabulary import VocabIndex

_JSON_WS = r'[ \n\r\t]*'

_PARTIAL_NUMBER_RE = re.compile(
    fr'^{_JSON_WS}-?(?:0|[1-9]\d*)?(?:\.\d*)?(?:[eE][+-]?\d*)?$')

_COMPLETE_NUMBER_PREFIX_RE = re.compile(
    fr'^{_JSON_WS}-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?')

_NUMBER_DELIMITERS = ',}]\n'


def is_partial_json_string_content(candidate: str) -> bool:
    """Return whether text can still become valid JSON string content."""
    index = 0
    while index < len(candidate):
        character = candidate[index]
        if character == '"' or ord(character) < 0x20:
            return False
        if character != '\\':
            index += 1
            continue

        index += 1
        if index == len(candidate):
            return True

        escape = candidate[index]
        if escape in '"\\/bfnrt':
            index += 1
            continue
        if escape != 'u':
            return False

        hex_start = index + 1
        hex_end = min(hex_start + 4, len(candidate))
        if any(char not in '0123456789abcdefABCDEF'
               for char in candidate[hex_start:hex_end]):
            return False
        if len(candidate) - hex_start < 4:
            return True
        index = hex_start + 4

    return True


def is_complete_json_string_content(candidate: str) -> bool:
    """Return whether text is valid, complete JSON string content."""
    index = 0
    while index < len(candidate):
        character = candidate[index]
        if character == '"' or ord(character) < 0x20:
            return False
        if character != '\\':
            index += 1
            continue

        index += 1
        if index == len(candidate):
            return False

        escape = candidate[index]
        if escape in '"\\/bfnrt':
            index += 1
            continue
        if escape != 'u' or index + 4 >= len(candidate):
            return False
        if any(char not in '0123456789abcdefABCDEF'
               for char in candidate[index + 1:index + 5]):
            return False
        index += 5

    return True


def is_partial_json_number(candidate: str) -> bool:
    """Return whether text can still become a valid JSON number."""
    return _PARTIAL_NUMBER_RE.fullmatch(candidate) is not None


def split_complete_json_number(candidate: str) -> tuple[str, str]:
    """Split a complete number prefix from any following token text."""
    match = _COMPLETE_NUMBER_PREFIX_RE.match(candidate)
    if match is None:
        return "", candidate
    number_text = match.group()
    return number_text, candidate[len(number_text):]


class State(BaseModel, ABC):
    """Base interface for states that accept constrained token text."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    buffer: str = Field(default="")

    @abstractmethod
    def get_valid_tokens(self, vocab_index: VocabIndex) -> set[int]:
        """Return IDs that can be emitted without invalidating this state."""
        raise NotImplementedError

    @abstractmethod
    def transition(self, token_str: str) -> tuple["State", str]:
        """Consume token text and return the next state plus overflow text."""
        raise NotImplementedError


class StateTerminal(State):
    """Represent successful completion of the constrained output."""

    def get_valid_tokens(self, vocab_index: VocabIndex) -> set[int]:
        """Return no candidates because generation is complete."""
        return set()

    def transition(self, token_str: str) -> tuple["State", str]:
        """Ignore all input after terminal completion."""
        return self, ""


class StateExpectLiteral(State):
    """Represent a JSON literal written directly by the decoder."""

    expected: str = Field(...)
    next_state: State | None = Field(default=None)

    def get_valid_tokens(self, vocab_index: VocabIndex) -> set[int]:
        """Return no candidates because literals are flushed directly."""
        return set()

    def transition(self, token_str: str) -> tuple["State", str]:
        """Advance after the expected literal has been fully consumed."""
        self.buffer += token_str

        if not self.buffer.startswith(self.expected):
            return self, ""

        overflow = self.buffer[len(self.expected):]
        return self.next_state or StateTerminal(), overflow


class StateBranch(State):
    """Constrain output to one of several fixed literal choices."""

    choices: dict[str, State] = Field(...)

    def get_valid_tokens(self, vocab_index: VocabIndex) -> set[int]:
        """Return IDs that can extend at least one possible branch."""
        valid_ids: set[int] = set()

        for candidate in self.choices:
            if not candidate.startswith(self.buffer):
                continue
            still_needed = candidate[len(self.buffer):]
            if still_needed:
                valid_ids |= vocab_index.get_literal_matches(still_needed)

        return valid_ids

    def transition(self, token_str: str) -> tuple["State", str]:
        """Advance when a branch has been completed by emitted text."""
        self.buffer += token_str

        for candidate, target_state in self.choices.items():
            if self.buffer.startswith(candidate):
                overflow = self.buffer[len(candidate):]
                return target_state, overflow

        return self, ""


class StateParseNumber(State):
    """Constrain output to a syntactically valid JSON number."""

    next_state: State | None = Field(default=None)

    def get_valid_tokens(self, vocab_index: VocabIndex) -> set[int]:
        """Return numeric IDs that preserve JSON-number validity."""
        following_literal = getattr(self.next_state, 'expected', '')

        return {
            token_id
            for token_id in vocab_index.filter_vocab.numeric_tokens
            if self._keeps_number_valid(
                vocab_index.clean_vocab[token_id], following_literal)
        }

    def _keeps_number_valid(
            self, token_str: str, following_literal: str) -> bool:
        """Check that one candidate preserves the number and next literal."""
        candidate = self.buffer + token_str

        if is_partial_json_number(candidate):
            return True

        number_text, trailing = split_complete_json_number(candidate)
        if not number_text:
            return False

        return not trailing or following_literal.startswith(trailing)

    def transition(self, token_str: str) -> tuple["State", str]:
        """Advance once a complete number is followed by a JSON delimiter."""
        self.buffer += token_str
        number_text, trailing = split_complete_json_number(self.buffer)

        is_delimited = bool(
            number_text and trailing and trailing[0] in _NUMBER_DELIMITERS)
        if not is_delimited:
            return self, ""

        return self.next_state or StateTerminal(), trailing


class StateParseString(State):
    """Constrain output to a quoted JSON string with valid escaping."""

    next_state: State | None = Field(default=None)
    has_opened: bool = Field(default=False)

    def get_valid_tokens(self, vocab_index: VocabIndex) -> set[int]:
        """Return quote or content IDs that preserve JSON-string validity."""
        if not self.has_opened:
            return vocab_index.filter_vocab.exact_quote_tokens

        valid_content = {
            token_id
            for token_id in vocab_index.filter_vocab.string_content_tokens
            if is_partial_json_string_content(
                self.buffer + vocab_index.clean_vocab[token_id])
        }
        valid_content |= {
            token_id
            for token_id in vocab_index.filter_vocab.string_closer_tokens
            if self._can_close_with(vocab_index.clean_vocab[token_id])
        }
        return valid_content

    def transition(self, token_str: str) -> tuple["State", str]:
        """Track opening and closing quotes, then pass overflow onward."""
        if not self.has_opened:
            self.has_opened = token_str == '"'
            return self, ""

        combined = self.buffer + token_str
        closing_index = self._closing_quote_index(combined)
        if closing_index is None:
            self.buffer = combined
            return self, ""

        return self.next_state or StateTerminal(), combined[closing_index + 1:]

    def _can_close_with(self, token_str: str) -> bool:
        """Return whether a candidate closes a valid string at this point."""
        combined = self.buffer + token_str
        closing_index = self._closing_quote_index(combined)
        if closing_index is None:
            return False

        content = combined[:closing_index]
        trailing = combined[closing_index + 1:]
        following_literal = getattr(self.next_state, 'expected', '')
        return (
            is_complete_json_string_content(content)
            and (not trailing or following_literal.startswith(trailing))
        )

    @staticmethod
    def _closing_quote_index(text: str) -> int | None:
        """Return the first unescaped quote index, if one exists."""
        escaped = False
        for index, character in enumerate(text):
            if escaped:
                escaped = False
                continue
            if character == '\\':
                escaped = True
            elif character == '"':
                return index
        return None
