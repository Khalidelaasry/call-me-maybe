import re
from abc import ABC, abstractmethod
from pydantic import BaseModel, ConfigDict, Field
from src.vocabulary import VocabIndex

# Optional JSON whitespace, reused by both number-matching patterns below.
_JSON_WS = r'[ \n\r\t]*'

# A number that is still "in progress" - every prefix of a valid JSON
# number matches this, including the empty string and a lone "-".
_PARTIAL_NUMBER_RE = re.compile(
    fr'^{_JSON_WS}-?(?:0|[1-9]\d*)?(?:\.\d*)?(?:[eE][+-]?\d*)?$')

# A number that is syntactically complete; used to peel the longest valid
# number off the front of a buffer, leaving whatever trails behind it.
_COMPLETE_NUMBER_PREFIX_RE = re.compile(
    fr'^{_JSON_WS}-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?')

# Characters that may legally follow a JSON number value.
_NUMBER_DELIMITERS = ',}]\n'


def is_partial_json_string_content(candidate: str) -> bool:
    """Check whether text can still form the contents of a JSON string.

    The opening and closing quotes are handled by :class:`StateParseString`.
    This recognises JSON escapes while allowing an unfinished escape at the
    end of a token, so multi-token escape sequences remain possible.
    """
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
    """Check whether text is valid completed JSON string content.

    Unlike :func:`is_partial_json_string_content`, this rejects a trailing
    backslash and a ``\\u`` escape with fewer than four hexadecimal digits.
    Those forms are useful while generating content but cannot precede the
    closing quote of a JSON string.
    """
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
    """Check whether text is a syntactically valid partial JSON number.

    Args:
        candidate: Candidate numeric prefix to validate.

    Returns:
        bool: True if candidate can still become a valid JSON number,
        otherwise False.
    """
    return _PARTIAL_NUMBER_RE.fullmatch(candidate) is not None


def split_complete_json_number(candidate: str) -> tuple[str, str]:
    """Split off the longest complete JSON number prefix from text.

    Args:
        candidate: Text beginning with potential numeric content.

    Returns:
        tuple[str, str]: Extracted number prefix and remaining suffix. Both
        are empty strings if no complete number could be found.
    """
    match = _COMPLETE_NUMBER_PREFIX_RE.match(candidate)
    if match is None:
        return "", candidate
    number_text = match.group()
    return number_text, candidate[len(number_text):]


class State(BaseModel, ABC):
    """Base state class for the state machine."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    buffer: str = Field(default="")

    @abstractmethod
    def get_valid_tokens(self, vocab_index: VocabIndex) -> set[int]:
        """Return token IDs that keep generation valid for this state.

        Args:
            vocab_index: Vocabulary metadata used to compute valid IDs.

        Returns:
            set[int]: Allowed next token IDs under this state's constraints.
        """
        raise NotImplementedError

    @abstractmethod
    def transition(self, token_str: str) -> tuple["State", str]:
        """Consume token text and compute the next parser state.

        Args:
            token_str: Generated token text to consume.

        Returns:
            tuple[State, str]: Next state and unconsumed remainder text.
        """
        raise NotImplementedError


class StateTerminal(State):
    """Terminal state - generation is complete."""

    def get_valid_tokens(self, vocab_index: VocabIndex) -> set[int]:
        """No token can ever be valid once generation has finished.

        Args:
            vocab_index: Vocabulary index parameter from the shared interface.

        Returns:
            set[int]: Always an empty set.
        """
        return set()

    def transition(self, token_str: str) -> tuple["State", str]:
        """Absorb any incoming text without leaving the terminal state.

        Args:
            token_str: Incoming token text, ignored in the terminal state.

        Returns:
            tuple[State, str]: This state and an empty remainder.
        """
        return self, ""


class StateExpectLiteral(State):
    """Expect an exact literal string to be generated."""

    expected: str = Field(...)
    next_state: State | None = Field(default=None)

    def get_valid_tokens(self, vocab_index: VocabIndex) -> set[int]:
        """Report no candidate tokens; literals bypass token selection.

        The decoder injects literal text directly rather than asking the
        model to choose among vocabulary tokens for it, so this state never
        needs to expose choices.

        Args:
            vocab_index: Vocabulary index parameter from the shared interface.

        Returns:
            set[int]: Always an empty set.
        """
        return set()

    def transition(self, token_str: str) -> tuple["State", str]:
        """Accumulate literal text and hand off once it is fully matched.

        Args:
            token_str: Incoming token text appended to the internal buffer.

        Returns:
            tuple[State, str]: Next state with overflow text when literal is
            complete; otherwise this state and an empty remainder.
        """
        self.buffer += token_str

        if not self.buffer.startswith(self.expected):
            return self, ""

        overflow = self.buffer[len(self.expected):]
        return self.next_state or StateTerminal(), overflow


class StateBranch(State):
    """Choose between multiple possible branches."""

    choices: dict[str, State] = Field(...)

    def get_valid_tokens(self, vocab_index: VocabIndex) -> set[int]:
        """Compute tokens that can continue at least one branch choice.

        Args:
            vocab_index: Vocabulary index used for literal token matching.

        Returns:
            set[int]: Token IDs that preserve a valid branch continuation.
        """
        valid_ids: set[int] = set()

        for candidate in self.choices:
            if not candidate.startswith(self.buffer):
                continue
            still_needed = candidate[len(self.buffer):]
            if still_needed:
                valid_ids |= vocab_index.get_literal_matches(still_needed)

        return valid_ids

    def transition(self, token_str: str) -> tuple["State", str]:
        """Accumulate branch text and resolve once one choice matches.

        Args:
            token_str: Generated token text to append to branch buffer.

        Returns:
            tuple[State, str]: Matching branch target and remainder, or this
            state with an empty remainder if still incomplete.
        """
        self.buffer += token_str

        for candidate, target_state in self.choices.items():
            if self.buffer.startswith(candidate):
                overflow = self.buffer[len(candidate):]
                return target_state, overflow

        return self, ""


class StateParseNumber(State):
    """Parse a JSON number."""

    next_state: State | None = Field(default=None)

    def get_valid_tokens(self, vocab_index: VocabIndex) -> set[int]:
        """Compute token IDs that keep the number parse valid.

        Args:
            vocab_index: Vocabulary index containing numeric token groups.

        Returns:
            set[int]: Token IDs that preserve a valid partial number or can
            complete a number before valid following literal text.
        """
        following_literal = getattr(self.next_state, 'expected', '')

        return {
            token_id
            for token_id in vocab_index.filter_vocab.numeric_tokens
            if self._keeps_number_valid(
                vocab_index.clean_vocab[token_id], following_literal)
        }

    def _keeps_number_valid(
            self, token_str: str, following_literal: str) -> bool:
        """Check whether appending one token keeps the number well-formed.

        Args:
            token_str: Decoded text of the candidate token.
            following_literal: Literal text expected right after the number,
                used to validate the delimiter that closes it.

        Returns:
            bool: True if the token is either a valid partial continuation,
            or completes the number just before an acceptable delimiter.
        """
        candidate = self.buffer + token_str

        if is_partial_json_number(candidate):
            return True

        number_text, trailing = split_complete_json_number(candidate)
        if not number_text:
            return False

        return not trailing or following_literal.startswith(trailing)

    def transition(self, token_str: str) -> tuple["State", str]:
        """Accumulate digits and exit once a full number is delimited.

        Args:
            token_str: Generated token text appended to the number buffer.

        Returns:
            tuple[State, str]: Next state with delimiter remainder when a
            complete number is found; otherwise this state and empty
            remainder.
        """
        self.buffer += token_str
        number_text, trailing = split_complete_json_number(self.buffer)

        is_delimited = bool(
            number_text and trailing and trailing[0] in _NUMBER_DELIMITERS)
        if not is_delimited:
            return self, ""

        return self.next_state or StateTerminal(), trailing


class StateParseString(State):
    """Parse a JSON string while preserving JSON escaping rules."""

    next_state: State | None = Field(default=None)
    has_opened: bool = Field(default=False)

    def get_valid_tokens(self, vocab_index: VocabIndex) -> set[int]:
        """Return valid token IDs for JSON string opening or content.

        Args:
            vocab_index: Vocabulary index with string-related token filters.

        Returns:
            set[int]: Quote token before opening; content and closing tokens
            once the opening quote has been generated.
        """
        if not self.has_opened:
            return vocab_index.filter_vocab.exact_quote_tokens

        valid_content = {
            token_id
            for token_id in vocab_index.filter_vocab.string_content_tokens
            if is_partial_json_string_content(
                self.buffer + vocab_index.clean_vocab[token_id])
        }
        # A string may only close with a standalone quote.  Allowing tokens
        # that merely contain a quote can emit unvalidated text after it and
        # leave the enclosing JSON object incomplete.
        if is_complete_json_string_content(self.buffer):
            valid_content |= vocab_index.filter_vocab.exact_quote_tokens
        return valid_content

    def transition(self, token_str: str) -> tuple["State", str]:
        """Consume string token text and exit when a closing quote appears.

        Args:
            token_str: Generated token text for the string value.

        Returns:
            tuple[State, str]: Next state with overflow after closing quote,
            or this state and empty remainder while string parsing
            continues.
        """
        if not self.has_opened:
            self.has_opened = token_str == '"'
            return self, ""

        if token_str != '"':
            self.buffer += token_str
            return self, ""

        return self.next_state or StateTerminal(), ""
