import numpy as np
from pydantic import BaseModel, ConfigDict
from llm_sdk import Small_LLM_Model
from src.vocabulary import VocabIndex
from src.state_machine import State, StateTerminal, StateExpectLiteral


class ConstrainedDecoder(BaseModel):
    """Engine that guides LLM generation using a provided state machine."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    llm: Small_LLM_Model
    vocab_index: VocabIndex

    def generate(
            self, prompt: str, state: State, max_tokens: int = 150) -> str:
        """Produce a completion by following the rules of the given state.

        Args:
            prompt: Initial text to prime the generation.
            state: Node state that defines valid token sequences.
            max_tokens: Maximum number of model-selected tokens to generate.

        Returns:
            Generated text that adheres to the constraints
            of the state machine.

        Raises:
            ValueError: If the state machine provides
            no valid tokens to continue.
        """
        context_ids = self.llm.encode(prompt)[0].tolist()
        output_chunks: list[str] = []
        node: State = state

        generated_tokens = 0
        while not isinstance(node, StateTerminal):
            if isinstance(node, StateExpectLiteral):
                node = self._flush_literal(node, context_ids, output_chunks)
                continue

            if generated_tokens >= max_tokens:
                raise ValueError(
                    "Constrained generation ended before reaching a complete "
                    "JSON state.")

            node = self._advance_with_llm(node, context_ids, output_chunks)
            generated_tokens += 1

        return "".join(output_chunks)

    def _flush_literal(
            self,
            node: StateExpectLiteral,
            context_ids: list[int],
            output_chunks: list[str]) -> State:
        """Inject the remaining text of a literal state without querying
        the model, then move on to whatever follows it.

        Args:
            node: Literal state whose still-missing text must be emitted.
            context_ids: Token IDs fed so far; extended in place.
            output_chunks: Accumulated output fragments; appended in place.

        Returns:
            State: The state that follows the fully emitted literal.
        """
        missing_text = node.expected[len(node.buffer):]
        output_chunks.append(missing_text)
        context_ids.extend(self.llm.encode(missing_text)[0].tolist())
        return node.next_state or StateTerminal()

    def _advance_with_llm(
            self,
            node: State,
            context_ids: list[int],
            output_chunks: list[str]) -> State:
        """Ask the model for one token, apply it, and report the new state.

        Args:
            node: Current finite-state-machine node driving constraints.
            context_ids: Token IDs fed so far; extended in place with the
                chosen token.
            output_chunks: Accumulated output fragments; appended in place
                with whatever text the transition actually consumed.

        Returns:
            State: The state reached after consuming the chosen token.
        """
        token_id, token_text = self._choose_token(context_ids, node)
        next_node, leftover = self._drive_transitions(node, token_text)

        consumed = token_text[:len(token_text) - len(leftover)]
        output_chunks.append(consumed)
        context_ids.append(token_id)

        return next_node

    def _choose_token(
            self, context_ids: list[int], node: State) -> tuple[int, str]:
        """Select the best next token among those allowed by the state.

        Args:
            context_ids: Token IDs already present in the model context.
            node: Current finite-state-machine node driving constraints.

        Returns:
            tuple[int, str]: The chosen token ID and its decoded string.

        Raises:
            ValueError: If the current state exposes no valid next tokens.
        """
        candidates = node.get_valid_tokens(self.vocab_index)
        if not candidates:
            raise ValueError(
                "No valid tokens available to continue generation.")

        if len(candidates) == 1:
            token_id = next(iter(candidates))
            return token_id, self.vocab_index.clean_vocab[token_id]

        candidate_ids = np.fromiter(candidates, dtype=np.int32)
        logits = np.asarray(
            self.llm.get_logits_from_input_ids(context_ids),
            dtype=np.float32)
        best_index = int(np.argmax(logits[candidate_ids]))
        token_id = int(candidate_ids[best_index])

        return token_id, self.vocab_index.clean_vocab[token_id]

    @staticmethod
    def _drive_transitions(node: State, token_text: str) -> tuple[State, str]:
        """Feed generated text through transitions until it is fully spent.

        A single generated token may complete more than one state in a row
        (for example finishing a number and then immediately satisfying the
        literal that follows it), so transitions are repeated until either
        no text remains or a terminal state is reached.

        Args:
            node: Current state before consuming the new token.
            token_text: The raw string of the newly generated token.

        Returns:
            tuple[State, str]: The state reached once the token text has
            been consumed as far as possible, and whatever text (if any)
            could not be consumed.
        """
        remaining = token_text

        while remaining and not isinstance(node, StateTerminal):
            node, remaining = node.transition(remaining)

        return node, remaining
