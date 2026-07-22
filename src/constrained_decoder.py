import numpy as np
from typing import Any
from pydantic import BaseModel, ConfigDict
from llm_sdk import Small_LLM_Model
from src.vocabulary import VocabIndex
from src.state_machine import State, StateTerminal, StateExpectLiteral


class ConstrainedDecoder(BaseModel):

    model_config = ConfigDict(arbitrary_types_allowed=True)
    llm: Small_LLM_Model
    vocab_index: VocabIndex

    def generate(
            self, prompt: str, state: State, max_tokens: int = 150) -> str:
        context_ids = self.llm.encode(prompt)[0].tolist()
        pending_ids = context_ids.copy()
        past_key_values: Any = None
        output_chunks: list[str] = []
        node: State = state

        generated_tokens = 0
        while not isinstance(node, StateTerminal):
            if isinstance(node, StateExpectLiteral):
                node = self._flush_literal(
                    node, context_ids, pending_ids, output_chunks)
                continue

            if generated_tokens >= max_tokens:
                raise ValueError(
                    "Constrained generation ended before reaching a complete "
                    "JSON state.")

            node, past_key_values = self._advance_with_llm(
                node, context_ids, pending_ids, output_chunks,
                past_key_values)
            generated_tokens += 1

        return "".join(output_chunks)

    def _flush_literal(
            self,
            node: StateExpectLiteral,
            context_ids: list[int],
            pending_ids: list[int],
            output_chunks: list[str]) -> State:
        missing_text = node.expected[len(node.buffer):]
        output_chunks.append(missing_text)
        literal_ids = self.llm.encode(missing_text)[0].tolist()
        context_ids.extend(literal_ids)
        pending_ids.extend(literal_ids)
        return node.next_state or StateTerminal()

    def _advance_with_llm(
            self,
            node: State,
            context_ids: list[int],
            pending_ids: list[int],
            output_chunks: list[str],
            past_key_values: Any) -> tuple[State, Any]:
        token_id, token_text, past_key_values = self._choose_token(
            pending_ids, node, past_key_values)
        pending_ids.clear()
        next_node, leftover = self._drive_transitions(node, token_text)

        consumed = token_text[:len(token_text) - len(leftover)]
        output_chunks.append(consumed)
        context_ids.append(token_id)
        pending_ids.append(token_id)

        return next_node, past_key_values

    def _choose_token(
            self,
            pending_ids: list[int],
            node: State,
            past_key_values: Any) -> tuple[int, str, Any]:
        candidates = node.get_valid_tokens(self.vocab_index)
        if not candidates:
            raise ValueError(
                "No valid tokens available to continue generation.")

        candidate_ids = np.fromiter(candidates, dtype=np.int32)
        raw_logits, past_key_values = self.llm.get_logits_from_input_ids(
            pending_ids, past_key_values)
        logits = np.asarray(
            raw_logits,
            dtype=np.float32)
        best_index = int(np.argmax(logits[candidate_ids]))
        token_id = int(candidate_ids[best_index])

        return (
            token_id,
            self.vocab_index.clean_vocab[token_id],
            past_key_values,
        )

    @staticmethod
    def _drive_transitions(node: State, token_text: str) -> tuple[State, str]:
        remaining = token_text

        while remaining and not isinstance(node, StateTerminal):
            node, remaining = node.transition(remaining)

        return node, remaining
