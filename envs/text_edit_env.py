# envs/text_edit_env.py

import torch
import config

from envs.actions import EditAction
from envs.response_buffer import ResponseBuffer


class TextEditEnv:
    def __init__(
        self,
        tokenizer,
        max_length=128,
        reward_cfg=None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.reward_cfg = reward_cfg or {}
        self.reset()

    # -------------------------
    # Episode control
    # -------------------------
    def reset(self):
            """
            Initialize a fresh response buffer with a BOS token
            so the model always sees at least one token.
            """
            if self.tokenizer.bos_token_id is not None:
                bos_id = self.tokenizer.bos_token_id
            else:
                # GPT-2 fallback
                bos_id = self.tokenizer.eos_token_id

            self.buffer = ResponseBuffer(tokens=[bos_id], cursor=1)
            self.done = False
            self.prev_score = 0.0
            return self.buffer

    # -------------------------
    # State encoding
    # -------------------------
    def encode_state(self, buffer):
        """
        Local window around cursor.
        """
        window = config.EDIT_WINDOW

        start = max(0, buffer.cursor - window)
        end = buffer.cursor + window

        tokens = buffer.tokens[start:end]
        cursor_pos = buffer.cursor - start

        return {
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "cursor": cursor_pos,
        }

    # -------------------------
    # Environment step
    # -------------------------
    def step(self, buffer, action, token=None):
        """
        Apply edit action and compute reward.
        """
        if self.done:
            raise RuntimeError("Episode already terminated")

        # Apply edit
        new_buffer = self._apply_edit(buffer, action, token)

        # Truncate if too long
        if len(new_buffer.tokens) > self.max_length:
            new_buffer.tokens = new_buffer.tokens[: self.max_length]
            new_buffer.cursor = min(
                new_buffer.cursor, len(new_buffer.tokens)
            )

        # Compute reward
        reward = self.compute_reward(buffer, new_buffer, action)

        # STOP action ends episode
        if action == EditAction.STOP:
            self.done = True

        self.buffer = new_buffer
        return new_buffer, reward, self.done

    # -------------------------
    # Edit logic (your code)
    # -------------------------
    def _apply_edit(self, buffer, action, token=None):
        buf = buffer.copy()

        if action == EditAction.MOVE_LEFT:
            buf.cursor = max(0, buf.cursor - 1)

        elif action == EditAction.MOVE_RIGHT:
            buf.cursor = min(len(buf.tokens), buf.cursor + 1)

        elif action == EditAction.DELETE and buf.tokens:
            if buf.cursor < len(buf.tokens):
                del buf.tokens[buf.cursor]

        elif action == EditAction.INSERT and token is not None:
            buf.tokens.insert(buf.cursor, token)
            buf.cursor += 1

        elif action == EditAction.REPLACE and token is not None:
            if buf.cursor < len(buf.tokens):
                buf.tokens[buf.cursor] = token

        return buf

    # -------------------------
    # Reward function
    # -------------------------
    def compute_reward(self, old_buffer, new_buffer, action):
        """
        Simple shaped reward.
        """

        reward = 0.0

        # Penalty per edit
        reward -= self.reward_cfg.get("edit_penalty", 0.0)

        # Length penalty
        reward -= self.reward_cfg.get("length_penalty", 0.0) * len(new_buffer.tokens)

        # STOP bonus
        if action == EditAction.STOP:
            reward += self.reward_cfg.get("stop_bonus", 0.0)

        return reward
