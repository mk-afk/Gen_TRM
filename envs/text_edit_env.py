# envs/text_edit_env.py

import torch
import torch.nn.functional as F
import config

from envs.actions import EditAction
from envs.response_buffer import ResponseBuffer


class TextEditEnv:
    def __init__(
        self,
        tokenizer,
        model,          # LanguageTRM (or compatible LM)
        device,
        max_length=128,
        reward_cfg=None,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

        self.max_length = max_length
        self.reward_cfg = reward_cfg or {}

        self.reset()

    # -------------------------------------------------
    # Episode control
    # -------------------------------------------------
    def reset(self, prompt_ids=None):
        if prompt_ids is None:
            bos_id = (
                self.tokenizer.bos_token_id
                if self.tokenizer.bos_token_id is not None
                else self.tokenizer.eos_token_id
            )
            tokens = [bos_id]
        else:
            tokens = prompt_ids.copy()

        self.buffer = ResponseBuffer(tokens=tokens, cursor=len(tokens))
        self.done = False

        self.initial_score = self._lm_score(self.buffer.tokens)
        self.prev_score = self.initial_score
        self.has_improved = False
        self.num_edits = 0

        return self.buffer


    # -------------------------------------------------
    # State encoding
    # -------------------------------------------------
    def encode_state(self, buffer):
        window_size = getattr(config, "EDIT_WINDOW", 32)
        max_len = window_size * 2

        start = max(0, buffer.cursor - window_size)
        end   = buffer.cursor + window_size
        chunk = buffer.tokens[start:end]

        t = torch.tensor(chunk, dtype=torch.long)

        # clamp
        vocab_size = self.model.vocab_size
        t = torch.clamp(t, 0, vocab_size - 1)

        if len(t) < max_len:
            pad_val = self.tokenizer.eos_token_id or 0
            t = F.pad(t, (0, max_len - len(t)), value=pad_val)
        else:
            t = t[:max_len]

        return {
            "tokens": t,
            "cursor": buffer.cursor - start,
        }


    # -------------------------------------------------
    # Environment step
    # -------------------------------------------------
    def step(self, buffer, action, token=None):
        if self.done:
            raise RuntimeError("Episode already terminated")

        new_buffer = self._apply_edit(buffer, action, token)

        # Truncate if too long
        if len(new_buffer.tokens) > self.max_length:
            new_buffer.tokens = new_buffer.tokens[: self.max_length]
            new_buffer.cursor = min(
                new_buffer.cursor, len(new_buffer.tokens)
            )

        reward = self.compute_reward(buffer, new_buffer, action)

        # Track edit count
        if action != EditAction.STOP:
            self.num_edits += 1
        else:
            self.done = True

        self.buffer = new_buffer
        return new_buffer, reward, self.done

    # -------------------------------------------------
    # Edit logic
    # -------------------------------------------------
    def _apply_edit(self, buffer, action, token=None):
        buf = buffer.copy()

        # Never delete BOS
        if action == EditAction.DELETE:
            if buf.cursor > 0 and buf.cursor < len(buf.tokens):
                del buf.tokens[buf.cursor]

        elif action == EditAction.ADD and token is not None:
            buf.tokens.insert(buf.cursor, token)
            buf.cursor += 1

        elif action == EditAction.REFINE and token is not None:
            if buf.cursor < len(buf.tokens):
                buf.tokens[buf.cursor] = token
                buf.cursor += 1

        return buf

    # -------------------------------------------------
    # Reward function (PPO-safe, STOP-aware)
    # -------------------------------------------------
    def compute_reward(self, old_buffer, new_buffer, action):
        reward = 0.0

        # --- LM improvement reward ---
        new_score = self._lm_score(new_buffer.tokens)
        diff = new_score - self.prev_score

        lm_weight = self.reward_cfg.get("lm_weight", 1.0)

        # scale LM improvement into PPO-friendly range
        reward += lm_weight * torch.tanh(torch.tensor(diff * 10.0)).item()


        # Require meaningful improvement (noise guard)
        if diff > 1e-3:
            self.has_improved = True

        self.prev_score = new_score

        # --- Edit cost ---
        reward -= self.reward_cfg.get("edit_penalty", 0.0)

        # --- Length penalty ---
        length_penalty = self.reward_cfg.get("length_penalty", 0.0)
        delta_len = len(new_buffer.tokens) - len(old_buffer.tokens)
        reward -= length_penalty * max(0, delta_len)


        # --- STOP gating (time-aware, PPO-safe) ---
        if action == EditAction.STOP:
            stop_bonus = self.reward_cfg.get("stop_bonus", 0.0)

            # Require both improvement AND sufficient edits
            if self.has_improved and self.num_edits >= 3:
                scale = min(1.0, self.num_edits / 5)
                reward += stop_bonus * scale
            else:
                # Strong penalty for premature STOP
                reward -= 0.2

        return reward

    def _lm_score(self, tokens):
        """
        Compute negative log-likelihood score of token sequence
        """

        # ---- HARD NORMALIZATION (FINAL FIX) ----
        def flatten_tokens(x):
            if isinstance(x, torch.Tensor):
                return x.view(-1).tolist()
            if isinstance(x, (list, tuple)):
                out = []
                for v in x:
                    out.extend(flatten_tokens(v))
                return out
            # scalar
            return [int(x)]

        tokens = flatten_tokens(tokens)

        t = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)

        # Need at least 2 tokens
        if t.size(1) < 2:
            return 0.0

        with torch.no_grad():
            out = self.model(t)
            logits = out["logits"]

            shift_logits = logits[:, :-1, :]
            shift_labels = t[:, 1:]

            loss = torch.nn.functional.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                reduction="mean",
            )

        return -loss.item()

