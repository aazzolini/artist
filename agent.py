"""
Canvas agent that manages a reference letter canvas, a state canvas, and an output canvas.
Supports requesting new letter references, issuing drawing commands to canvases, and
comparing the output canvas to the reference. All canvases are kept the same size.
"""

import torch
import torch.nn.functional as F
import random
from pathlib import Path
from typing import Generator, Optional, Tuple

from gen_data import generate_random_letters
from gpu_canvas import GPUCanvas


class CanvasAgent:
    """
    Simple agent that holds three canvases:
    - reference_canvas: generated letter canvas to match
    - state_canvas: scratch pad for intermediate work
    - output_canvas: final output to compare against the reference
    """

    def __init__(
        self,
        letter_stream: Optional[Generator] = None,
        device: Optional[str] = None,
    ):
        # letter_stream yields (svg, img_array, metadata) tuples
        self.letter_stream = letter_stream or generate_random_letters()
        self.device = device

        self.reference_canvas: Optional[GPUCanvas] = None
        self.reference_metadata: Optional[dict] = None
        self.state_canvas: Optional[GPUCanvas] = None
        self.output_canvas: Optional[GPUCanvas] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.learning_rate = 1e-4  # conservative default
        # PPO hyperparameters
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        self.ppo_epochs = 4
        self.rollout_length = 32
        self.entropy_coef = 0.2
        self.entropy_floor_coef = 0.5
        self.target_entropy = 0.75  # nudges toward near-uniform over 8 actions (max ln(8)=2.08)
        self.value_coef = 0.5
        self.rollout_buffer = []

        # Allowed canvas commands the agent can emit to its canvases
        self._allowed_commands = {"clear", "add_rectangle"}
        self._mutable_targets = {"state", "output"}

    def request_new_reference(self) -> dict:
        """
        Ask for a new randomly generated letter canvas.
        """
        svg, img_array, metadata = next(self.letter_stream)
        self.reference_canvas = GPUCanvas.from_array(
            img_array[:, :, :3], device=self.device
        )
        self.reference_metadata = metadata
        return metadata

    def reset_canvases(self):
        """
        Reset state and output canvases to blank white canvases matching the reference size.
        """
        if self.reference_canvas is None:
            raise RuntimeError(
                "Cannot reset canvases without a reference. Call request_new_reference() first."
            )
        self.state_canvas = GPUCanvas(
            self.reference_canvas.width,
            self.reference_canvas.height,
            device=self.device,
        )
        self.output_canvas = GPUCanvas(
            self.reference_canvas.width,
            self.reference_canvas.height,
            device=self.device,
        )

    def _resolve_canvas(self, target: str) -> GPUCanvas:
        if self.reference_canvas is None:
            raise RuntimeError(
                "No reference loaded. Call request_new_reference() first."
            )

        target_map = {
            "state": self.state_canvas,
            "output": self.output_canvas,
            "reference": self.reference_canvas,
        }
        canvas = target_map.get(target)
        if canvas is None:
            raise ValueError(
                f"Unknown canvas target '{target}'. Choose from {list(target_map.keys())}."
            )
        return canvas

    def emit_canvas_command(self, target: str, command: str, *args, **kwargs):
        """
        Send a drawing command to one of the canvases.
        Allowed commands map directly to GPUCanvas methods (currently: clear, add_rectangle).
        """
        if command not in self._allowed_commands:
            raise ValueError(
                f"Command '{command}' is not allowed. Allowed: {sorted(self._allowed_commands)}"
            )

        if target == "reference":
            raise ValueError(
                "Reference canvas is read-only; cannot emit mutating commands."
            )
        if target not in self._mutable_targets:
            raise ValueError(
                f"Unknown canvas target '{target}'. Choose from {sorted(self._mutable_targets)}."
            )

        canvas = self._resolve_canvas(target)
        fn = getattr(canvas, command, None)
        if fn is None:
            raise AttributeError(f"Canvas does not support command '{command}'")

        return fn(*args, **kwargs)

    def compare_output_to_reference(self) -> float:
        """
        Compute normalized L2 distance between output and reference canvases.
        Uses inverted intensities so white background contributes zero.
        metric = ||a' - b'||2 / (||a'||2 + ||b'||2), where a' = 1 - a/255.
        """
        if self.reference_canvas is None or self.output_canvas is None:
            raise RuntimeError(
                "Reference and output canvases must be initialized first."
            )
        a = 1.0 - self.output_canvas.canvas.float() / 255.0
        b = 1.0 - self.reference_canvas.canvas.float() / 255.0
        diff_norm = (a - b).pow(2).sum().sqrt()
        a_norm = a.pow(2).sum().sqrt()
        b_norm = b.pow(2).sum().sqrt()
        denom = a_norm + b_norm
        if denom == 0:
            return 0.0
        return float((diff_norm / denom).item())

    def get_reference_info(self) -> Tuple[Optional[dict], Optional[GPUCanvas]]:
        """Return the metadata and reference canvas for inspection."""
        return self.reference_metadata, self.reference_canvas

    def _ensure_canvas_alignment(self):
        """
        Ensure state and output canvases exist and match the reference size.
        """
        if self.reference_canvas is None:
            return
        width, height = self.reference_canvas.width, self.reference_canvas.height
        if (
            self.state_canvas is None
            or self.state_canvas.width != width
            or self.state_canvas.height != height
        ):
            self.state_canvas = GPUCanvas(width, height, device=self.device)
        if (
            self.output_canvas is None
            or self.output_canvas.width != width
            or self.output_canvas.height != height
        ):
            self.output_canvas = GPUCanvas(width, height, device=self.device)


class RandomCanvasAgent(CanvasAgent):
    """
    An agent that chooses random actions across its canvases.
    Actions:
    - request_new_reference
    - reset_canvases
    - clear (state/output/reference)
    - add_rectangle (state/output/reference)
    - compare output vs reference
    """

    def __init__(
        self,
        letter_stream: Optional[Generator] = None,
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(letter_stream=letter_stream, device=device)
        self.rng = random.Random(seed)
        self.step_index = 0
        self.policy: Optional[MOEPolicy] = None
        self.prev = None
        # Small incentives for exploration and non-empty canvases
        self.change_reward_scale = 0.01
        self.occupancy_reward_scale = 0.005
        self.compare_reward_scale = 0.1
        self.compare_bonus = 0.01
        self.prev_state_snapshot = None
        self.prev_output_snapshot = None

    def _random_rect_args(self, canvas: GPUCanvas):
        width, height = canvas.width, canvas.height
        rect_w = self.rng.randint(max(1, width // 12), max(1, width // 2))
        rect_h = self.rng.randint(max(1, height // 12), max(1, height // 2))
        x1 = self.rng.randint(0, max(0, width - rect_w))
        y1 = self.rng.randint(0, max(0, height - rect_h))
        x2 = x1 + rect_w
        y2 = y1 + rect_h
        color = (
            self.rng.randint(0, 255),
            self.rng.randint(0, 255),
            self.rng.randint(0, 255),
        )
        return x1, y1, x2, y2, color

    def _copy_canvas(self, canvas: GPUCanvas) -> GPUCanvas:
        # Copy via CPU numpy to decouple from future mutations.
        return GPUCanvas.from_array(canvas.to_numpy(), device=self.device)

    def _maybe_init_policy(self, force: bool = False):
        if self.reference_canvas is None:
            return
        if self.policy is not None and not force:
            return
        width, height = self.reference_canvas.width, self.reference_canvas.height
        self.policy = MOEPolicy(width, height)
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(
                self.policy.parameters(), lr=self.learning_rate
            )

    def step(self) -> dict:
        """
        Execute a single random action. Returns a dict describing the action.
        """
        self.step_index += 1
        step_id = self.step_index

        if self.reference_canvas is None:
            metadata = self.request_new_reference()
            self._ensure_canvas_alignment()
            self._maybe_init_policy()
            return {
                "action": "request_reference",
                "metadata": metadata,
                "step_index": step_id,
            }

        self._ensure_canvas_alignment()
        self._maybe_init_policy()

        action_info = self.policy.act(
            self.reference_canvas,
            self.state_canvas,
            self.output_canvas,
        )
        action = action_info["action"]
        action_log_prob = action_info.get("action_log_prob")
        entropy = action_info.get("entropy")
        entropy_val = float(entropy.item()) if entropy is not None else 0.0

        if self.prev is None or self.prev["action"] != action:
            reward = 1e-6
        else:
            reward = -1e-6
        base_reward = reward
        compare_reward_component = 0.0  # small bonus for taking compare action
        compare_value_reward = 0.0     # scaled by quality of comparison
        compare_value_loss = None
        result = None

        if action == "no_action":
            result = {"action": action, "entropy": entropy_val, "step_index": step_id}

        if action == "request_reference":
            metadata = self.request_new_reference()
            # self._maybe_init_policy(force=True)
            result = {
                "action": action,
                "metadata": metadata,
                "entropy": entropy_val,
                "step_index": step_id,
            }

        if action == "reset_canvases":
            self.reset_canvases()
            result = {"action": action, "entropy": entropy_val, "step_index": step_id}

        if action.startswith("clear_"):
            target = action.split("_")[1]
            self.emit_canvas_command(target, "clear")
            result = {"action": action, "entropy": entropy_val, "step_index": step_id}

        if action.startswith("add_rect_"):
            target = action.split("_")[2]
            canvas = self._resolve_canvas(target)
            if action_info is not None:
                x1, y1, x2, y2 = action_info["rect"]
                color = action_info["color"]
            else:
                x1, y1, x2, y2, color = self._random_rect_args(canvas)
            self.emit_canvas_command(
                target, "add_rectangle", x1, y1, x2, y2, color=color
            )
            result = {
                "action": action,
                "target": target,
                "rect": (x1, y1, x2, y2),
                "color": color,
                "entropy": entropy_val,
                "step_index": step_id,
            }

        if action == "compare":
            if self.reference_canvas is None or self.output_canvas is None:
                raise RuntimeError(
                    "Reference and output canvases must be initialized first."
                )
            loss = self.compare_output_to_reference()
            reward += self.compare_bonus
            compare_reward_component = self.compare_bonus
            result = {
                "action": action,
                "loss": loss,
                "reward": reward,
                "entropy": entropy_val,
                "metadata": self.reference_metadata,
                "step_index": step_id,
            }
        assert result is not None, "No action result generated"

        # Provide comparison value reward every step (if canvases available)
        compare_value_reward, compare_value_loss = self._comparison_value_reward()
        reward += compare_value_reward

        # Encourage spatial changes and non-empty canvases
        delta_reward, occupancy_reward = self._spatial_rewards()
        reward += delta_reward + occupancy_reward

        if "reward" in result:
            result["reward"] = reward
        result["reward_base"] = base_reward
        result["reward_compare"] = compare_reward_component
        result["reward_change"] = delta_reward
        result["reward_occupancy"] = occupancy_reward
        result["reward_compare_action"] = compare_reward_component
        result["reward_compare_value"] = compare_value_reward
        if compare_value_loss is not None:
            result["compare_value_loss"] = compare_value_loss

        # Update snapshots after computing rewards
        self._update_snapshots()

        self._store_transition(action_info, reward)

        if len(self.rollout_buffer) >= self.rollout_length:
            bootstrap_value = self._current_value()
            self._ppo_update(bootstrap_value)

        self.prev = result
        print(
            f"[step {step_id}] action={action} reward={reward:.6f} "
            f"(delta={delta_reward:.6f}, occ={occupancy_reward:.6f})"
        )
        return result

    def _canvas_change_and_occupancy(
        self, canvas: Optional[GPUCanvas], prev_snapshot: Optional[torch.Tensor]
    ) -> Tuple[float, float, Optional[torch.Tensor]]:
        """
        Compute mean absolute change vs previous snapshot and occupancy of non-white pixels.
        """
        if canvas is None:
            return 0.0, 0.0, prev_snapshot
        arr = torch.from_numpy(canvas.to_numpy().astype("float32"))
        occupancy = float(max(0.0, (255.0 - arr.mean().item()) / 255.0))
        if prev_snapshot is not None:
            change = float((arr - prev_snapshot).abs().mean().item() / 255.0)
        else:
            change = 0.0
        return change, occupancy, arr

    def _spatial_rewards(self) -> Tuple[float, float]:
        """
        Returns (change_reward, occupancy_reward) from state/output canvases.
        """
        change_s, occ_s, _ = self._canvas_change_and_occupancy(
            self.state_canvas, self.prev_state_snapshot
        )
        change_o, occ_o, _ = self._canvas_change_and_occupancy(
            self.output_canvas, self.prev_output_snapshot
        )
        change_reward = self.change_reward_scale * (change_s + change_o)
        occupancy_reward = self.occupancy_reward_scale * (occ_s + occ_o)
        return change_reward, occupancy_reward

    def _update_snapshots(self):
        _, _, self.prev_state_snapshot = self._canvas_change_and_occupancy(
            self.state_canvas, None
        )
        _, _, self.prev_output_snapshot = self._canvas_change_and_occupancy(
            self.output_canvas, None
        )

    def _comparison_value_reward(self) -> Tuple[float, Optional[float]]:
        """
        Compute reward shaped by how close output is to reference (lower loss = higher reward).
        Returned reward is in (0, compare_reward_scale] when canvases are available.
        """
        if self.reference_canvas is None or self.output_canvas is None:
            return 0.0, None
        loss = self.compare_output_to_reference()
        reward = self.compare_reward_scale * (1.0 / (1.0 + loss))
        return reward, loss

    def _store_transition(self, action_info: dict, reward: float):
        """
        Save transition data for PPO updates.
        """
        if self.policy is None:
            return
        entry = {
            "log_prob": action_info.get("action_log_prob").detach()
            if action_info.get("action_log_prob") is not None
            else None,
            "value": action_info.get("value").detach()
            if action_info.get("value") is not None
            else None,
            "entropy": action_info.get("entropy").detach()
            if action_info.get("entropy") is not None
            else None,
            "action_index": action_info.get("action_index"),
            "features": action_info.get("features"),
            "reward": float(reward),
        }
        self.rollout_buffer.append(entry)

    def _current_value(self) -> float:
        """
        Estimate the current state's value for bootstrap.
        """
        if self.policy is None or self.reference_canvas is None:
            return 0.0
        logits, value, _ = self.policy.forward(
            self.reference_canvas,
            self.state_canvas,
            self.output_canvas,
            return_features=True,
        )
        return float(value.detach().item())

    def _ppo_update(self, bootstrap_value: float):
        """
        Run a PPO update over the collected rollout buffer.
        """
        if not self.rollout_buffer or self.policy is None or self.optimizer is None:
            self.rollout_buffer.clear()
            return

        rewards = [t["reward"] for t in self.rollout_buffer]
        values = [t["value"].detach() if t["value"] is not None else torch.tensor(0.0) for t in self.rollout_buffer]
        values.append(torch.tensor(bootstrap_value))

        gae = 0.0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1].item() - values[step].item()
            gae = delta + self.gamma * self.gae_lambda * gae
            returns.insert(0, gae + values[step].item())

        values_tensor = torch.stack(values[:-1]).float()
        returns_tensor = torch.tensor(
            returns, dtype=torch.float32, device=values_tensor.device
        )
        advantages = returns_tensor - values_tensor.detach()
        # Normalize advantages for stability
        adv_mean = advantages.mean()
        adv_std = advantages.std(unbiased=False) + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        for _ in range(self.ppo_epochs):
            policy_losses = []
            value_losses = []
            entropy_terms = []
            for transition, ret, adv in zip(
                self.rollout_buffer, returns_tensor, advantages
            ):
                feats = transition["features"]
                action_idx = transition["action_index"]
                old_log_prob = transition["log_prob"]
                if feats is None or action_idx is None or old_log_prob is None:
                    continue
                new_log_prob, entropy, value_pred = self.policy.evaluate_from_features(
                    feats, action_idx
                )
                ratio = torch.exp(new_log_prob - old_log_prob)
                clipped = torch.clamp(
                    ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon
                ) * adv
                policy_loss = -torch.min(ratio * adv, clipped)
                value_loss = F.mse_loss(value_pred, ret.to(value_pred))
                policy_losses.append(policy_loss)
                value_losses.append(value_loss)
                entropy_terms.append(entropy)

            if not policy_losses:
                continue

            self.optimizer.zero_grad()
            entropy_stack = torch.stack(entropy_terms)
            entropy_mean = entropy_stack.mean()
            entropy_deficit = torch.relu(
                torch.tensor(self.target_entropy, device=entropy_stack.device)
                - entropy_stack
            ).mean()
            loss = (
                torch.stack(policy_losses).mean()
                + self.value_coef * torch.stack(value_losses).mean()
                - self.entropy_coef * entropy_mean
                + self.entropy_floor_coef * entropy_deficit
            )
            loss.backward()
            self.optimizer.step()

        self.rollout_buffer.clear()


class ConvEncoder(torch.nn.Module):
    """
    Lightweight CNN encoder that maps an RGB canvas to a fixed embedding.
    """

    def __init__(self, in_channels: int = 3, hidden: int = 64, embed_dim: int = 256):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, hidden, kernel_size=5, stride=2, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(hidden, hidden * 2, kernel_size=5, stride=2, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(hidden * 2, hidden * 4, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(hidden * 4, hidden * 4, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.proj = torch.nn.Linear(hidden * 4, embed_dim)
        self.norm = torch.nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W) float in [0,1]
        returns: (B, embed_dim)
        """
        h = self.conv(x)
        h = h.mean(dim=(-2, -1))  # global average pool
        h = self.proj(h)
        return self.norm(h)


class MOEPolicy(torch.nn.Module):
    """Policy + value head over convolutional canvas encodings."""

    def __init__(
        self,
        width: int,
        height: int,
        num_channels: int = 3,
        actions: Optional[Tuple[str, ...]] = None,
        checkpoint_path: Optional[str] = None,
        encoder_hidden: int = 64,
        encoder_dim: int = 256,
    ):
        super().__init__()
        self.width = int(width)
        self.height = int(height)
        self.actions = actions or (
            "no_action",
            "request_reference",
            "reset_canvases",
            "clear_state",
            "clear_output",
            "add_rect_state",
            "add_rect_output",
            "compare",
        )
        self.encoder = ConvEncoder(
            in_channels=num_channels, hidden=encoder_hidden, embed_dim=encoder_dim
        )
        self.embed_dim = encoder_dim
        self.feature_dim = self.embed_dim * 3  # reference + state + output
        self.logit_dim = len(self.actions)
        rect_params = 4 * 2  # mean + log_std per coordinate
        color_params = 3 * 2  # mean + log_std per channel
        self.policy_head = torch.nn.Linear(
            self.feature_dim,
            len(self.actions) + rect_params + color_params,
        )
        self.value_head = torch.nn.Linear(self.feature_dim, 1)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.last_checkpoint_step = 0
        if self.checkpoint_path and self.checkpoint_path.exists():
            self.load_checkpoint()

    def _flatten_canvas(self, canvas: Optional[GPUCanvas]) -> torch.Tensor:
        if canvas is None:
            device = self.policy_head.weight.device
            return torch.zeros(self.embed_dim, dtype=torch.float32, device=device)
        tensor = canvas.canvas.float().permute(2, 0, 1).unsqueeze(0) / 255.0
        tensor = tensor.to(self.policy_head.weight.device)
        return self.encoder(tensor).squeeze(0)

    def _encode_state(
        self, reference: GPUCanvas, state: GPUCanvas, output: GPUCanvas
    ) -> torch.Tensor:
        feats = torch.cat(
            (
                self._flatten_canvas(reference),
                self._flatten_canvas(state),
                self._flatten_canvas(output),
            )
        )
        return feats.to(self.policy_head.weight.device)

    def forward(
        self,
        reference: GPUCanvas,
        state: GPUCanvas,
        output: GPUCanvas,
        return_features: bool = False,
    ):
        feats = self._encode_state(reference, state, output)
        logits = self.policy_head(feats)
        value = self.value_head(feats).squeeze(-1)
        if return_features:
            return logits, value, feats
        return logits, value

    def _split_outputs(self, logits: torch.Tensor):
        offset = len(self.actions)
        action_logits = logits[:offset]
        rect_mu = logits[offset : offset + 4]
        rect_log_std = logits[offset + 4 : offset + 8]
        color_mu = logits[offset + 8 : offset + 11]
        color_log_std = logits[offset + 11 : offset + 14]
        return action_logits, rect_mu, rect_log_std, color_mu, color_log_std

    def sample_action(
        self, action_logits: torch.Tensor, return_log_prob: bool = False
    ):
        probs = torch.softmax(action_logits, dim=0)
        dist = torch.distributions.Categorical(probs=probs)
        idx_tensor = dist.sample()
        idx = int(idx_tensor.item())
        entropy = dist.entropy()
        if return_log_prob:
            return self.actions[idx], dist.log_prob(idx_tensor), idx, entropy
        return self.actions[idx], None, idx, entropy

    def _sample_gaussian(
        self, mean: torch.Tensor, log_std: torch.Tensor
    ) -> torch.Tensor:
        std = F.softplus(log_std) + 1e-3
        dist = torch.distributions.Normal(mean, std)
        return dist.sample()

    def sample_rectangle(
        self,
        rect_mu: torch.Tensor,
        rect_log_std: torch.Tensor,
    ) -> Tuple[Tuple[int, int, int, int], torch.Tensor]:
        rect_raw = self._sample_gaussian(rect_mu, rect_log_std)
        coords = torch.sigmoid(rect_raw)
        x1 = int(coords[0].item() * self.width)
        y1 = int(coords[1].item() * self.height)
        x2 = int(coords[2].item() * self.width)
        y2 = int(coords[3].item() * self.height)
        # ensure ordering
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        x2 = max(x2, x1 + 1)
        y2 = max(y2, y1 + 1)
        return (x1, y1, x2, y2), rect_raw

    def sample_color(
        self,
        color_mu: torch.Tensor,
        color_log_std: torch.Tensor,
    ) -> Tuple[Tuple[int, int, int], torch.Tensor]:
        color_raw = self._sample_gaussian(color_mu, color_log_std)
        rgb_f = torch.sigmoid(color_raw) * 255.0
        rgb = tuple(int(c.item()) for c in rgb_f)
        return rgb, color_raw

    def act(
        self,
        reference: GPUCanvas,
        state: GPUCanvas,
        output: GPUCanvas,
    ) -> dict:
        logits, value, feats = self.forward(
            reference,
            state,
            output,
            return_features=True,
        )
        action_logits, rect_mu, rect_log_std, color_mu, color_log_std = (
            self._split_outputs(logits)
        )
        rect_coords, rect_tensor = self.sample_rectangle(rect_mu, rect_log_std)
        color_rgb, color_tensor = self.sample_color(color_mu, color_log_std)
        action, action_log_prob, action_idx, entropy = self.sample_action(
            action_logits, return_log_prob=True
        )
        return {
            "rect": rect_coords,
            "rect_tensor": rect_tensor,
            "color": color_rgb,
            "color_tensor": color_tensor,
            "action_logits": action_logits,
            "action": action,
            "action_index": action_idx,
            "action_log_prob": action_log_prob,
            "entropy": entropy,
            "value": value.detach(),
            "features": feats.detach(),
        }

    def evaluate_from_features(
        self, feats: torch.Tensor, action_index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Recompute log prob, entropy, and value for a stored feature vector.
        """
        logits = self.policy_head(feats)
        value = self.value_head(feats).squeeze(-1)
        action_logits = logits[: len(self.actions)]
        log_probs = torch.log_softmax(action_logits, dim=0)
        probs = torch.softmax(action_logits, dim=0)
        entropy = -(probs * log_probs).sum()
        return log_probs[action_index], entropy, value

    def save_checkpoint(
        self,
        optimizer: Optional[torch.optim.Optimizer] = None,
        step: Optional[int] = None,
        path: Optional[str] = None,
    ):
        target_path = Path(path) if path else self.checkpoint_path
        if target_path is None:
            raise ValueError("No checkpoint path specified")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state": self.state_dict(),
            "step": step if step is not None else self.last_checkpoint_step,
        }
        if optimizer is not None:
            payload["optimizer_state"] = optimizer.state_dict()
        torch.save(payload, target_path)
        self.last_checkpoint_step = payload["step"]
        return target_path

    def load_checkpoint(
        self,
        optimizer: Optional[torch.optim.Optimizer] = None,
        path: Optional[str] = None,
    ) -> Optional[int]:
        source_path = Path(path) if path else self.checkpoint_path
        if source_path is None or not source_path.exists():
            return None
        checkpoint = torch.load(
            source_path, map_location=self.policy_head.weight.device
        )
        self.load_state_dict(checkpoint["model_state"])
        if optimizer is not None and "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.last_checkpoint_step = int(checkpoint.get("step", 0))
        return self.last_checkpoint_step
