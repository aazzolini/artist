"""
Canvas agent that manages a reference letter canvas, a state canvas, and an output canvas.
Supports requesting new letter references, issuing drawing commands to canvases, and
comparing the output canvas to the reference. All canvases are kept the same size.
"""
import math
import os
import random
from pathlib import Path
from typing import Generator, List, Optional, Tuple

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

        # Allowed canvas commands the agent can emit to its canvases
        self._allowed_commands = {"clear", "add_rectangle"}
        self._mutable_targets = {"state", "output"}

    def request_new_reference(self) -> dict:
        """
        Ask for a new randomly generated letter canvas.
        """
        svg, img_array, metadata = next(self.letter_stream)
        self.reference_canvas = GPUCanvas.from_array(img_array[:, :, :3], device=self.device)
        self.reference_metadata = metadata
        return metadata

    def reset_canvases(self):
        """
        Reset state and output canvases to blank white canvases matching the reference size.
        """
        if self.reference_canvas is None:
            raise RuntimeError("Cannot reset canvases without a reference. Call request_new_reference() first.")
        self.state_canvas = GPUCanvas(self.reference_canvas.width, self.reference_canvas.height, device=self.device)
        self.output_canvas = GPUCanvas(self.reference_canvas.width, self.reference_canvas.height, device=self.device)

    def _resolve_canvas(self, target: str) -> GPUCanvas:
        if self.reference_canvas is None:
            raise RuntimeError("No reference loaded. Call request_new_reference() first.")

        target_map = {
            "state": self.state_canvas,
            "output": self.output_canvas,
            "reference": self.reference_canvas,
        }
        canvas = target_map.get(target)
        if canvas is None:
            raise ValueError(f"Unknown canvas target '{target}'. Choose from {list(target_map.keys())}.")
        return canvas

    def emit_canvas_command(self, target: str, command: str, *args, **kwargs):
        """
        Send a drawing command to one of the canvases.
        Allowed commands map directly to GPUCanvas methods (currently: clear, add_rectangle).
        """
        if command not in self._allowed_commands:
            raise ValueError(f"Command '{command}' is not allowed. Allowed: {sorted(self._allowed_commands)}")

        if target == "reference":
            raise ValueError("Reference canvas is read-only; cannot emit mutating commands.")
        if target not in self._mutable_targets:
            raise ValueError(f"Unknown canvas target '{target}'. Choose from {sorted(self._mutable_targets)}.")

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
            raise RuntimeError("Reference and output canvases must be initialized first.")
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
        if self.state_canvas is None or self.state_canvas.width != width or self.state_canvas.height != height:
            self.state_canvas = GPUCanvas(width, height, device=self.device)
        if self.output_canvas is None or self.output_canvas.width != width or self.output_canvas.height != height:
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
        policy_max_context_steps: int = 100,
    ):
        super().__init__(letter_stream=letter_stream, device=device)
        self.rng = random.Random(seed)
        self.step_index = 0
        self.policy: Optional[MOEPolicy] = None
        self.policy_max_context_steps: int = policy_max_context_steps
        self._feature_history: List[torch.Tensor] = []

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
        self.policy = MOEPolicy(width, height, max_context_steps=self.policy_max_context_steps)
        self._feature_history = []

    def _update_feature_history(self, action_info: Optional[dict]):
        if action_info is None:
            return
        snapshot = action_info.get("logit_history")
        if snapshot:
            self._feature_history = [feat.clone() for feat in snapshot][-self.policy_max_context_steps :]
        logits = action_info.get("action_logits")
        if logits is not None:
            self._feature_history.append(logits.clone())
            if len(self._feature_history) > self.policy_max_context_steps:
                self._feature_history = self._feature_history[-self.policy_max_context_steps :]

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
            return {"action": "request_reference", "metadata": metadata, "step_index": step_id}

        self._ensure_canvas_alignment()
        self._maybe_init_policy()

        action_info = self.policy.act(
            self.reference_canvas,
            self.state_canvas,
            self.output_canvas,
            history_features=self._feature_history,
        )
        self._update_feature_history(action_info)
        action = self.policy.sample_action(action_info["action_logits"])

        if action == "no_action":
            return {"action": action, "step_index": step_id}

        if action == "request_reference":
            metadata = self.request_new_reference()
            # self._maybe_init_policy(force=True)
            return {"action": action, "metadata": metadata, "step_index": step_id}

        if action == "reset_canvases":
            self.reset_canvases()
            return {"action": action, "step_index": step_id}

        if action.startswith("clear_"):
            target = action.split("_")[1]
            self.emit_canvas_command(target, "clear")
            return {"action": action, "step_index": step_id}

        if action.startswith("add_rect_"):
            target = action.split("_")[2]
            canvas = self._resolve_canvas(target)
            if action_info is not None:
                x1, y1, x2, y2 = action_info["rect"]
                color = action_info["color"]
            else:
                x1, y1, x2, y2, color = self._random_rect_args(canvas)
            self.emit_canvas_command(target, "add_rectangle", x1, y1, x2, y2, color=color)
            return {
                "action": action,
                "target": target,
                "rect": (x1, y1, x2, y2),
                "color": color,
                "step_index": step_id,
            }

        if action == "compare":
            if self.reference_canvas is None or self.output_canvas is None:
                raise RuntimeError("Reference and output canvases must be initialized first.")
            loss = self.compare_output_to_reference()
            return {
                "action": action,
                "loss": loss,
                "metadata": self.reference_metadata,
                "step_index": step_id,
            }
        return {"action": action, "step_index": step_id}


import torch
import torch.nn.functional as F


class MOEPolicy(torch.nn.Module):
    """Single-layer MLP policy with global trajectory attention."""

    def __init__(
        self,
        width: int,
        height: int,
        num_channels: int = 3,
        actions: Optional[Tuple[str, ...]] = None,
        checkpoint_path: Optional[str] = None,
        max_context_steps: int = 512,
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
        self.canvas_feature_dim = self.width * self.height * num_channels
        self.feature_dim = self.canvas_feature_dim * 3  # reference + state + output
        self.logit_dim = len(self.actions)
        rect_params = 4 * 2  # mean + log_std per coordinate
        color_params = 3 * 2  # mean + log_std per channel
        self.policy_head = torch.nn.Linear(
            self.feature_dim + self.logit_dim,
            len(self.actions) + rect_params + color_params,
        )
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.last_checkpoint_step = 0
        if self.checkpoint_path and self.checkpoint_path.exists():
            self.load_checkpoint()
        self.max_context_steps = max(1, max_context_steps)

    def _flatten_canvas(self, canvas: Optional[GPUCanvas]) -> torch.Tensor:
        if canvas is None:
            device = self.policy_head.weight.device
            size = self.canvas_feature_dim
            return torch.zeros(size, dtype=torch.float32, device=device)
        tensor = canvas.canvas.float() / 255.0
        return tensor.reshape(-1)

    def _encode_state(self, reference: GPUCanvas, state: GPUCanvas, output: GPUCanvas) -> torch.Tensor:
        feats = torch.cat(
            (
                self._flatten_canvas(reference),
                self._flatten_canvas(state),
                self._flatten_canvas(output),
            )
        )
        return feats.to(self.policy_head.weight.device)

    def _summarize_logit_history(self, history_logits: Optional[List[torch.Tensor]]) -> torch.Tensor:
        device = self.policy_head.weight.device
        if not history_logits:
            return torch.zeros(self.logit_dim, dtype=torch.float32, device=device)
        
        recent = history_logits[-self.max_context_steps :]
        print(len(recent))
        filtered = [logit.to(device) for logit in recent if logit is not None]
        if not filtered:
            return torch.zeros(self.logit_dim, dtype=torch.float32, device=device)
        stacked = torch.stack(filtered, dim=0)
        return stacked.mean(dim=0)

    def forward(
        self,
        reference: GPUCanvas,
        state: GPUCanvas,
        output: GPUCanvas,
        history_features: Optional[List[torch.Tensor]] = None,
        return_features: bool = False,
    ):
        feats = self._encode_state(reference, state, output)
        context = self._summarize_logit_history(history_features)
        combined = torch.cat([feats, context], dim=0)
        logits = self.policy_head(combined)
        if return_features:
            return logits, feats
        return logits

    def _split_outputs(self, logits: torch.Tensor):
        offset = len(self.actions)
        action_logits = logits[:offset]
        rect_mu = logits[offset : offset + 4]
        rect_log_std = logits[offset + 4 : offset + 8]
        color_mu = logits[offset + 8 : offset + 11]
        color_log_std = logits[offset + 11 : offset + 14]
        return action_logits, rect_mu, rect_log_std, color_mu, color_log_std

    def sample_action(self, action_logits: torch.Tensor) -> str:
        probs = torch.softmax(action_logits, dim=0)
        dist = torch.distributions.Categorical(probs=probs)
        idx = int(dist.sample().item())
        return self.actions[idx]

    def _sample_gaussian(self, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
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
        history_features: Optional[List[torch.Tensor]] = None,
    ) -> dict:
        logits, feats = self.forward(
            reference,
            state,
            output,
            history_features=history_features,
            return_features=True,
        )
        action_logits, rect_mu, rect_log_std, color_mu, color_log_std = self._split_outputs(logits)
        rect_coords, rect_tensor = self.sample_rectangle(rect_mu, rect_log_std)
        color_rgb, color_tensor = self.sample_color(color_mu, color_log_std)
        history_snapshot = None
        if history_features:
            history_snapshot = [
                logit.clone()
                for logit in history_features[-self.max_context_steps :]
            ]
        return {
            "rect": rect_coords,
            "rect_tensor": rect_tensor,
            "color": color_rgb,
            "color_tensor": color_tensor,
            "action_logits": action_logits,
            "logit_history": history_snapshot,
        }

    def save_checkpoint(self, optimizer: Optional[torch.optim.Optimizer] = None, step: Optional[int] = None, path: Optional[str] = None):
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

    def load_checkpoint(self, optimizer: Optional[torch.optim.Optimizer] = None, path: Optional[str] = None) -> Optional[int]:
        source_path = Path(path) if path else self.checkpoint_path
        if source_path is None or not source_path.exists():
            return None
        checkpoint = torch.load(source_path, map_location=self.policy_head.weight.device)
        self.load_state_dict(checkpoint["model_state"])
        if optimizer is not None and "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.last_checkpoint_step = int(checkpoint.get("step", 0))
        return self.last_checkpoint_step
