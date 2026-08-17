"""ACT policy wrapper: one chunk prediction, plus the features the reversibility head eats.

Two jobs:

1. `predict_chunk(obs)` — turn a gym observation into a `(chunk_size, 14)` action chunk.
   lerobot's `select_action` hides the chunk behind an internal action queue, which is
   exactly the thing this project needs to control, so we call `predict_action_chunk`
   and manage commitment ourselves in `faact.runtime.executor`.

2. Feature extraction — via **forward hooks** on the ACT transformer, so we read the
   policy's internal representation without forking lerobot (v1 forked it and pinned a
   submodule commit that did not exist upstream).

Every hook asserts it fired. A hook that silently produces nothing would hand the
reversibility head a vector of zeros and train it to predict the mean — a failure that
looks like "the features just aren't informative" rather than like a bug.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from faact.utils import resolve_device

# lerobot's canonical batch keys.
STATE_KEY = "observation.state"
IMAGE_KEY_TEMPLATE = "observation.images.{camera}"


class ACTWrapper:
    """Wraps a lerobot `ACTPolicy` for open-loop chunk execution with feature readout."""

    def __init__(
        self,
        checkpoint: str,
        device: str | None = None,
        camera: str = "top",
    ) -> None:
        from lerobot.policies.act.modeling_act import ACTPolicy

        self.device = resolve_device(device)
        self.camera = camera
        self.image_key = IMAGE_KEY_TEMPLATE.format(camera=camera)

        self.policy = ACTPolicy.from_pretrained(checkpoint).to(self.device).eval()
        self.chunk_size = int(self.policy.config.chunk_size)

        expected = set(self.policy.config.input_features)
        if not {self.image_key, STATE_KEY} <= expected:
            raise KeyError(
                f"checkpoint expects inputs {sorted(expected)}, which does not cover "
                f"{self.image_key!r} and {STATE_KEY!r}"
            )

        self._captured: dict[str, torch.Tensor] = {}
        self._register_hooks()

    # -- feature hooks -----------------------------------------------------------------

    def _register_hooks(self) -> None:
        """Hook the transformer encoder and decoder outputs.

        Encoder output summarises "what the policy sees"; decoder output summarises "what
        it intends". Both are (seq, batch, dim) in lerobot's ACT, so we mean-pool over the
        sequence axis to get one fixed-width vector per forward pass.
        """
        model = self.policy.model
        for name in ("encoder", "decoder"):
            module = getattr(model, name, None)
            if module is None:
                raise AttributeError(f"ACT model has no submodule {name!r}; lerobot API changed?")
            module.register_forward_hook(self._make_hook(name))

    def _make_hook(self, name: str):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            # Some blocks return tuples; take the first tensor element.
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"hook {name!r} received a {type(tensor).__name__}, not a Tensor")
            self._captured[name] = tensor.detach()

        return hook

    def _pop_features(self) -> dict[str, np.ndarray]:
        """Drain captured activations into pooled numpy vectors, asserting both fired."""
        missing = {"encoder", "decoder"} - set(self._captured)
        if missing:
            raise RuntimeError(
                f"forward hooks did not fire for {sorted(missing)}. Refusing to emit zeros — "
                "v1's silent placeholders are what this check exists to prevent."
            )

        feats: dict[str, np.ndarray] = {}
        for name, tensor in self._captured.items():
            # ACT emits (seq, batch, dim); mean-pool the sequence, keep batch item 0.
            pooled = tensor.float().mean(dim=0) if tensor.ndim == 3 else tensor.float()
            feats[f"feat_{name}_mean"] = pooled.reshape(-1).cpu().numpy()
        self._captured.clear()
        return feats

    # -- inference ---------------------------------------------------------------------

    def reset(self) -> None:
        """Clear lerobot's internal action queue at the start of an episode."""
        self.policy.reset()

    def _to_batch(self, obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        """gym observation -> lerobot batch: CHW float image in [0,1], plus state."""
        pixels = obs.get("pixels")
        if not isinstance(pixels, dict) or self.camera not in pixels:
            raise KeyError(f"observation has no pixels[{self.camera!r}]")

        img = np.asarray(pixels[self.camera])
        if img.dtype != np.uint8:
            raise TypeError(f"expected uint8 image, got {img.dtype}")
        img_t = torch.from_numpy(img).to(self.device).permute(2, 0, 1).float().div_(255.0)

        state = np.asarray(obs["agent_pos"], dtype=np.float32)
        state_t = torch.from_numpy(state).to(self.device)

        return {self.image_key: img_t.unsqueeze(0), STATE_KEY: state_t.unsqueeze(0)}

    @torch.no_grad()
    def predict_chunk(self, obs: dict[str, Any]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Predict one action chunk and the features describing the state it came from.

        Returns `(chunk[chunk_size, 14], features)`. Actions are already unnormalised by
        lerobot, so they can be passed straight to `env.step`.
        """
        batch = self._to_batch(obs)
        # Drop anything left over from a previous call before the forward pass, so a hook
        # that stops firing surfaces as an error rather than as silently stale features.
        self._captured.clear()
        chunk = self.policy.predict_action_chunk(batch)[0].float().cpu().numpy()

        features = self._pop_features()
        # Cheap action-space features. These are informative on their own and act as a
        # sanity floor: if the transformer features add nothing over these, we want to see
        # that in the M4 numbers rather than assume the deep ones are doing the work.
        features["feat_state"] = np.asarray(obs["agent_pos"], dtype=np.float32)
        features["feat_action_first"] = chunk[0].astype(np.float32)
        features["feat_action_prefix_mean_10"] = chunk[:10].mean(axis=0).astype(np.float32)
        features["feat_action_prefix_flat_10"] = chunk[:10].reshape(-1).astype(np.float32)
        return chunk, features


def concat_features(features: dict[str, np.ndarray], keys: list[str] | None = None) -> np.ndarray:
    """Flatten a feature dict into one vector, in a fixed, explicit key order.

    Order is sorted-by-key rather than dict order so a vector built today lines up with a
    model trained yesterday.
    """
    keys = sorted(features) if keys is None else keys
    missing = [k for k in keys if k not in features]
    if missing:
        raise KeyError(f"missing feature keys {missing}")
    return np.concatenate([np.asarray(features[k], dtype=np.float32).reshape(-1) for k in keys])
