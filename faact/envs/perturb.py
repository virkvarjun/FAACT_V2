"""Perturbation suite — the thing v1 never implemented.

v1's `--perturbation_mode` flag was an unused placeholder, so nothing was ever disturbed:
its "failures" were just a weak policy failing, with no recoverable disturbance to detect
and no headroom above a 0% baseline. Everything downstream depends on this file actually
changing the world.

Four kinds, each hooking the episode loop at a different point:

    kind              hook            what it does
    ----------------  --------------  --------------------------------------------------
    object_displace   on_env          teleports the cube mid-episode (state-level)
    grasp_slip        on_action       forces the grippers open for a few steps
    actuation_noise   on_action       additive Gaussian on the commanded joint targets
    occlusion         on_obs          zeroes a rectangular patch in the camera image

`onset_step` is chosen by us, so it is *known ground truth* — that is what makes lead-time
(how early the controller reacts relative to the disturbance) measurable at all.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

KINDS = ("object_displace", "grasp_slip", "actuation_noise", "occlusion")

# Starting magnitudes. These are *calibration targets*, not final values: M2 tunes each one
# until perturbed success lands in the 25-65% band, and the tuned values are written to
# artifacts/perturbation_calibration.json.
DEFAULTS: dict[str, dict[str, float]] = {
    # metres of planar cube displacement (the plan's 2-5 cm), applied on a single step
    "object_displace": {"magnitude": 0.035, "duration": 1},
    # gripper held open; magnitude is unused (the open value comes from the action space)
    "grasp_slip": {"magnitude": 1.0, "duration": 3},
    # std-dev of Gaussian noise added to normalised joint targets, over a 20-step window
    "actuation_noise": {"magnitude": 0.05, "duration": 20},
    # fraction of image *area* blacked out, over a 20-step window
    "occlusion": {"magnitude": 0.15, "duration": 20},
}


@dataclass(frozen=True)
class PerturbationSpec:
    """A fully-determined disturbance. Frozen so it can be persisted verbatim per episode."""

    kind: str
    onset_step: int
    magnitude: float
    duration: int
    seed: int = 0

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown perturbation kind {self.kind!r}; expected one of {KINDS}")
        if self.onset_step < 0:
            raise ValueError(f"onset_step must be >= 0, got {self.onset_step}")
        if self.duration < 1:
            raise ValueError(f"duration must be >= 1, got {self.duration}")

    def is_active(self, t: int) -> bool:
        """Half-open window [onset, onset + duration)."""
        return self.onset_step <= t < self.onset_step + self.duration

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PerturbationSpec":
        return cls(**{k: d[k] for k in ("kind", "onset_step", "magnitude", "duration", "seed")})


def sample_spec(
    kind: str,
    rng: np.random.Generator,
    onset_range: tuple[int, int],
    magnitude: float | None = None,
) -> PerturbationSpec:
    """Draw a spec with a random onset inside `onset_range` (inclusive, exclusive).

    Onset is randomised so the reversibility head cannot learn "trouble always starts at
    step 120"; magnitude defaults to the calibrated value for the kind.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown perturbation kind {kind!r}; expected one of {KINDS}")
    lo, hi = onset_range
    if not lo < hi:
        raise ValueError(f"onset_range must be non-empty, got {onset_range}")
    d = DEFAULTS[kind]
    return PerturbationSpec(
        kind=kind,
        onset_step=int(rng.integers(lo, hi)),
        magnitude=float(d["magnitude"] if magnitude is None else magnitude),
        duration=int(d["duration"]),
        # Own seed so the *same* spec reproduces the same noise/patch/direction.
        seed=int(rng.integers(0, 2**31 - 1)),
    )


class Perturbation:
    """Stateful applier for one spec over one episode.

    Constructed per episode, then `bind()` to the env so every environment-dependent
    constant (action bounds, image shape, cube joint address) is resolved once, up front,
    and loudly. Nothing here silently no-ops: if a hook cannot do its job it raises.
    """

    def __init__(self, spec: PerturbationSpec, camera: str = "top") -> None:
        self.spec = spec
        self.camera = camera
        self._rng = np.random.default_rng(spec.seed)
        self._bound = False
        # Set by bind():
        self._action_low: np.ndarray | None = None
        self._action_high: np.ndarray | None = None
        self._gripper_idx: tuple[int, ...] = ()
        self._patch: tuple[int, int, int, int] | None = None
        self._displacement: np.ndarray | None = None
        # Records whether the one-shot state edit actually happened, for the M2 assertion.
        self.applied_steps: list[int] = []

    # -- setup -------------------------------------------------------------------------

    def bind(self, env: Any, image_hw: tuple[int, int], gripper_idx: tuple[int, ...]) -> "Perturbation":
        """Resolve env-dependent constants and pre-draw all randomness.

        Randomness is drawn *now* rather than at the hook, so the disturbance is a pure
        function of the spec: replaying an episode from a snapshot reproduces it exactly.
        """
        low = np.asarray(env.action_space.low, dtype=np.float64)
        high = np.asarray(env.action_space.high, dtype=np.float64)
        if low.shape != high.shape:
            raise ValueError("action_space low/high shape mismatch")
        self._action_low, self._action_high = low, high

        if not gripper_idx or max(gripper_idx) >= low.size:
            raise ValueError(f"gripper_idx {gripper_idx} out of range for action dim {low.size}")
        self._gripper_idx = tuple(gripper_idx)

        if self.spec.kind == "occlusion":
            self._patch = _sample_patch(image_hw, self.spec.magnitude, self._rng)
        if self.spec.kind == "object_displace":
            # Uniform random planar direction, fixed radius = magnitude.
            theta = self._rng.uniform(0.0, 2.0 * np.pi)
            self._displacement = self.spec.magnitude * np.array([np.cos(theta), np.sin(theta)])

        self._bound = True
        return self

    def _check_bound(self) -> None:
        if not self._bound:
            raise RuntimeError("Perturbation.bind(env, ...) must be called before use")

    # -- hooks -------------------------------------------------------------------------

    def on_env(self, env: Any, t: int) -> bool:
        """State-level hook, called *before* the action at step `t`. Returns True if fired."""
        self._check_bound()
        if self.spec.kind != "object_displace" or not self.spec.is_active(t):
            return False
        # Imported lazily: the action/obs hooks are pure numpy and must stay importable
        # (and unit-testable) without MuJoCo present.
        from faact.envs.state import displace_free_body

        displace_free_body(env, np.asarray(self._displacement))
        self.applied_steps.append(t)
        return True

    def on_action(self, action: np.ndarray, t: int) -> np.ndarray:
        """Action-level hook. Returns a new array; never mutates the caller's action."""
        self._check_bound()
        if not self.spec.is_active(t):
            return action
        a = np.array(action, dtype=np.float64, copy=True)

        if self.spec.kind == "grasp_slip":
            # Fully open both grippers: whatever was being held is dropped.
            for i in self._gripper_idx:
                a[i] = self._action_high[i]
        elif self.spec.kind == "actuation_noise":
            a = a + self._rng.normal(0.0, self.spec.magnitude, size=a.shape)
        else:
            return action  # env/obs-level kind — nothing to do here

        self.applied_steps.append(t)
        return np.clip(a, self._action_low, self._action_high).astype(action.dtype, copy=False)

    def on_obs(self, obs: dict[str, Any], t: int) -> dict[str, Any]:
        """Observation-level hook. Shallow-copies down to the image it edits."""
        self._check_bound()
        if self.spec.kind != "occlusion" or not self.spec.is_active(t):
            return obs
        if self._patch is None:
            raise RuntimeError("occlusion patch was not resolved; bind() did not run")

        pixels = obs.get("pixels")
        if not isinstance(pixels, dict) or self.camera not in pixels:
            raise KeyError(
                f"expected obs['pixels'][{self.camera!r}]; got keys "
                f"{sorted(pixels) if isinstance(pixels, dict) else type(pixels)}"
            )
        img = np.array(pixels[self.camera], copy=True)
        y0, y1, x0, x1 = self._patch
        img[y0:y1, x0:x1] = 0
        self.applied_steps.append(t)
        return {**obs, "pixels": {**pixels, self.camera: img}}

    # -- reporting ---------------------------------------------------------------------

    @property
    def fired(self) -> bool:
        """Did this perturbation ever actually touch the episode?

        The eval runner asserts this at the end of every perturbed episode — a spec whose
        onset lands past the episode horizon is a silent no-op, which is exactly the class
        of bug that produced v1's 0/20.
        """
        return bool(self.applied_steps)


def _sample_patch(
    image_hw: tuple[int, int], area_fraction: float, rng: np.random.Generator
) -> tuple[int, int, int, int]:
    """Pick a square patch covering `area_fraction` of the image. Returns (y0, y1, x0, x1)."""
    if not 0.0 < area_fraction <= 1.0:
        raise ValueError(f"area_fraction must be in (0, 1], got {area_fraction}")
    h, w = image_hw
    side_h = max(1, int(round(h * np.sqrt(area_fraction))))
    side_w = max(1, int(round(w * np.sqrt(area_fraction))))
    y0 = int(rng.integers(0, max(1, h - side_h + 1)))
    x0 = int(rng.integers(0, max(1, w - side_w + 1)))
    return y0, y0 + side_h, x0, x0 + side_w
