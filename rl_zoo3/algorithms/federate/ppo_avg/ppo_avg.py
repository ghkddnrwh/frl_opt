from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch as th
from stable_baselines3.ppo import PPO

from rl_zoo3.algorithms.federate.common.federated_algorithm import (
    FederatedAlgorithmMixin,
    FederatedModules,
    FederatedPayload,
)

VecNormalizeState = dict[str, Any]
RunningMeanStdState = dict[str, Any]


class PPOAvg(FederatedAlgorithmMixin, PPO):
    """PPO with a FedAvg-style synchronization rule.

    - local client optimization is delegated to SB3's original PPO implementation,
    - clients upload synchronizable policy parameters and VecNormalize statistics,
    - the server averages those parameters/statistics,
    - the averaged payload is pushed back to all clients.

    Critic synchronization is configurable via ``critic_sync_mode``:
      - ``fedavg``: synchronize the full SB3 policy, including actor and critic
        parameters. This preserves the original PPOAvg behavior.
      - ``local``: synchronize only actor / non-critic parameters. Critic/value
        parameters remain client-local and are not uploaded, averaged, or
        overwritten by the server.

    Important VecNormalize detail:
    After every communication round, all clients receive the same global
    VecNormalize state. Therefore, the next upload from each client contains the
    same old global RunningMeanStd history plus that client's new local samples.
    If we simply pool the full uploaded counts, the old history is counted once
    per client every round and ``count`` grows exponentially. To avoid that, the
    server stores the last global VecNormalize state and merges only the newly
    added client increments.
    """

    federated_modules: tuple[str, ...] = ("policy",)
    federated_manager_keys: tuple[str, ...] = (
        "num_clients",
        "local_steps",
        "server_update_weight",
        "log_wandb",
        "critic_sync_mode",
        "vecnormalize_sync_mode",
    )
    valid_critic_sync_modes: tuple[str, ...] = ("fedavg", "local")
    valid_vecnormalize_sync_modes: tuple[str, ...] = ("none", "obs", "reward", "obs_reward")

    # Last VecNormalize state produced by the server. This lets us subtract the
    # shared pre-round normalizer history from each client upload and aggregate
    # only the new per-client samples.
    _last_global_vecnormalize_state: VecNormalizeState | None = None

    def __init__(self, *args, **kwargs):
        # Default to the original PPOAvg behavior: average the full SB3 policy,
        # including actor and critic. Set ``critic_sync_mode="local"`` to keep
        # value/critic parameters client-local and synchronize only actor-like
        # parameters.
        self.critic_sync_mode = self._normalize_critic_sync_mode(kwargs.pop("critic_sync_mode", "fedavg"))
        self.vecnormalize_sync_mode = self._normalize_vecnormalize_sync_mode(
            kwargs.pop("vecnormalize_sync_mode", "obs_reward")
        )
        self._federated_progress_lock: float | None = None

        for key in self.federated_manager_keys:
            kwargs.pop(key, None)
        super().__init__(*args, **kwargs)
        self._last_federated_metrics: dict[str, float] = {}

    @classmethod
    def reset_federated_state(cls) -> None:
        cls._last_global_vecnormalize_state = None

    @classmethod
    def uses_federated_client_n_envs(cls) -> bool:
        return True

    def prepare_federated_training(self, clients: Sequence[FederatedAlgorithmMixin]) -> None:
        del clients
        if self.vecnormalize_sync_mode == "none":
            type(self)._last_global_vecnormalize_state = None
            return
        type(self)._last_global_vecnormalize_state = type(self)._clone_state(
            self._get_filtered_vecnormalize_state()
        )

    def _update_current_progress_remaining(self, num_timesteps: int, total_timesteps: int) -> None:
        if self._federated_progress_lock is None:
            super()._update_current_progress_remaining(num_timesteps, total_timesteps)
            return
        self._current_progress_remaining = self._federated_progress_lock

    @classmethod
    def _normalize_critic_sync_mode(cls, mode: str) -> str:
        """Normalize critic synchronization mode aliases.

        Modes:
          - fedavg: average and broadcast the full SB3 policy state_dict.
          - local: average and broadcast only actor/non-critic entries; keep
            value/critic entries local to each client.
        """
        normalized = str(mode).strip().lower().replace("-", "_")
        aliases = {
            "avg": "fedavg",
            "average": "fedavg",
            "global": "fedavg",
            "server": "fedavg",
            "sync": "fedavg",
            "synchronized": "fedavg",
            "none": "local",
            "no_sync": "local",
            "local_only": "local",
            "client_local": "local",
            "private": "local",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in cls.valid_critic_sync_modes:
            raise ValueError(
                f"Unsupported critic_sync_mode={mode!r}. "
                f"Choose one of {cls.valid_critic_sync_modes}."
            )
        return normalized

    @classmethod
    def _normalize_vecnormalize_sync_mode(cls, mode: str) -> str:
        """Normalize VecNormalize synchronization mode aliases.

        Modes:
          - none: do not upload, aggregate, or apply VecNormalize statistics.
          - obs: synchronize only observation RunningMeanStd statistics.
          - reward: synchronize only reward/return RunningMeanStd statistics.
          - obs_reward: synchronize both observation and reward statistics.
        """
        normalized = str(mode).strip().lower().replace("-", "_")
        aliases = {
            "false": "none",
            "off": "none",
            "no": "none",
            "0": "none",
            "disable": "none",
            "disabled": "none",
            "true": "obs_reward",
            "on": "obs_reward",
            "yes": "obs_reward",
            "1": "obs_reward",
            "all": "obs_reward",
            "full": "obs_reward",
            "both": "obs_reward",
            "obs_only": "obs",
            "observation": "obs",
            "observations": "obs",
            "state": "obs",
            "states": "obs",
            "ret": "reward",
            "return": "reward",
            "returns": "reward",
            "reward_only": "reward",
            "rewards": "reward",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in cls.valid_vecnormalize_sync_modes:
            raise ValueError(
                f"Unsupported vecnormalize_sync_mode={mode!r}. "
                f"Choose one of {cls.valid_vecnormalize_sync_modes}."
            )
        return normalized

    @staticmethod
    def _is_critic_key(key: str) -> bool:
        """Return True for SB3 value/critic parameters.

        SB3 stores actor and critic inside one ActorCriticPolicy. For MlpPolicy,
        the critic keys are typically ``mlp_extractor.value_net.*`` and
        ``value_net.*``. For policies with separate value feature extractors,
        ``vf_features_extractor.*`` is also critic-local.
        """
        return (
            key.startswith("value_net.")
            or key.startswith("mlp_extractor.value_net.")
            or key.startswith("vf_features_extractor.")
        )

    def _module_state_keys(self, module_name: str) -> tuple[str, ...]:
        """Return state_dict keys that should be synchronized for a module."""
        module = getattr(self, module_name)
        state = module.state_dict()
        if self.critic_sync_mode == "fedavg":
            return tuple(state.keys())

        # Local-critic mode: keep value/critic parameters private to each client.
        # All non-critic entries are synchronized. This includes actor heads and
        # shared feature-extractor entries. If a custom policy uses a feature
        # extractor shared by actor and critic, that shared representation is
        # necessarily synchronized because the actor depends on it.
        return tuple(key for key in state.keys() if not self._is_critic_key(key))

    def federated_local_update(self, local_steps: int, **kwargs) -> None:
        """Run one local PPO stage on this client's vectorized environment.

        ``local_steps`` is interpreted as environment interactions. As in SB3
        PPO, the realized amount of data is quantized by ``n_steps * n_envs``
        because rollouts are collected in fixed-size batches.
        """
        self._federated_progress_lock = float(self._current_progress_remaining)
        try:
            self.learn(total_timesteps=local_steps, **kwargs)
        finally:
            self._federated_progress_lock = None

    def _get_module_states(self) -> FederatedModules:
        """Return the policy entries that participate in server aggregation.

        In ``critic_sync_mode="fedavg"`` this is the full policy state_dict,
        matching the original PPOAvg implementation. In
        ``critic_sync_mode="local"`` this excludes value/critic entries so
        local critics are never uploaded to the server.
        """
        module_states: FederatedModules = {}
        for module_name in self.federated_modules:
            module = getattr(self, module_name)
            state = module.state_dict()
            keys = self._module_state_keys(module_name)
            module_states[module_name] = OrderedDict(
                (key, state[key].detach().cpu().clone()) for key in keys
            )
        return module_states

    def _set_module_states(self, module_states: FederatedModules) -> None:
        """Apply a global federated payload while preserving local-only entries.

        For ``critic_sync_mode="local"``, the incoming payload contains only
        actor/non-critic keys. We merge those keys into the current full
        state_dict and leave critic/value keys untouched.
        """
        device = self.device
        for module_name in self.federated_modules:
            if module_name not in module_states:
                raise KeyError(f"Missing module {module_name!r} in federated payload.")

            module = getattr(self, module_name)
            current_state = module.state_dict()
            incoming_state = module_states[module_name]
            expected_keys = set(self._module_state_keys(module_name))
            incoming_keys = set(incoming_state.keys())

            unexpected = incoming_keys - set(current_state.keys())
            if unexpected:
                raise KeyError(f"Incoming payload contains unknown keys for {module_name}: {sorted(unexpected)}")

            missing = expected_keys - incoming_keys
            if missing:
                raise KeyError(f"Incoming payload is missing synchronized keys for {module_name}: {sorted(missing)}")

            for key, value in incoming_state.items():
                current_state[key] = value.to(device)
            module.load_state_dict(current_state, strict=True)

    @staticmethod
    def _get_rms_state(rms: Any) -> RunningMeanStdState | dict[str, RunningMeanStdState]:
        """Serialize SB3 RunningMeanStd, including dict-observation variants."""
        if isinstance(rms, Mapping):
            return {key: PPOAvg._get_rms_state(value) for key, value in rms.items()}

        return {
            "mean": np.asarray(rms.mean, dtype=np.float64).copy(),
            "var": np.asarray(rms.var, dtype=np.float64).copy(),
            "count": float(rms.count),
        }

    @staticmethod
    def _set_rms_state(rms: Any, state: RunningMeanStdState | dict[str, RunningMeanStdState]) -> None:
        """Restore SB3 RunningMeanStd, including dict-observation variants."""
        if isinstance(rms, Mapping):
            for key, value in state.items():
                if key in rms:
                    PPOAvg._set_rms_state(rms[key], value)
            return

        mean = np.asarray(state["mean"], dtype=np.float64)
        var = np.asarray(state["var"], dtype=np.float64)
        count = float(state["count"])

        # Do not inject invalid normalizer statistics into the env.
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(var)) or not np.isfinite(count):
            raise ValueError(f"Invalid VecNormalize RMS state: count={count}, mean={mean}, var={var}")

        rms.mean = mean.copy()
        rms.var = np.maximum(var, 1e-12).copy()
        rms.count = max(count, 1e-4)

    def _get_vecnormalize_state(self) -> VecNormalizeState | None:
        """Serialize VecNormalize statistics from this model's env, if present."""
        vecnormalize = self.get_vec_normalize_env()
        if vecnormalize is None:
            return None

        state: VecNormalizeState = {
            "norm_obs": bool(vecnormalize.norm_obs),
            "norm_reward": bool(vecnormalize.norm_reward),
            "clip_obs": float(vecnormalize.clip_obs),
            "clip_reward": float(vecnormalize.clip_reward),
            "gamma": float(vecnormalize.gamma),
            "epsilon": float(vecnormalize.epsilon),
            "training": bool(vecnormalize.training),
            "obs_rms": None,
            "ret_rms": None,
        }

        if getattr(vecnormalize, "obs_rms", None) is not None:
            state["obs_rms"] = self._get_rms_state(vecnormalize.obs_rms)
        if getattr(vecnormalize, "ret_rms", None) is not None:
            state["ret_rms"] = self._get_rms_state(vecnormalize.ret_rms)

        return state

    def _get_filtered_vecnormalize_state(self) -> VecNormalizeState | None:
        """Serialize only the VecNormalize statistics selected by sync mode."""
        if self.vecnormalize_sync_mode == "none":
            return None

        state = self._get_vecnormalize_state()
        if state is None:
            return None

        if self.vecnormalize_sync_mode == "obs":
            state["ret_rms"] = None
        elif self.vecnormalize_sync_mode == "reward":
            state["obs_rms"] = None

        return state

    def _set_vecnormalize_state(self, state: VecNormalizeState | None) -> None:
        """Apply VecNormalize statistics to this model's env, if present."""
        if state is None:
            return

        vecnormalize = self.get_vec_normalize_env()
        if vecnormalize is None:
            return

        current_original_obs = None
        if self._last_obs is not None:
            current_original_obs = vecnormalize.get_original_obs()

        # Keep the wrapper configuration aligned across server and clients.
        for attr in ("norm_obs", "norm_reward", "clip_obs", "clip_reward", "gamma", "epsilon", "training"):
            if attr in state:
                setattr(vecnormalize, attr, state[attr])

        if state.get("obs_rms") is not None and getattr(vecnormalize, "obs_rms", None) is not None:
            self._set_rms_state(vecnormalize.obs_rms, state["obs_rms"])
        if state.get("ret_rms") is not None and getattr(vecnormalize, "ret_rms", None) is not None:
            self._set_rms_state(vecnormalize.ret_rms, state["ret_rms"])

        if current_original_obs is not None:
            self._last_obs = vecnormalize.normalize_obs(current_original_obs)
            self._last_original_obs = current_original_obs

    def _reset_after_parameter_sync(self) -> None:
        """Drop PPO state that is stale after replacing policy/normalizer state."""
        if hasattr(self.policy, "optimizer"):
            self.policy.optimizer.state.clear()

    @staticmethod
    def _clone_state(state: Any) -> Any:
        if state is None:
            return None
        if isinstance(state, np.ndarray):
            return state.copy()
        if isinstance(state, Mapping):
            return {key: PPOAvg._clone_state(value) for key, value in state.items()}
        return state

    @staticmethod
    def _module_l2_norm(modules: FederatedModules) -> float:
        total = 0.0
        for module_state in modules.values():
            for value in module_state.values():
                if th.is_floating_point(value):
                    tensor = value.detach().to(dtype=th.float64)
                    total += float(th.sum(tensor * tensor).cpu().item())
        return float(np.sqrt(total))

    @staticmethod
    def _module_delta_l2_norm(after: FederatedModules, before: FederatedModules) -> float:
        total = 0.0
        for module_name, after_state in after.items():
            if module_name not in before:
                continue
            for key, after_value in after_state.items():
                if key not in before[module_name] or not th.is_floating_point(after_value):
                    continue
                delta = after_value.detach().to(dtype=th.float64) - before[module_name][key].detach().to(dtype=th.float64)
                total += float(th.sum(delta * delta).cpu().item())
        return float(np.sqrt(total))

    @staticmethod
    def _is_single_rms_state(state: Any) -> bool:
        return isinstance(state, Mapping) and {"mean", "var", "count"}.issubset(state.keys())

    @staticmethod
    def _rms_is_finite(state: RunningMeanStdState) -> bool:
        return (
            np.isfinite(float(state["count"]))
            and np.all(np.isfinite(np.asarray(state["mean"], dtype=np.float64)))
            and np.all(np.isfinite(np.asarray(state["var"], dtype=np.float64)))
        )

    @classmethod
    def _merge_single_rms_states(
        cls,
        states: Sequence[RunningMeanStdState],
    ) -> RunningMeanStdState:
        """Merge independent RunningMeanStd states exactly via M2 statistics."""
        valid_states = [state for state in states if cls._rms_is_finite(state) and float(state["count"]) > 0.0]
        if len(valid_states) == 0:
            reference = states[0]
            return {
                "mean": np.asarray(reference["mean"], dtype=np.float64).copy(),
                "var": np.maximum(np.asarray(reference["var"], dtype=np.float64), 1e-12).copy(),
                "count": max(float(reference["count"]), 1e-4),
            }

        mean = np.asarray(valid_states[0]["mean"], dtype=np.float64).copy()
        var = np.maximum(np.asarray(valid_states[0]["var"], dtype=np.float64), 0.0).copy()
        count = float(valid_states[0]["count"])
        m2 = var * count

        for state in valid_states[1:]:
            other_count = float(state["count"])
            other_mean = np.asarray(state["mean"], dtype=np.float64)
            other_var = np.maximum(np.asarray(state["var"], dtype=np.float64), 0.0)
            other_m2 = other_var * other_count

            total = count + other_count
            if total <= 0.0:
                continue
            delta = other_mean - mean
            mean = mean + delta * other_count / total
            m2 = m2 + other_m2 + np.square(delta) * count * other_count / total
            count = total

        var = np.maximum(m2 / max(count, 1e-12), 1e-12)
        return {"mean": mean, "var": var, "count": max(count, 1e-4)}

    @classmethod
    def _subtract_single_rms_state(
        cls,
        final_state: RunningMeanStdState,
        base_state: RunningMeanStdState,
    ) -> RunningMeanStdState | None:
        """Recover the incremental samples D from final_state = merge(base_state, D).

        Returns None when subtraction is not numerically or structurally valid.
        """
        if not cls._rms_is_finite(final_state) or not cls._rms_is_finite(base_state):
            return None

        final_count = float(final_state["count"])
        base_count = float(base_state["count"])
        inc_count = final_count - base_count

        # If the client did not start from the stored global state, subtraction is invalid.
        if inc_count <= 1e-8:
            return None

        final_mean = np.asarray(final_state["mean"], dtype=np.float64)
        base_mean = np.asarray(base_state["mean"], dtype=np.float64)
        final_var = np.maximum(np.asarray(final_state["var"], dtype=np.float64), 0.0)
        base_var = np.maximum(np.asarray(base_state["var"], dtype=np.float64), 0.0)

        if final_mean.shape != base_mean.shape or final_var.shape != base_var.shape:
            return None

        # From: final_count * final_mean = base_count * base_mean + inc_count * inc_mean
        inc_mean = (final_count * final_mean - base_count * base_mean) / inc_count

        # Merge formula:
        # M2_final = M2_base + M2_inc + (inc_mean - base_mean)^2 * base_count * inc_count / final_count
        m2_final = final_var * final_count
        m2_base = base_var * base_count
        correction = np.square(inc_mean - base_mean) * base_count * inc_count / final_count
        m2_inc = m2_final - m2_base - correction

        # Tiny negative values can occur from floating point cancellation.
        m2_inc = np.maximum(m2_inc, 0.0)
        inc_var = np.maximum(m2_inc / inc_count, 1e-12)

        if not np.all(np.isfinite(inc_mean)) or not np.all(np.isfinite(inc_var)):
            return None

        return {"mean": inc_mean, "var": inc_var, "count": inc_count}

    @classmethod
    def _aggregate_rms_states(
        cls,
        states: Sequence[RunningMeanStdState | dict[str, RunningMeanStdState]],
        base_state: RunningMeanStdState | dict[str, RunningMeanStdState] | None = None,
    ) -> RunningMeanStdState | dict[str, RunningMeanStdState]:
        """Aggregate RunningMeanStd states without double-counting shared history.

        If base_state is provided, each client upload is interpreted as
        merge(base_state, local_increment), and only local_increment is merged
        into the new global state. If subtraction fails, fall back to pooling the
        full states once.
        """
        if len(states) == 0:
            raise ValueError("At least one RunningMeanStd state is required.")

        reference = states[0]
        if not cls._is_single_rms_state(reference):
            base_mapping = base_state if isinstance(base_state, Mapping) else None
            return {
                key: cls._aggregate_rms_states(
                    [state[key] for state in states],
                    base_mapping.get(key) if base_mapping is not None and key in base_mapping else None,
                )
                for key in reference.keys()
            }

        if base_state is not None and cls._is_single_rms_state(base_state):
            increments = [cls._subtract_single_rms_state(state, base_state) for state in states]
            if all(increment is not None for increment in increments):
                merged_increments = cls._merge_single_rms_states(increments)  # type: ignore[arg-type]
                # Correct global state = old global history + union of new client samples.
                return cls._merge_single_rms_states([base_state, merged_increments])  # type: ignore[list-item]

        # First round or incompatible state: pool the full uploaded states once.
        return cls._merge_single_rms_states(states)  # type: ignore[arg-type]

    @classmethod
    def average_vecnormalize_states(
        cls,
        states: Sequence[VecNormalizeState | None],
    ) -> VecNormalizeState | None:
        """Aggregate VecNormalize obs_rms/ret_rms states from clients."""
        valid_states = [state for state in states if state is not None]
        if len(valid_states) == 0:
            cls._last_global_vecnormalize_state = None
            return None

        reference = valid_states[0]
        previous_global = cls._last_global_vecnormalize_state

        averaged: VecNormalizeState = {
            "norm_obs": bool(reference["norm_obs"]),
            "norm_reward": bool(reference["norm_reward"]),
            "clip_obs": float(reference["clip_obs"]),
            "clip_reward": float(reference["clip_reward"]),
            "gamma": float(reference["gamma"]),
            "epsilon": float(reference["epsilon"]),
            "training": bool(reference["training"]),
            "obs_rms": None,
            "ret_rms": None,
        }

        if reference.get("obs_rms") is not None:
            averaged["obs_rms"] = cls._aggregate_rms_states(
                [state["obs_rms"] for state in valid_states if state.get("obs_rms") is not None],
                previous_global.get("obs_rms") if previous_global is not None else None,
            )
        if reference.get("ret_rms") is not None:
            averaged["ret_rms"] = cls._aggregate_rms_states(
                [state["ret_rms"] for state in valid_states if state.get("ret_rms") is not None],
                previous_global.get("ret_rms") if previous_global is not None else None,
            )

        cls._last_global_vecnormalize_state = cls._clone_state(averaged)
        return averaged

    def get_upload_payload(self) -> FederatedPayload:
        return {
            "modules": self._get_module_states(),
            "vecnormalize": self._get_filtered_vecnormalize_state(),
            "critic_sync_mode": self.critic_sync_mode,
            "vecnormalize_sync_mode": self.vecnormalize_sync_mode,
            "meta": {
                "client_weight": self.get_client_weight(),
                "num_timesteps": int(self.num_timesteps),
            },
        }

    @classmethod
    def aggregate_uploads(
        cls,
        uploads: Sequence[FederatedPayload],
        weights: Sequence[float] | None = None,
    ) -> FederatedPayload:
        if len(uploads) == 0:
            raise ValueError("At least one upload is required for federated aggregation.")

        critic_modes = {
            cls._normalize_critic_sync_mode(str(upload.get("critic_sync_mode", "fedavg")))
            for upload in uploads
        }
        if len(critic_modes) != 1:
            raise ValueError(f"Mixed critic_sync_mode values are not supported in one aggregation: {critic_modes}")
        critic_sync_mode = next(iter(critic_modes))

        vecnormalize_modes = {
            cls._normalize_vecnormalize_sync_mode(str(upload.get("vecnormalize_sync_mode", "obs_reward")))
            for upload in uploads
        }
        if len(vecnormalize_modes) != 1:
            raise ValueError(
                f"Mixed vecnormalize_sync_mode values are not supported in one aggregation: {vecnormalize_modes}"
            )
        vecnormalize_sync_mode = next(iter(vecnormalize_modes))

        module_states = [upload["modules"] for upload in uploads]
        normalized_weights = cls.normalize_weights(len(uploads), weights)
        aggregated_modules = cls.average_module_states(module_states, weights=weights)
        if vecnormalize_sync_mode == "none":
            aggregated_vecnormalize = None
            cls._last_global_vecnormalize_state = None
        else:
            aggregated_vecnormalize = cls.average_vecnormalize_states(
                [upload.get("vecnormalize") for upload in uploads]
            )
        return {
            "modules": aggregated_modules,
            "vecnormalize": aggregated_vecnormalize,
            "critic_sync_mode": critic_sync_mode,
            "vecnormalize_sync_mode": vecnormalize_sync_mode,
            "meta": {
                "num_clients": len(uploads),
                "critic_sync_mode": critic_sync_mode,
                "vecnormalize_sync_mode": vecnormalize_sync_mode,
                "client_weights": normalized_weights,
                "client_timesteps": [
                    int(upload.get("meta", {}).get("num_timesteps", 0)) for upload in uploads
                ],
            },
        }

    def apply_global_payload(self, payload: FederatedPayload, mix_weight: float = 1.0) -> None:
        if not (0.0 < mix_weight <= 1.0):
            raise ValueError("mix_weight must be in (0, 1].")

        payload_critic_mode = self._normalize_critic_sync_mode(
            str(payload.get("critic_sync_mode", payload.get("meta", {}).get("critic_sync_mode", self.critic_sync_mode)))
        )
        if payload_critic_mode != self.critic_sync_mode:
            raise ValueError(
                f"Client critic_sync_mode={self.critic_sync_mode!r} does not match "
                f"payload critic_sync_mode={payload_critic_mode!r}."
            )

        payload_vecnormalize_mode = self._normalize_vecnormalize_sync_mode(
            str(
                payload.get(
                    "vecnormalize_sync_mode",
                    payload.get("meta", {}).get("vecnormalize_sync_mode", self.vecnormalize_sync_mode),
                )
            )
        )
        if payload_vecnormalize_mode != self.vecnormalize_sync_mode:
            raise ValueError(
                f"Client vecnormalize_sync_mode={self.vecnormalize_sync_mode!r} does not match "
                f"payload vecnormalize_sync_mode={payload_vecnormalize_mode!r}."
            )

        old_modules = self._get_module_states()
        incoming_modules = payload["modules"]
        if mix_weight < 1.0:
            mixed_modules: FederatedModules = {}
            for module_name, module_state in incoming_modules.items():
                mixed_modules[module_name] = OrderedDict()
                for key, value in module_state.items():
                    current_value = old_modules[module_name][key]
                    if th.is_floating_point(value):
                        mixed_modules[module_name][key] = (
                            mix_weight * value + (1.0 - mix_weight) * current_value.to(value.dtype)
                        )
                    else:
                        mixed_modules[module_name][key] = value.clone()
            incoming_modules = mixed_modules

        self._set_module_states(incoming_modules)
        new_modules = self._get_module_states()

        # VecNormalize statistics are environment statistics, not gradient updates.
        # Apply the aggregated global statistics directly when synchronization is enabled.
        if self.vecnormalize_sync_mode != "none":
            self._set_vecnormalize_state(payload.get("vecnormalize"))
        self._reset_after_parameter_sync()

        meta = payload.get("meta", {})
        client_weights = np.asarray(meta.get("client_weights", []), dtype=np.float64)
        client_timesteps = np.asarray(meta.get("client_timesteps", []), dtype=np.float64)
        metrics: dict[str, float] = {
            "server/num_clients": float(meta.get("num_clients", 0)),
            "server/ppo_avg/model_norm": self._module_l2_norm(new_modules),
            "server/ppo_avg/model_delta_norm": self._module_delta_l2_norm(new_modules, old_modules),
            "server/ppo_avg/critic_local": float(self.critic_sync_mode == "local"),
            "server/ppo_avg/critic_fedavg": float(self.critic_sync_mode == "fedavg"),
            "server/ppo_avg/vecnormalize_none": float(self.vecnormalize_sync_mode == "none"),
            "server/ppo_avg/vecnormalize_obs": float(self.vecnormalize_sync_mode == "obs"),
            "server/ppo_avg/vecnormalize_reward": float(self.vecnormalize_sync_mode == "reward"),
            "server/ppo_avg/vecnormalize_obs_reward": float(self.vecnormalize_sync_mode == "obs_reward"),
        }
        if client_weights.size > 0:
            metrics.update(
                {
                    "server/client_weight_mean": float(np.mean(client_weights)),
                    "server/client_weight_min": float(np.min(client_weights)),
                    "server/client_weight_max": float(np.max(client_weights)),
                }
            )
        if client_timesteps.size > 0:
            metrics.update(
                {
                    "server/client_timesteps_mean": float(np.mean(client_timesteps)),
                    "server/client_timesteps_min": float(np.min(client_timesteps)),
                    "server/client_timesteps_max": float(np.max(client_timesteps)),
                }
            )
        self._last_federated_metrics = metrics

    def get_client_weight(self) -> float:
        return 1.0



# # from __future__ import annotations

# # from collections import OrderedDict
# # from collections.abc import Sequence

# # import torch as th
# # from stable_baselines3.ppo import PPO

# # from rl_zoo3.algorithms.federate.buff.buff.federated_algorithm import (
# #     FederatedAlgorithmMixin,
# #     FederatedModules,
# #     FederatedPayload,
# # )


# # class PPOAvg(FederatedAlgorithmMixin, PPO):
# #     """PPO with a FedAvg-style synchronization rule.

# #     - local client optimization is delegated to SB3's original PPO implementation,
# #     - clients upload policy parameters,
# #     - the server averages those parameters,
# #     - the averaged parameters are pushed back to all clients.

# #     For PPO, synchronizing ``policy`` is enough because it contains both the
# #     actor and critic parameters used by the algorithm.
# #     """

# #     federated_modules: tuple[str, ...] = ("policy",)
# #     federated_manager_keys: tuple[str, ...] = ("num_clients", "local_steps", "server_update_weight")

# #     def __init__(self, *args, **kwargs):
# #         for key in self.federated_manager_keys:
# #             kwargs.pop(key, None)
# #         super().__init__(*args, **kwargs)

# #     def federated_local_update(self, local_steps: int, **kwargs) -> None:
# #         self.learn(total_timesteps=local_steps, **kwargs)

# #     def _get_module_states(self) -> FederatedModules:
# #         module_states: FederatedModules = {}
# #         for module_name in self.federated_modules:
# #             module = getattr(self, module_name)
# #             module_states[module_name] = OrderedDict(
# #                 (key, value.detach().cpu().clone()) for key, value in module.state_dict().items()
# #             )
# #         return module_states

# #     def _set_module_states(self, module_states: FederatedModules) -> None:
# #         device = self.device
# #         for module_name in self.federated_modules:
# #             module = getattr(self, module_name)
# #             state_dict = OrderedDict((key, value.to(device)) for key, value in module_states[module_name].items())
# #             module.load_state_dict(state_dict)

# #     def get_upload_payload(self) -> FederatedPayload:
# #         return {
# #             "modules": self._get_module_states(),
# #             "meta": {
# #                 "client_weight": self.get_client_weight(),
# #                 "num_timesteps": int(self.num_timesteps),
# #             },
# #         }

# #     @classmethod
# #     def aggregate_uploads(
# #         cls,
# #         uploads: Sequence[FederatedPayload],
# #         weights: Sequence[float] | None = None,
# #     ) -> FederatedPayload:
# #         if len(uploads) == 0:
# #             raise ValueError("At least one upload is required for federated aggregation.")

# #         module_states = [upload["modules"] for upload in uploads]
# #         aggregated_modules = cls.average_module_states(module_states, weights=weights)
# #         return {
# #             "modules": aggregated_modules,
# #             "meta": {
# #                 "num_clients": len(uploads),
# #             },
# #         }

# #     def apply_global_payload(self, payload: FederatedPayload, mix_weight: float = 1.0) -> None:
# #         if not (0.0 < mix_weight <= 1.0):
# #             raise ValueError("mix_weight must be in (0, 1].")

# #         incoming_modules = payload["modules"]
# #         if mix_weight < 1.0:
# #             current_modules = self._get_module_states()
# #             mixed_modules: FederatedModules = {}
# #             for module_name, module_state in incoming_modules.items():
# #                 mixed_modules[module_name] = OrderedDict()
# #                 for key, value in module_state.items():
# #                     current_value = current_modules[module_name][key]
# #                     if th.is_floating_point(value):
# #                         mixed_modules[module_name][key] = (
# #                             mix_weight * value + (1.0 - mix_weight) * current_value.to(value.dtype)
# #                         )
# #                     else:
# #                         mixed_modules[module_name][key] = value.clone()
# #             incoming_modules = mixed_modules

# #         self._set_module_states(incoming_modules)

# #     def get_client_weight(self) -> float:
# #         return 1.0
# from __future__ import annotations

# from collections import OrderedDict
# from collections.abc import Mapping, Sequence
# from typing import Any

# import numpy as np
# import torch as th
# from stable_baselines3.ppo import PPO

# from rl_zoo3.algorithms.federate.common.federated_algorithm import (
#     FederatedAlgorithmMixin,
#     FederatedModules,
#     FederatedPayload,
# )

# VecNormalizeState = dict[str, Any]
# RunningMeanStdState = dict[str, Any]


# class PPOAvg(FederatedAlgorithmMixin, PPO):
#     """PPO with a FedAvg-style synchronization rule.

#     - local client optimization is delegated to SB3's original PPO implementation,
#     - clients upload synchronizable policy parameters and VecNormalize statistics,
#     - the server averages those parameters/statistics,
#     - the averaged payload is pushed back to all clients.

#     Critic synchronization is configurable via ``critic_sync_mode``:
#       - ``fedavg``: synchronize the full SB3 policy, including actor and critic
#         parameters. This preserves the original PPOAvg behavior.
#       - ``local``: synchronize only actor / non-critic parameters. Critic/value
#         parameters remain client-local and are not uploaded, averaged, or
#         overwritten by the server.

#     Important VecNormalize detail:
#     After every communication round, all clients receive the same global
#     VecNormalize state. Therefore, the next upload from each client contains the
#     same old global RunningMeanStd history plus that client's new local samples.
#     If we simply pool the full uploaded counts, the old history is counted once
#     per client every round and ``count`` grows exponentially. To avoid that, the
#     server stores the last global VecNormalize state and merges only the newly
#     added client increments.
#     """

#     federated_modules: tuple[str, ...] = ("policy",)
#     federated_manager_keys: tuple[str, ...] = (
#         "num_clients",
#         "local_steps",
#         "server_update_weight",
#         "log_wandb",
#         "critic_sync_mode",
#     )
#     valid_critic_sync_modes: tuple[str, ...] = ("fedavg", "local")

#     # Last VecNormalize state produced by the server. This lets us subtract the
#     # shared pre-round normalizer history from each client upload and aggregate
#     # only the new per-client samples.
#     _last_global_vecnormalize_state: VecNormalizeState | None = None

#     def __init__(self, *args, **kwargs):
#         # Default to the original PPOAvg behavior: average the full SB3 policy,
#         # including actor and critic. Set ``critic_sync_mode="local"`` to keep
#         # value/critic parameters client-local and synchronize only actor-like
#         # parameters.
#         self.critic_sync_mode = self._normalize_critic_sync_mode(kwargs.pop("critic_sync_mode", "fedavg"))

#         for key in self.federated_manager_keys:
#             kwargs.pop(key, None)
#         super().__init__(*args, **kwargs)
#         self._last_federated_metrics: dict[str, float] = {}

#     @classmethod
#     def reset_federated_state(cls) -> None:
#         cls._last_global_vecnormalize_state = None

#     @classmethod
#     def uses_federated_client_n_envs(cls) -> bool:
#         return True

#     @classmethod
#     def _normalize_critic_sync_mode(cls, mode: str) -> str:
#         """Normalize critic synchronization mode aliases.

#         Modes:
#           - fedavg: average and broadcast the full SB3 policy state_dict.
#           - local: average and broadcast only actor/non-critic entries; keep
#             value/critic entries local to each client.
#         """
#         normalized = str(mode).strip().lower().replace("-", "_")
#         aliases = {
#             "avg": "fedavg",
#             "average": "fedavg",
#             "global": "fedavg",
#             "server": "fedavg",
#             "sync": "fedavg",
#             "synchronized": "fedavg",
#             "none": "local",
#             "no_sync": "local",
#             "local_only": "local",
#             "client_local": "local",
#             "private": "local",
#         }
#         normalized = aliases.get(normalized, normalized)
#         if normalized not in cls.valid_critic_sync_modes:
#             raise ValueError(
#                 f"Unsupported critic_sync_mode={mode!r}. "
#                 f"Choose one of {cls.valid_critic_sync_modes}."
#             )
#         return normalized

#     @staticmethod
#     def _is_critic_key(key: str) -> bool:
#         """Return True for SB3 value/critic parameters.

#         SB3 stores actor and critic inside one ActorCriticPolicy. For MlpPolicy,
#         the critic keys are typically ``mlp_extractor.value_net.*`` and
#         ``value_net.*``. For policies with separate value feature extractors,
#         ``vf_features_extractor.*`` is also critic-local.
#         """
#         return (
#             key.startswith("value_net.")
#             or key.startswith("mlp_extractor.value_net.")
#             or key.startswith("vf_features_extractor.")
#         )

#     def _module_state_keys(self, module_name: str) -> tuple[str, ...]:
#         """Return state_dict keys that should be synchronized for a module."""
#         module = getattr(self, module_name)
#         state = module.state_dict()
#         if self.critic_sync_mode == "fedavg":
#             return tuple(state.keys())

#         # Local-critic mode: keep value/critic parameters private to each client.
#         # All non-critic entries are synchronized. This includes actor heads and
#         # shared feature-extractor entries. If a custom policy uses a feature
#         # extractor shared by actor and critic, that shared representation is
#         # necessarily synchronized because the actor depends on it.
#         return tuple(key for key in state.keys() if not self._is_critic_key(key))

#     def federated_local_update(self, local_steps: int, **kwargs) -> None:
#         """Run one local PPO stage on this client's vectorized environment.

#         ``local_steps`` is interpreted as environment interactions. As in SB3
#         PPO, the realized amount of data is quantized by ``n_steps * n_envs``
#         because rollouts are collected in fixed-size batches.
#         """
#         self.learn(total_timesteps=local_steps, **kwargs)

#     def _get_module_states(self) -> FederatedModules:
#         """Return the policy entries that participate in server aggregation.

#         In ``critic_sync_mode="fedavg"`` this is the full policy state_dict,
#         matching the original PPOAvg implementation. In
#         ``critic_sync_mode="local"`` this excludes value/critic entries so
#         local critics are never uploaded to the server.
#         """
#         module_states: FederatedModules = {}
#         for module_name in self.federated_modules:
#             module = getattr(self, module_name)
#             state = module.state_dict()
#             keys = self._module_state_keys(module_name)
#             module_states[module_name] = OrderedDict(
#                 (key, state[key].detach().cpu().clone()) for key in keys
#             )
#         return module_states

#     def _set_module_states(self, module_states: FederatedModules) -> None:
#         """Apply a global federated payload while preserving local-only entries.

#         For ``critic_sync_mode="local"``, the incoming payload contains only
#         actor/non-critic keys. We merge those keys into the current full
#         state_dict and leave critic/value keys untouched.
#         """
#         device = self.device
#         for module_name in self.federated_modules:
#             if module_name not in module_states:
#                 raise KeyError(f"Missing module {module_name!r} in federated payload.")

#             module = getattr(self, module_name)
#             current_state = module.state_dict()
#             incoming_state = module_states[module_name]
#             expected_keys = set(self._module_state_keys(module_name))
#             incoming_keys = set(incoming_state.keys())

#             unexpected = incoming_keys - set(current_state.keys())
#             if unexpected:
#                 raise KeyError(f"Incoming payload contains unknown keys for {module_name}: {sorted(unexpected)}")

#             missing = expected_keys - incoming_keys
#             if missing:
#                 raise KeyError(f"Incoming payload is missing synchronized keys for {module_name}: {sorted(missing)}")

#             for key, value in incoming_state.items():
#                 current_state[key] = value.to(device)
#             module.load_state_dict(current_state, strict=True)

#     @staticmethod
#     def _get_rms_state(rms: Any) -> RunningMeanStdState | dict[str, RunningMeanStdState]:
#         """Serialize SB3 RunningMeanStd, including dict-observation variants."""
#         if isinstance(rms, Mapping):
#             return {key: PPOAvg._get_rms_state(value) for key, value in rms.items()}

#         return {
#             "mean": np.asarray(rms.mean, dtype=np.float64).copy(),
#             "var": np.asarray(rms.var, dtype=np.float64).copy(),
#             "count": float(rms.count),
#         }

#     @staticmethod
#     def _set_rms_state(rms: Any, state: RunningMeanStdState | dict[str, RunningMeanStdState]) -> None:
#         """Restore SB3 RunningMeanStd, including dict-observation variants."""
#         if isinstance(rms, Mapping):
#             for key, value in state.items():
#                 if key in rms:
#                     PPOAvg._set_rms_state(rms[key], value)
#             return

#         mean = np.asarray(state["mean"], dtype=np.float64)
#         var = np.asarray(state["var"], dtype=np.float64)
#         count = float(state["count"])

#         # Do not inject invalid normalizer statistics into the env.
#         if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(var)) or not np.isfinite(count):
#             raise ValueError(f"Invalid VecNormalize RMS state: count={count}, mean={mean}, var={var}")

#         rms.mean = mean.copy()
#         rms.var = np.maximum(var, 1e-12).copy()
#         rms.count = max(count, 1e-4)

#     def _get_vecnormalize_state(self) -> VecNormalizeState | None:
#         """Serialize VecNormalize statistics from this model's env, if present."""
#         vecnormalize = self.get_vec_normalize_env()
#         if vecnormalize is None:
#             return None

#         state: VecNormalizeState = {
#             "norm_obs": bool(vecnormalize.norm_obs),
#             "norm_reward": bool(vecnormalize.norm_reward),
#             "clip_obs": float(vecnormalize.clip_obs),
#             "clip_reward": float(vecnormalize.clip_reward),
#             "gamma": float(vecnormalize.gamma),
#             "epsilon": float(vecnormalize.epsilon),
#             "training": bool(vecnormalize.training),
#             "obs_rms": None,
#             "ret_rms": None,
#         }

#         if getattr(vecnormalize, "obs_rms", None) is not None:
#             state["obs_rms"] = self._get_rms_state(vecnormalize.obs_rms)
#         if getattr(vecnormalize, "ret_rms", None) is not None:
#             state["ret_rms"] = self._get_rms_state(vecnormalize.ret_rms)

#         return state

#     def _set_vecnormalize_state(self, state: VecNormalizeState | None) -> None:
#         """Apply VecNormalize statistics to this model's env, if present."""
#         if state is None:
#             return

#         vecnormalize = self.get_vec_normalize_env()
#         if vecnormalize is None:
#             return

#         # Keep the wrapper configuration aligned across server and clients.
#         for attr in ("norm_obs", "norm_reward", "clip_obs", "clip_reward", "gamma", "epsilon", "training"):
#             if attr in state:
#                 setattr(vecnormalize, attr, state[attr])

#         if state.get("obs_rms") is not None and getattr(vecnormalize, "obs_rms", None) is not None:
#             self._set_rms_state(vecnormalize.obs_rms, state["obs_rms"])
#         if state.get("ret_rms") is not None and getattr(vecnormalize, "ret_rms", None) is not None:
#             self._set_rms_state(vecnormalize.ret_rms, state["ret_rms"])

#         # VecNormalize.returns is a transient discounted-return accumulator.
#         # SB3 does not pickle it when saving VecNormalize, so after synchronizing
#         # global ret_rms it is safer to restart the accumulator than to average it.
#         if getattr(vecnormalize, "returns", None) is not None:
#             vecnormalize.returns = np.zeros_like(vecnormalize.returns)

#     def _reset_after_parameter_sync(self) -> None:
#         """Drop PPO state that is stale after replacing policy/normalizer state."""
#         if hasattr(self.policy, "optimizer"):
#             self.policy.optimizer.state.clear()

#         self._last_obs = None
#         self._last_original_obs = None
#         self._last_episode_starts = None

#     @staticmethod
#     def _clone_state(state: Any) -> Any:
#         if state is None:
#             return None
#         if isinstance(state, np.ndarray):
#             return state.copy()
#         if isinstance(state, Mapping):
#             return {key: PPOAvg._clone_state(value) for key, value in state.items()}
#         return state

#     @staticmethod
#     def _module_l2_norm(modules: FederatedModules) -> float:
#         total = 0.0
#         for module_state in modules.values():
#             for value in module_state.values():
#                 if th.is_floating_point(value):
#                     tensor = value.detach().to(dtype=th.float64)
#                     total += float(th.sum(tensor * tensor).cpu().item())
#         return float(np.sqrt(total))

#     @staticmethod
#     def _module_delta_l2_norm(after: FederatedModules, before: FederatedModules) -> float:
#         total = 0.0
#         for module_name, after_state in after.items():
#             if module_name not in before:
#                 continue
#             for key, after_value in after_state.items():
#                 if key not in before[module_name] or not th.is_floating_point(after_value):
#                     continue
#                 delta = after_value.detach().to(dtype=th.float64) - before[module_name][key].detach().to(dtype=th.float64)
#                 total += float(th.sum(delta * delta).cpu().item())
#         return float(np.sqrt(total))

#     @staticmethod
#     def _is_single_rms_state(state: Any) -> bool:
#         return isinstance(state, Mapping) and {"mean", "var", "count"}.issubset(state.keys())

#     @staticmethod
#     def _rms_is_finite(state: RunningMeanStdState) -> bool:
#         return (
#             np.isfinite(float(state["count"]))
#             and np.all(np.isfinite(np.asarray(state["mean"], dtype=np.float64)))
#             and np.all(np.isfinite(np.asarray(state["var"], dtype=np.float64)))
#         )

#     @classmethod
#     def _merge_single_rms_states(
#         cls,
#         states: Sequence[RunningMeanStdState],
#     ) -> RunningMeanStdState:
#         """Merge independent RunningMeanStd states exactly via M2 statistics."""
#         valid_states = [state for state in states if cls._rms_is_finite(state) and float(state["count"]) > 0.0]
#         if len(valid_states) == 0:
#             reference = states[0]
#             return {
#                 "mean": np.asarray(reference["mean"], dtype=np.float64).copy(),
#                 "var": np.maximum(np.asarray(reference["var"], dtype=np.float64), 1e-12).copy(),
#                 "count": max(float(reference["count"]), 1e-4),
#             }

#         mean = np.asarray(valid_states[0]["mean"], dtype=np.float64).copy()
#         var = np.maximum(np.asarray(valid_states[0]["var"], dtype=np.float64), 0.0).copy()
#         count = float(valid_states[0]["count"])
#         m2 = var * count

#         for state in valid_states[1:]:
#             other_count = float(state["count"])
#             other_mean = np.asarray(state["mean"], dtype=np.float64)
#             other_var = np.maximum(np.asarray(state["var"], dtype=np.float64), 0.0)
#             other_m2 = other_var * other_count

#             total = count + other_count
#             if total <= 0.0:
#                 continue
#             delta = other_mean - mean
#             mean = mean + delta * other_count / total
#             m2 = m2 + other_m2 + np.square(delta) * count * other_count / total
#             count = total

#         var = np.maximum(m2 / max(count, 1e-12), 1e-12)
#         return {"mean": mean, "var": var, "count": max(count, 1e-4)}

#     @classmethod
#     def _subtract_single_rms_state(
#         cls,
#         final_state: RunningMeanStdState,
#         base_state: RunningMeanStdState,
#     ) -> RunningMeanStdState | None:
#         """Recover the incremental samples D from final_state = merge(base_state, D).

#         Returns None when subtraction is not numerically or structurally valid.
#         """
#         if not cls._rms_is_finite(final_state) or not cls._rms_is_finite(base_state):
#             return None

#         final_count = float(final_state["count"])
#         base_count = float(base_state["count"])
#         inc_count = final_count - base_count

#         # If the client did not start from the stored global state, subtraction is invalid.
#         if inc_count <= 1e-8:
#             return None

#         final_mean = np.asarray(final_state["mean"], dtype=np.float64)
#         base_mean = np.asarray(base_state["mean"], dtype=np.float64)
#         final_var = np.maximum(np.asarray(final_state["var"], dtype=np.float64), 0.0)
#         base_var = np.maximum(np.asarray(base_state["var"], dtype=np.float64), 0.0)

#         if final_mean.shape != base_mean.shape or final_var.shape != base_var.shape:
#             return None

#         # From: final_count * final_mean = base_count * base_mean + inc_count * inc_mean
#         inc_mean = (final_count * final_mean - base_count * base_mean) / inc_count

#         # Merge formula:
#         # M2_final = M2_base + M2_inc + (inc_mean - base_mean)^2 * base_count * inc_count / final_count
#         m2_final = final_var * final_count
#         m2_base = base_var * base_count
#         correction = np.square(inc_mean - base_mean) * base_count * inc_count / final_count
#         m2_inc = m2_final - m2_base - correction

#         # Tiny negative values can occur from floating point cancellation.
#         m2_inc = np.maximum(m2_inc, 0.0)
#         inc_var = np.maximum(m2_inc / inc_count, 1e-12)

#         if not np.all(np.isfinite(inc_mean)) or not np.all(np.isfinite(inc_var)):
#             return None

#         return {"mean": inc_mean, "var": inc_var, "count": inc_count}

#     @classmethod
#     def _aggregate_rms_states(
#         cls,
#         states: Sequence[RunningMeanStdState | dict[str, RunningMeanStdState]],
#         base_state: RunningMeanStdState | dict[str, RunningMeanStdState] | None = None,
#     ) -> RunningMeanStdState | dict[str, RunningMeanStdState]:
#         """Aggregate RunningMeanStd states without double-counting shared history.

#         If base_state is provided, each client upload is interpreted as
#         merge(base_state, local_increment), and only local_increment is merged
#         into the new global state. If subtraction fails, fall back to pooling the
#         full states once.
#         """
#         if len(states) == 0:
#             raise ValueError("At least one RunningMeanStd state is required.")

#         reference = states[0]
#         if not cls._is_single_rms_state(reference):
#             base_mapping = base_state if isinstance(base_state, Mapping) else None
#             return {
#                 key: cls._aggregate_rms_states(
#                     [state[key] for state in states],
#                     base_mapping.get(key) if base_mapping is not None and key in base_mapping else None,
#                 )
#                 for key in reference.keys()
#             }

#         if base_state is not None and cls._is_single_rms_state(base_state):
#             increments = [cls._subtract_single_rms_state(state, base_state) for state in states]
#             if all(increment is not None for increment in increments):
#                 merged_increments = cls._merge_single_rms_states(increments)  # type: ignore[arg-type]
#                 # Correct global state = old global history + union of new client samples.
#                 return cls._merge_single_rms_states([base_state, merged_increments])  # type: ignore[list-item]

#         # First round or incompatible state: pool the full uploaded states once.
#         return cls._merge_single_rms_states(states)  # type: ignore[arg-type]

#     @classmethod
#     def average_vecnormalize_states(
#         cls,
#         states: Sequence[VecNormalizeState | None],
#     ) -> VecNormalizeState | None:
#         """Aggregate VecNormalize obs_rms/ret_rms states from clients."""
#         valid_states = [state for state in states if state is not None]
#         if len(valid_states) == 0:
#             cls._last_global_vecnormalize_state = None
#             return None

#         reference = valid_states[0]
#         previous_global = cls._last_global_vecnormalize_state

#         averaged: VecNormalizeState = {
#             "norm_obs": bool(reference["norm_obs"]),
#             "norm_reward": bool(reference["norm_reward"]),
#             "clip_obs": float(reference["clip_obs"]),
#             "clip_reward": float(reference["clip_reward"]),
#             "gamma": float(reference["gamma"]),
#             "epsilon": float(reference["epsilon"]),
#             "training": bool(reference["training"]),
#             "obs_rms": None,
#             "ret_rms": None,
#         }

#         if reference.get("obs_rms") is not None:
#             averaged["obs_rms"] = cls._aggregate_rms_states(
#                 [state["obs_rms"] for state in valid_states if state.get("obs_rms") is not None],
#                 previous_global.get("obs_rms") if previous_global is not None else None,
#             )
#         if reference.get("ret_rms") is not None:
#             averaged["ret_rms"] = cls._aggregate_rms_states(
#                 [state["ret_rms"] for state in valid_states if state.get("ret_rms") is not None],
#                 previous_global.get("ret_rms") if previous_global is not None else None,
#             )

#         cls._last_global_vecnormalize_state = cls._clone_state(averaged)
#         return averaged

#     def get_upload_payload(self) -> FederatedPayload:
#         return {
#             "modules": self._get_module_states(),
#             "vecnormalize": self._get_vecnormalize_state(),
#             "critic_sync_mode": self.critic_sync_mode,
#             "meta": {
#                 "client_weight": self.get_client_weight(),
#                 "num_timesteps": int(self.num_timesteps),
#             },
#         }

#     @classmethod
#     def aggregate_uploads(
#         cls,
#         uploads: Sequence[FederatedPayload],
#         weights: Sequence[float] | None = None,
#     ) -> FederatedPayload:
#         if len(uploads) == 0:
#             raise ValueError("At least one upload is required for federated aggregation.")

#         critic_modes = {
#             cls._normalize_critic_sync_mode(str(upload.get("critic_sync_mode", "fedavg")))
#             for upload in uploads
#         }
#         if len(critic_modes) != 1:
#             raise ValueError(f"Mixed critic_sync_mode values are not supported in one aggregation: {critic_modes}")
#         critic_sync_mode = next(iter(critic_modes))

#         module_states = [upload["modules"] for upload in uploads]
#         normalized_weights = cls.normalize_weights(len(uploads), weights)
#         aggregated_modules = cls.average_module_states(module_states, weights=weights)
#         aggregated_vecnormalize = cls.average_vecnormalize_states(
#             [upload.get("vecnormalize") for upload in uploads]
#         )
#         return {
#             "modules": aggregated_modules,
#             "vecnormalize": aggregated_vecnormalize,
#             "critic_sync_mode": critic_sync_mode,
#             "meta": {
#                 "num_clients": len(uploads),
#                 "critic_sync_mode": critic_sync_mode,
#                 "client_weights": normalized_weights,
#                 "client_timesteps": [
#                     int(upload.get("meta", {}).get("num_timesteps", 0)) for upload in uploads
#                 ],
#             },
#         }

#     def apply_global_payload(self, payload: FederatedPayload, mix_weight: float = 1.0) -> None:
#         if not (0.0 < mix_weight <= 1.0):
#             raise ValueError("mix_weight must be in (0, 1].")

#         payload_critic_mode = self._normalize_critic_sync_mode(
#             str(payload.get("critic_sync_mode", payload.get("meta", {}).get("critic_sync_mode", self.critic_sync_mode)))
#         )
#         if payload_critic_mode != self.critic_sync_mode:
#             raise ValueError(
#                 f"Client critic_sync_mode={self.critic_sync_mode!r} does not match "
#                 f"payload critic_sync_mode={payload_critic_mode!r}."
#             )

#         old_modules = self._get_module_states()
#         incoming_modules = payload["modules"]
#         if mix_weight < 1.0:
#             mixed_modules: FederatedModules = {}
#             for module_name, module_state in incoming_modules.items():
#                 mixed_modules[module_name] = OrderedDict()
#                 for key, value in module_state.items():
#                     current_value = old_modules[module_name][key]
#                     if th.is_floating_point(value):
#                         mixed_modules[module_name][key] = (
#                             mix_weight * value + (1.0 - mix_weight) * current_value.to(value.dtype)
#                         )
#                     else:
#                         mixed_modules[module_name][key] = value.clone()
#             incoming_modules = mixed_modules

#         self._set_module_states(incoming_modules)
#         new_modules = self._get_module_states()

#         # VecNormalize statistics are environment statistics, not gradient updates.
#         # Apply the aggregated global statistics directly so all clients normalize
#         # observations/rewards on the same scale after every communication round.
#         self._set_vecnormalize_state(payload.get("vecnormalize"))
#         self._reset_after_parameter_sync()

#         meta = payload.get("meta", {})
#         client_weights = np.asarray(meta.get("client_weights", []), dtype=np.float64)
#         client_timesteps = np.asarray(meta.get("client_timesteps", []), dtype=np.float64)
#         metrics: dict[str, float] = {
#             "server/num_clients": float(meta.get("num_clients", 0)),
#             "server/ppo_avg/model_norm": self._module_l2_norm(new_modules),
#             "server/ppo_avg/model_delta_norm": self._module_delta_l2_norm(new_modules, old_modules),
#             "server/ppo_avg/critic_local": float(self.critic_sync_mode == "local"),
#             "server/ppo_avg/critic_fedavg": float(self.critic_sync_mode == "fedavg"),
#         }
#         if client_weights.size > 0:
#             metrics.update(
#                 {
#                     "server/client_weight_mean": float(np.mean(client_weights)),
#                     "server/client_weight_min": float(np.min(client_weights)),
#                     "server/client_weight_max": float(np.max(client_weights)),
#                 }
#             )
#         if client_timesteps.size > 0:
#             metrics.update(
#                 {
#                     "server/client_timesteps_mean": float(np.mean(client_timesteps)),
#                     "server/client_timesteps_min": float(np.min(client_timesteps)),
#                     "server/client_timesteps_max": float(np.max(client_timesteps)),
#                 }
#             )
#         self._last_federated_metrics = metrics

#     def get_client_weight(self) -> float:
#         return 1.0


# # # from __future__ import annotations

# # # from collections import OrderedDict
# # # from collections.abc import Sequence

# # # import torch as th
# # # from stable_baselines3.ppo import PPO

# # # from rl_zoo3.algorithms.federate.buff.buff.federated_algorithm import (
# # #     FederatedAlgorithmMixin,
# # #     FederatedModules,
# # #     FederatedPayload,
# # # )


# # # class PPOAvg(FederatedAlgorithmMixin, PPO):
# # #     """PPO with a FedAvg-style synchronization rule.

# # #     - local client optimization is delegated to SB3's original PPO implementation,
# # #     - clients upload policy parameters,
# # #     - the server averages those parameters,
# # #     - the averaged parameters are pushed back to all clients.

# # #     For PPO, synchronizing ``policy`` is enough because it contains both the
# # #     actor and critic parameters used by the algorithm.
# # #     """

# # #     federated_modules: tuple[str, ...] = ("policy",)
# # #     federated_manager_keys: tuple[str, ...] = ("num_clients", "local_steps", "server_update_weight")

# # #     def __init__(self, *args, **kwargs):
# # #         for key in self.federated_manager_keys:
# # #             kwargs.pop(key, None)
# # #         super().__init__(*args, **kwargs)

# # #     def federated_local_update(self, local_steps: int, **kwargs) -> None:
# # #         self.learn(total_timesteps=local_steps, **kwargs)

# # #     def _get_module_states(self) -> FederatedModules:
# # #         module_states: FederatedModules = {}
# # #         for module_name in self.federated_modules:
# # #             module = getattr(self, module_name)
# # #             module_states[module_name] = OrderedDict(
# # #                 (key, value.detach().cpu().clone()) for key, value in module.state_dict().items()
# # #             )
# # #         return module_states

# # #     def _set_module_states(self, module_states: FederatedModules) -> None:
# # #         device = self.device
# # #         for module_name in self.federated_modules:
# # #             module = getattr(self, module_name)
# # #             state_dict = OrderedDict((key, value.to(device)) for key, value in module_states[module_name].items())
# # #             module.load_state_dict(state_dict)

# # #     def get_upload_payload(self) -> FederatedPayload:
# # #         return {
# # #             "modules": self._get_module_states(),
# # #             "meta": {
# # #                 "client_weight": self.get_client_weight(),
# # #                 "num_timesteps": int(self.num_timesteps),
# # #             },
# # #         }

# # #     @classmethod
# # #     def aggregate_uploads(
# # #         cls,
# # #         uploads: Sequence[FederatedPayload],
# # #         weights: Sequence[float] | None = None,
# # #     ) -> FederatedPayload:
# # #         if len(uploads) == 0:
# # #             raise ValueError("At least one upload is required for federated aggregation.")

# # #         module_states = [upload["modules"] for upload in uploads]
# # #         aggregated_modules = cls.average_module_states(module_states, weights=weights)
# # #         return {
# # #             "modules": aggregated_modules,
# # #             "meta": {
# # #                 "num_clients": len(uploads),
# # #             },
# # #         }

# # #     def apply_global_payload(self, payload: FederatedPayload, mix_weight: float = 1.0) -> None:
# # #         if not (0.0 < mix_weight <= 1.0):
# # #             raise ValueError("mix_weight must be in (0, 1].")

# # #         incoming_modules = payload["modules"]
# # #         if mix_weight < 1.0:
# # #             current_modules = self._get_module_states()
# # #             mixed_modules: FederatedModules = {}
# # #             for module_name, module_state in incoming_modules.items():
# # #                 mixed_modules[module_name] = OrderedDict()
# # #                 for key, value in module_state.items():
# # #                     current_value = current_modules[module_name][key]
# # #                     if th.is_floating_point(value):
# # #                         mixed_modules[module_name][key] = (
# # #                             mix_weight * value + (1.0 - mix_weight) * current_value.to(value.dtype)
# # #                         )
# # #                     else:
# # #                         mixed_modules[module_name][key] = value.clone()
# # #             incoming_modules = mixed_modules

# # #         self._set_module_states(incoming_modules)

# # #     def get_client_weight(self) -> float:
# # #         return 1.0
# # from __future__ import annotations

# # from collections import OrderedDict
# # from collections.abc import Mapping, Sequence
# # from typing import Any

# # import numpy as np
# # import torch as th
# # from stable_baselines3.ppo import PPO

# # from rl_zoo3.algorithms.federate.common.federated_algorithm import (
# #     FederatedAlgorithmMixin,
# #     FederatedModules,
# #     FederatedPayload,
# # )

# # VecNormalizeState = dict[str, Any]
# # RunningMeanStdState = dict[str, Any]


# # class PPOAvg(FederatedAlgorithmMixin, PPO):
# #     """PPO with a FedAvg-style synchronization rule.

# #     - local client optimization is delegated to SB3's original PPO implementation,
# #     - clients upload policy parameters and VecNormalize statistics,
# #     - the server averages those parameters/statistics,
# #     - the averaged payload is pushed back to all clients.

# #     Important VecNormalize detail:
# #     After every communication round, all clients receive the same global
# #     VecNormalize state. Therefore, the next upload from each client contains the
# #     same old global RunningMeanStd history plus that client's new local samples.
# #     If we simply pool the full uploaded counts, the old history is counted once
# #     per client every round and ``count`` grows exponentially. To avoid that, the
# #     server stores the last global VecNormalize state and merges only the newly
# #     added client increments.
# #     """

# #     federated_modules: tuple[str, ...] = ("policy",)
# #     federated_manager_keys: tuple[str, ...] = ("num_clients", "local_steps", "server_update_weight", "log_wandb")

# #     # Last VecNormalize state produced by the server. This lets us subtract the
# #     # shared pre-round normalizer history from each client upload and aggregate
# #     # only the new per-client samples.
# #     _last_global_vecnormalize_state: VecNormalizeState | None = None

# #     def __init__(self, *args, **kwargs):
# #         for key in self.federated_manager_keys:
# #             kwargs.pop(key, None)
# #         super().__init__(*args, **kwargs)
# #         self._last_federated_metrics: dict[str, float] = {}

# #     @classmethod
# #     def reset_federated_state(cls) -> None:
# #         cls._last_global_vecnormalize_state = None

# #     @classmethod
# #     def uses_federated_client_n_envs(cls) -> bool:
# #         return True

# #     def federated_local_update(self, local_steps: int, **kwargs) -> None:
# #         """Run one local PPO stage on this client's vectorized environment.

# #         ``local_steps`` is interpreted as environment interactions. As in SB3
# #         PPO, the realized amount of data is quantized by ``n_steps * n_envs``
# #         because rollouts are collected in fixed-size batches.
# #         """
# #         self.learn(total_timesteps=local_steps, **kwargs)

# #     def _get_module_states(self) -> FederatedModules:
# #         module_states: FederatedModules = {}
# #         for module_name in self.federated_modules:
# #             module = getattr(self, module_name)
# #             module_states[module_name] = OrderedDict(
# #                 (key, value.detach().cpu().clone()) for key, value in module.state_dict().items()
# #             )
# #         return module_states

# #     def _set_module_states(self, module_states: FederatedModules) -> None:
# #         device = self.device
# #         for module_name in self.federated_modules:
# #             module = getattr(self, module_name)
# #             state_dict = OrderedDict((key, value.to(device)) for key, value in module_states[module_name].items())
# #             module.load_state_dict(state_dict)

# #     @staticmethod
# #     def _get_rms_state(rms: Any) -> RunningMeanStdState | dict[str, RunningMeanStdState]:
# #         """Serialize SB3 RunningMeanStd, including dict-observation variants."""
# #         if isinstance(rms, Mapping):
# #             return {key: PPOAvg._get_rms_state(value) for key, value in rms.items()}

# #         return {
# #             "mean": np.asarray(rms.mean, dtype=np.float64).copy(),
# #             "var": np.asarray(rms.var, dtype=np.float64).copy(),
# #             "count": float(rms.count),
# #         }

# #     @staticmethod
# #     def _set_rms_state(rms: Any, state: RunningMeanStdState | dict[str, RunningMeanStdState]) -> None:
# #         """Restore SB3 RunningMeanStd, including dict-observation variants."""
# #         if isinstance(rms, Mapping):
# #             for key, value in state.items():
# #                 if key in rms:
# #                     PPOAvg._set_rms_state(rms[key], value)
# #             return

# #         mean = np.asarray(state["mean"], dtype=np.float64)
# #         var = np.asarray(state["var"], dtype=np.float64)
# #         count = float(state["count"])

# #         # Do not inject invalid normalizer statistics into the env.
# #         if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(var)) or not np.isfinite(count):
# #             raise ValueError(f"Invalid VecNormalize RMS state: count={count}, mean={mean}, var={var}")

# #         rms.mean = mean.copy()
# #         rms.var = np.maximum(var, 1e-12).copy()
# #         rms.count = max(count, 1e-4)

# #     def _get_vecnormalize_state(self) -> VecNormalizeState | None:
# #         """Serialize VecNormalize statistics from this model's env, if present."""
# #         vecnormalize = self.get_vec_normalize_env()
# #         if vecnormalize is None:
# #             return None

# #         state: VecNormalizeState = {
# #             "norm_obs": bool(vecnormalize.norm_obs),
# #             "norm_reward": bool(vecnormalize.norm_reward),
# #             "clip_obs": float(vecnormalize.clip_obs),
# #             "clip_reward": float(vecnormalize.clip_reward),
# #             "gamma": float(vecnormalize.gamma),
# #             "epsilon": float(vecnormalize.epsilon),
# #             "training": bool(vecnormalize.training),
# #             "obs_rms": None,
# #             "ret_rms": None,
# #         }

# #         if getattr(vecnormalize, "obs_rms", None) is not None:
# #             state["obs_rms"] = self._get_rms_state(vecnormalize.obs_rms)
# #         if getattr(vecnormalize, "ret_rms", None) is not None:
# #             state["ret_rms"] = self._get_rms_state(vecnormalize.ret_rms)

# #         return state

# #     def _set_vecnormalize_state(self, state: VecNormalizeState | None) -> None:
# #         """Apply VecNormalize statistics to this model's env, if present."""
# #         if state is None:
# #             return

# #         vecnormalize = self.get_vec_normalize_env()
# #         if vecnormalize is None:
# #             return

# #         # Keep the wrapper configuration aligned across server and clients.
# #         for attr in ("norm_obs", "norm_reward", "clip_obs", "clip_reward", "gamma", "epsilon", "training"):
# #             if attr in state:
# #                 setattr(vecnormalize, attr, state[attr])

# #         if state.get("obs_rms") is not None and getattr(vecnormalize, "obs_rms", None) is not None:
# #             self._set_rms_state(vecnormalize.obs_rms, state["obs_rms"])
# #         if state.get("ret_rms") is not None and getattr(vecnormalize, "ret_rms", None) is not None:
# #             self._set_rms_state(vecnormalize.ret_rms, state["ret_rms"])

# #         # VecNormalize.returns is a transient discounted-return accumulator.
# #         # SB3 does not pickle it when saving VecNormalize, so after synchronizing
# #         # global ret_rms it is safer to restart the accumulator than to average it.
# #         if getattr(vecnormalize, "returns", None) is not None:
# #             vecnormalize.returns = np.zeros_like(vecnormalize.returns)

# #     def _reset_after_parameter_sync(self) -> None:
# #         """Drop PPO state that is stale after replacing policy/normalizer state."""
# #         if hasattr(self.policy, "optimizer"):
# #             self.policy.optimizer.state.clear()

# #         self._last_obs = None
# #         self._last_original_obs = None
# #         self._last_episode_starts = None

# #     @staticmethod
# #     def _clone_state(state: Any) -> Any:
# #         if state is None:
# #             return None
# #         if isinstance(state, np.ndarray):
# #             return state.copy()
# #         if isinstance(state, Mapping):
# #             return {key: PPOAvg._clone_state(value) for key, value in state.items()}
# #         return state

# #     @staticmethod
# #     def _module_l2_norm(modules: FederatedModules) -> float:
# #         total = 0.0
# #         for module_state in modules.values():
# #             for value in module_state.values():
# #                 if th.is_floating_point(value):
# #                     tensor = value.detach().to(dtype=th.float64)
# #                     total += float(th.sum(tensor * tensor).cpu().item())
# #         return float(np.sqrt(total))

# #     @staticmethod
# #     def _module_delta_l2_norm(after: FederatedModules, before: FederatedModules) -> float:
# #         total = 0.0
# #         for module_name, after_state in after.items():
# #             if module_name not in before:
# #                 continue
# #             for key, after_value in after_state.items():
# #                 if key not in before[module_name] or not th.is_floating_point(after_value):
# #                     continue
# #                 delta = after_value.detach().to(dtype=th.float64) - before[module_name][key].detach().to(dtype=th.float64)
# #                 total += float(th.sum(delta * delta).cpu().item())
# #         return float(np.sqrt(total))

# #     @staticmethod
# #     def _is_single_rms_state(state: Any) -> bool:
# #         return isinstance(state, Mapping) and {"mean", "var", "count"}.issubset(state.keys())

# #     @staticmethod
# #     def _rms_is_finite(state: RunningMeanStdState) -> bool:
# #         return (
# #             np.isfinite(float(state["count"]))
# #             and np.all(np.isfinite(np.asarray(state["mean"], dtype=np.float64)))
# #             and np.all(np.isfinite(np.asarray(state["var"], dtype=np.float64)))
# #         )

# #     @classmethod
# #     def _merge_single_rms_states(
# #         cls,
# #         states: Sequence[RunningMeanStdState],
# #     ) -> RunningMeanStdState:
# #         """Merge independent RunningMeanStd states exactly via M2 statistics."""
# #         valid_states = [state for state in states if cls._rms_is_finite(state) and float(state["count"]) > 0.0]
# #         if len(valid_states) == 0:
# #             reference = states[0]
# #             return {
# #                 "mean": np.asarray(reference["mean"], dtype=np.float64).copy(),
# #                 "var": np.maximum(np.asarray(reference["var"], dtype=np.float64), 1e-12).copy(),
# #                 "count": max(float(reference["count"]), 1e-4),
# #             }

# #         mean = np.asarray(valid_states[0]["mean"], dtype=np.float64).copy()
# #         var = np.maximum(np.asarray(valid_states[0]["var"], dtype=np.float64), 0.0).copy()
# #         count = float(valid_states[0]["count"])
# #         m2 = var * count

# #         for state in valid_states[1:]:
# #             other_count = float(state["count"])
# #             other_mean = np.asarray(state["mean"], dtype=np.float64)
# #             other_var = np.maximum(np.asarray(state["var"], dtype=np.float64), 0.0)
# #             other_m2 = other_var * other_count

# #             total = count + other_count
# #             if total <= 0.0:
# #                 continue
# #             delta = other_mean - mean
# #             mean = mean + delta * other_count / total
# #             m2 = m2 + other_m2 + np.square(delta) * count * other_count / total
# #             count = total

# #         var = np.maximum(m2 / max(count, 1e-12), 1e-12)
# #         return {"mean": mean, "var": var, "count": max(count, 1e-4)}

# #     @classmethod
# #     def _subtract_single_rms_state(
# #         cls,
# #         final_state: RunningMeanStdState,
# #         base_state: RunningMeanStdState,
# #     ) -> RunningMeanStdState | None:
# #         """Recover the incremental samples D from final_state = merge(base_state, D).

# #         Returns None when subtraction is not numerically or structurally valid.
# #         """
# #         if not cls._rms_is_finite(final_state) or not cls._rms_is_finite(base_state):
# #             return None

# #         final_count = float(final_state["count"])
# #         base_count = float(base_state["count"])
# #         inc_count = final_count - base_count

# #         # If the client did not start from the stored global state, subtraction is invalid.
# #         if inc_count <= 1e-8:
# #             return None

# #         final_mean = np.asarray(final_state["mean"], dtype=np.float64)
# #         base_mean = np.asarray(base_state["mean"], dtype=np.float64)
# #         final_var = np.maximum(np.asarray(final_state["var"], dtype=np.float64), 0.0)
# #         base_var = np.maximum(np.asarray(base_state["var"], dtype=np.float64), 0.0)

# #         if final_mean.shape != base_mean.shape or final_var.shape != base_var.shape:
# #             return None

# #         # From: final_count * final_mean = base_count * base_mean + inc_count * inc_mean
# #         inc_mean = (final_count * final_mean - base_count * base_mean) / inc_count

# #         # Merge formula:
# #         # M2_final = M2_base + M2_inc + (inc_mean - base_mean)^2 * base_count * inc_count / final_count
# #         m2_final = final_var * final_count
# #         m2_base = base_var * base_count
# #         correction = np.square(inc_mean - base_mean) * base_count * inc_count / final_count
# #         m2_inc = m2_final - m2_base - correction

# #         # Tiny negative values can occur from floating point cancellation.
# #         m2_inc = np.maximum(m2_inc, 0.0)
# #         inc_var = np.maximum(m2_inc / inc_count, 1e-12)

# #         if not np.all(np.isfinite(inc_mean)) or not np.all(np.isfinite(inc_var)):
# #             return None

# #         return {"mean": inc_mean, "var": inc_var, "count": inc_count}

# #     @classmethod
# #     def _aggregate_rms_states(
# #         cls,
# #         states: Sequence[RunningMeanStdState | dict[str, RunningMeanStdState]],
# #         base_state: RunningMeanStdState | dict[str, RunningMeanStdState] | None = None,
# #     ) -> RunningMeanStdState | dict[str, RunningMeanStdState]:
# #         """Aggregate RunningMeanStd states without double-counting shared history.

# #         If base_state is provided, each client upload is interpreted as
# #         merge(base_state, local_increment), and only local_increment is merged
# #         into the new global state. If subtraction fails, fall back to pooling the
# #         full states once.
# #         """
# #         if len(states) == 0:
# #             raise ValueError("At least one RunningMeanStd state is required.")

# #         reference = states[0]
# #         if not cls._is_single_rms_state(reference):
# #             base_mapping = base_state if isinstance(base_state, Mapping) else None
# #             return {
# #                 key: cls._aggregate_rms_states(
# #                     [state[key] for state in states],
# #                     base_mapping.get(key) if base_mapping is not None and key in base_mapping else None,
# #                 )
# #                 for key in reference.keys()
# #             }

# #         if base_state is not None and cls._is_single_rms_state(base_state):
# #             increments = [cls._subtract_single_rms_state(state, base_state) for state in states]
# #             if all(increment is not None for increment in increments):
# #                 merged_increments = cls._merge_single_rms_states(increments)  # type: ignore[arg-type]
# #                 # Correct global state = old global history + union of new client samples.
# #                 return cls._merge_single_rms_states([base_state, merged_increments])  # type: ignore[list-item]

# #         # First round or incompatible state: pool the full uploaded states once.
# #         return cls._merge_single_rms_states(states)  # type: ignore[arg-type]

# #     @classmethod
# #     def average_vecnormalize_states(
# #         cls,
# #         states: Sequence[VecNormalizeState | None],
# #     ) -> VecNormalizeState | None:
# #         """Aggregate VecNormalize obs_rms/ret_rms states from clients."""
# #         valid_states = [state for state in states if state is not None]
# #         if len(valid_states) == 0:
# #             cls._last_global_vecnormalize_state = None
# #             return None

# #         reference = valid_states[0]
# #         previous_global = cls._last_global_vecnormalize_state

# #         averaged: VecNormalizeState = {
# #             "norm_obs": bool(reference["norm_obs"]),
# #             "norm_reward": bool(reference["norm_reward"]),
# #             "clip_obs": float(reference["clip_obs"]),
# #             "clip_reward": float(reference["clip_reward"]),
# #             "gamma": float(reference["gamma"]),
# #             "epsilon": float(reference["epsilon"]),
# #             "training": bool(reference["training"]),
# #             "obs_rms": None,
# #             "ret_rms": None,
# #         }

# #         if reference.get("obs_rms") is not None:
# #             averaged["obs_rms"] = cls._aggregate_rms_states(
# #                 [state["obs_rms"] for state in valid_states if state.get("obs_rms") is not None],
# #                 previous_global.get("obs_rms") if previous_global is not None else None,
# #             )
# #         if reference.get("ret_rms") is not None:
# #             averaged["ret_rms"] = cls._aggregate_rms_states(
# #                 [state["ret_rms"] for state in valid_states if state.get("ret_rms") is not None],
# #                 previous_global.get("ret_rms") if previous_global is not None else None,
# #             )

# #         cls._last_global_vecnormalize_state = cls._clone_state(averaged)
# #         return averaged

# #     def get_upload_payload(self) -> FederatedPayload:
# #         return {
# #             "modules": self._get_module_states(),
# #             "vecnormalize": self._get_vecnormalize_state(),
# #             "meta": {
# #                 "client_weight": self.get_client_weight(),
# #                 "num_timesteps": int(self.num_timesteps),
# #             },
# #         }

# #     @classmethod
# #     def aggregate_uploads(
# #         cls,
# #         uploads: Sequence[FederatedPayload],
# #         weights: Sequence[float] | None = None,
# #     ) -> FederatedPayload:
# #         if len(uploads) == 0:
# #             raise ValueError("At least one upload is required for federated aggregation.")

# #         module_states = [upload["modules"] for upload in uploads]
# #         normalized_weights = cls.normalize_weights(len(uploads), weights)
# #         aggregated_modules = cls.average_module_states(module_states, weights=weights)
# #         aggregated_vecnormalize = cls.average_vecnormalize_states(
# #             [upload.get("vecnormalize") for upload in uploads]
# #         )
# #         return {
# #             "modules": aggregated_modules,
# #             "vecnormalize": aggregated_vecnormalize,
# #             "meta": {
# #                 "num_clients": len(uploads),
# #                 "client_weights": normalized_weights,
# #                 "client_timesteps": [
# #                     int(upload.get("meta", {}).get("num_timesteps", 0)) for upload in uploads
# #                 ],
# #             },
# #         }

# #     def apply_global_payload(self, payload: FederatedPayload, mix_weight: float = 1.0) -> None:
# #         if not (0.0 < mix_weight <= 1.0):
# #             raise ValueError("mix_weight must be in (0, 1].")

# #         old_modules = self._get_module_states()
# #         incoming_modules = payload["modules"]
# #         if mix_weight < 1.0:
# #             mixed_modules: FederatedModules = {}
# #             for module_name, module_state in incoming_modules.items():
# #                 mixed_modules[module_name] = OrderedDict()
# #                 for key, value in module_state.items():
# #                     current_value = old_modules[module_name][key]
# #                     if th.is_floating_point(value):
# #                         mixed_modules[module_name][key] = (
# #                             mix_weight * value + (1.0 - mix_weight) * current_value.to(value.dtype)
# #                         )
# #                     else:
# #                         mixed_modules[module_name][key] = value.clone()
# #             incoming_modules = mixed_modules

# #         self._set_module_states(incoming_modules)
# #         new_modules = self._get_module_states()

# #         # VecNormalize statistics are environment statistics, not gradient updates.
# #         # Apply the aggregated global statistics directly so all clients normalize
# #         # observations/rewards on the same scale after every communication round.
# #         self._set_vecnormalize_state(payload.get("vecnormalize"))
# #         self._reset_after_parameter_sync()

# #         meta = payload.get("meta", {})
# #         client_weights = np.asarray(meta.get("client_weights", []), dtype=np.float64)
# #         client_timesteps = np.asarray(meta.get("client_timesteps", []), dtype=np.float64)
# #         metrics: dict[str, float] = {
# #             "server/num_clients": float(meta.get("num_clients", 0)),
# #             "server/ppo_avg/model_norm": self._module_l2_norm(new_modules),
# #             "server/ppo_avg/model_delta_norm": self._module_delta_l2_norm(new_modules, old_modules),
# #         }
# #         if client_weights.size > 0:
# #             metrics.update(
# #                 {
# #                     "server/client_weight_mean": float(np.mean(client_weights)),
# #                     "server/client_weight_min": float(np.min(client_weights)),
# #                     "server/client_weight_max": float(np.max(client_weights)),
# #                 }
# #             )
# #         if client_timesteps.size > 0:
# #             metrics.update(
# #                 {
# #                     "server/client_timesteps_mean": float(np.mean(client_timesteps)),
# #                     "server/client_timesteps_min": float(np.min(client_timesteps)),
# #                     "server/client_timesteps_max": float(np.max(client_timesteps)),
# #                 }
# #             )
# #         self._last_federated_metrics = metrics

# #     def get_client_weight(self) -> float:
# #         return 1.0
