"""RLlib training entry point for MetaDrive MultiAgentRoundaboutEnv (compat mode)."""

from __future__ import annotations

import argparse
import os
from typing import Any, Optional, Tuple

import ray
from metadrive import MultiAgentRoundaboutEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from envs import build_env_config, make_env

def _first_space(space: Any):
    """If Gymnasium Dict space, return the first sub-space; else return as-is."""
    if hasattr(space, "spaces") and isinstance(space.spaces, dict) and len(space.spaces) > 0:
        return next(iter(space.spaces.values()))
    return space


def _ckpt_path_str(ckpt_obj: Any) -> str:
    """
    RLlib/Ray versions vary: save() may return a string path, a Checkpoint,
    or a TrainingResult that contains a Checkpoint.
    This extracts a usable path string robustly.
    """
    if isinstance(ckpt_obj, str):
        return ckpt_obj

    # Try ckpt_obj.path
    if hasattr(ckpt_obj, "path") and isinstance(getattr(ckpt_obj, "path"), str):
        return ckpt_obj.path

    # Try ckpt_obj.checkpoint.path (TrainingResult(checkpoint=Checkpoint(...)))
    if hasattr(ckpt_obj, "checkpoint"):
        cp = getattr(ckpt_obj, "checkpoint")
        if hasattr(cp, "path") and isinstance(getattr(cp, "path"), str):
            return cp.path

    return str(ckpt_obj)


def build_algo_config(args: argparse.Namespace) -> PPOConfig:
    env_config = build_env_config(args.num_agents, args.render)

    # Dummy env to read spaces (MetaDrive sometimes exposes Dict spaces)
    dummy = MultiAgentRoundaboutEnv(env_config)
    obs_space = _first_space(dummy.observation_space)
    act_space = _first_space(dummy.action_space)
    dummy.close()

    policies = {
        "shared_policy": (
            None,       # RLlib builds default PPO torch policy
            obs_space,
            act_space,
            {},
        )
    }

    return (
        PPOConfig()
        # Keep compatibility (old RLlib API stack)
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(env="metadrive_roundabout", env_config=env_config)
        .framework("torch")
        # Old stack but using new config name (you already migrated off .rollouts)
        .env_runners(num_env_runners=args.workers)
        .training(
            train_batch_size=args.train_batch_size,
            lr=1e-4,
            gamma=0.99,
            use_critic=True,
            use_gae=True,
            lambda_=0.95,
            entropy_coeff=0.0,
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=lambda agent_id, *a, **k: "shared_policy",
        )
        .resources(num_gpus=args.gpus)
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MetaDrive MultiAgentRoundaboutEnv with RLlib PPO.")
    p.add_argument("--num-agents", type=int, default=1)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--gpus", type=int, default=0)
    p.add_argument("--train-batch-size", type=int, default=4000)
    p.add_argument("--stop-iters", type=int, default=50)
    p.add_argument("--render", action="store_true")

    # New: stable checkpoint dir (prevents temp-path saves)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")

    return p.parse_args()


def _extract_metrics(results: dict) -> Tuple[Optional[float], Optional[float]]:
    er = None
    el = None

    # Newer RLlib structure
    env_runners = results.get("env_runners")
    if isinstance(env_runners, dict):
        episode = env_runners.get("episode")
        if isinstance(episode, dict):
            er = episode.get("mean_reward")
            el = episode.get("mean_length")

    # Older structure
    if er is None:
        er = results.get("episode_reward_mean")
        el = results.get("episode_len_mean")

    # Old API stack: sampler_results
    if er is None:
        sampler = results.get("sampler_results")
        if isinstance(sampler, dict):
            er = sampler.get("episode_reward_mean")
            el = sampler.get("episode_len_mean")
            if er is None:
                hist = sampler.get("hist_stats", {})
                rewards = hist.get("episode_reward") or hist.get("episode_rewards")
                lengths = hist.get("episode_lengths")
                if rewards:
                    er = float(sum(rewards) / max(1, len(rewards)))
                if lengths:
                    el = float(sum(lengths) / max(1, len(lengths)))

    # Fallback: compute from hist_stats if available
    if er is None:
        hist = results.get("hist_stats", {})
        rewards = hist.get("episode_reward") or hist.get("episode_rewards")
        lengths = hist.get("episode_lengths")
        if rewards:
            er = float(sum(rewards) / max(1, len(rewards)))
        if lengths:
            el = float(sum(lengths) / max(1, len(lengths)))

    return er, el


def main() -> None:
    print("TRAIN.PY STARTED")

    args = parse_args()
    register_env("metadrive_roundabout", make_env)

    os.environ.setdefault("RAY_DISABLE_METRICS_EXPORT", "1")
    os.environ.setdefault("RAY_METRICS_EXPORT_PORT", "0")
    ray.init(ignore_reinit_error=True, include_dashboard=False, _metrics_export_port=0)

    algo_config = build_algo_config(args)
    if hasattr(algo_config, "build_algo"):
        algo = algo_config.build_algo()
    else:
        algo = algo_config.build()

    for it in range(1, args.stop_iters + 1):
        results = algo.train()

        er, el = _extract_metrics(results)

        print(f"iter={it} reward_mean={er} len_mean={el}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    print("ABOUT TO SAVE CHECKPOINT...")
    ckpt_obj = algo.save(args.checkpoint_dir)
    ckpt_path = _ckpt_path_str(ckpt_obj)
    print(f"Checkpoint saved to: {ckpt_path}")

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()