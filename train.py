"""RLlib training entry point for MetaDrive MultiAgentRoundaboutEnv (compat mode)."""

from __future__ import annotations

import argparse
import os
from typing import Any

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

    if hasattr(ckpt_obj, "checkpoint"):
        cp = getattr(ckpt_obj, "checkpoint")
        if hasattr(cp, "path") and isinstance(getattr(cp, "path"), str):
            return cp.path

    return str(ckpt_obj)


def build_algo_config(args: argparse.Namespace) -> PPOConfig:
    env_config = build_env_config(args.num_agents, args.render)

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
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(env="metadrive_roundabout", env_config=env_config)
        .framework("torch")
        .env_runners(num_env_runners=args.workers)
        .training(train_batch_size=args.train_batch_size)
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

    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")

    return p.parse_args()


def main() -> None:
    print("TRAIN.PY STARTED")

    args = parse_args()
    register_env("metadrive_roundabout", make_env)

    ray.init(ignore_reinit_error=True)

    algo = build_algo_config(args).build()

    for it in range(1, args.stop_iters + 1):
        results = algo.train()

        er = None
        el = None
        try:
            er = results["env_runners"]["episode_reward_mean"]
            el = results["env_runners"]["episode_len_mean"]
        except Exception:

            er = results.get("episode_reward_mean")
            el = results.get("episode_len_mean")

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