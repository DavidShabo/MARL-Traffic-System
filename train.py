"""RLlib training entry point for MetaDrive MultiAgentRoundaboutEnv (compat mode)."""

from __future__ import annotations

import argparse
import os
import signal
from typing import Any

import ray
from metadrive import MultiAgentRoundaboutEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from envs import build_env_config, make_env

# Global flag for graceful shutdown
_stop_training = False


def _signal_handler(signum, frame):
    """Handle Ctrl+C to stop training gracefully."""
    global _stop_training
    print("\n⚠️  Interrupt received! Finishing current iteration and saving checkpoint...")
    _stop_training = True


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
        .training(
            train_batch_size=args.train_batch_size,
            lr=0.0003,              # Learning rate (higher = faster but less stable)
            gamma=0.99,             # Discount factor (0.99 is good for driving)
            lambda_=0.95,           # GAE lambda (higher = less bias, more variance)
            num_sgd_iter=10,        # SGD passes per batch (more = better use of data)
            minibatch_size=512,     # Minibatch size (larger = faster GPU utilization)
            entropy_coeff=0.01,     # Exploration bonus (higher = more exploration)
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

    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")

    return p.parse_args()


def main() -> None:
    global _stop_training
    print("TRAIN.PY STARTED")
    print("💡 Press Ctrl+C at any time to stop and save checkpoint\n")

    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, _signal_handler)

    args = parse_args()
    register_env("metadrive_roundabout", make_env)

    ray.init(ignore_reinit_error=True)

    algo = build_algo_config(args).build()

    import datetime
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    last_ckpt_path = None
    for it in range(1, args.stop_iters + 1):
        if _stop_training:
            print(f"\n🛑 Stopping at iteration {it-1}")
            break

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

        # Save a uniquely named checkpoint every iteration
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_ckpt_dir = os.path.join(args.checkpoint_dir, f"ckpt_iter{it}_{timestamp}")
        os.makedirs(unique_ckpt_dir, exist_ok=True)
        ckpt_obj = algo.save(unique_ckpt_dir)
        ckpt_path = _ckpt_path_str(ckpt_obj)
        print(f"Checkpoint saved to: {ckpt_path}")
        last_ckpt_path = ckpt_path

    # After training, also copy the last checkpoint to a 'final' uniquely named location
    if last_ckpt_path:
        final_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        final_ckpt_dir = os.path.join(args.checkpoint_dir, f"final_ckpt_{final_timestamp}")
        os.makedirs(final_ckpt_dir, exist_ok=True)
        import shutil
        # Copy all files from last_ckpt_path's directory to final_ckpt_dir
        src_dir = os.path.dirname(last_ckpt_path)
        for fname in os.listdir(src_dir):
            shutil.copy2(os.path.join(src_dir, fname), final_ckpt_dir)
        print(f"Final checkpoint copied to: {final_ckpt_dir}")

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()