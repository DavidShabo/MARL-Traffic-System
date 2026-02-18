import argparse
from pathlib import Path

import ray
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env

from envs import make_env


def _resolve_checkpoint_path(checkpoint_path: str) -> str:
    path = Path(checkpoint_path).expanduser().resolve()

    if path.is_dir():
        if (path / "rllib_checkpoint.json").exists():
            return str(path)

    if path.is_file() and path.name == "rllib_checkpoint.json":
        return str(path.parent)

    return str(path)
from train import make_env  
from envs.env import make_env


def main(checkpoint_path: str, num_agents: int, explore: bool) -> None:
    register_env("metadrive_roundabout", make_env)

    ray.init(ignore_reinit_error=True)

    algo = PPO.from_checkpoint(_resolve_checkpoint_path(checkpoint_path))

    env = make_env({
        "use_render": True,
        "manual_control": False,
        "log_level": 50,
        "num_agents": num_agents,
    })

    obs, info = env.reset()

    while True:
        actions = {}
        for agent_id, agent_obs in obs.items():
            actions[agent_id] = algo.compute_single_action(
                agent_obs,
                policy_id="shared_policy",
                explore=explore,
            )

        obs, rew, term, trunc, info = env.step(actions)
        env.render()

        if term.get("__all__", False) or trunc.get("__all__", False):
            obs, info = env.reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-agents", type=int, default=8)
    parser.add_argument(
        "--explore",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable exploration for more stochastic actions (helps movement if policy is undertrained).",
    )
    args = parser.parse_args()
    main(args.checkpoint, args.num_agents, args.explore)
    main(args.checkpoint)
