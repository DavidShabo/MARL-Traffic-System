import argparse
from pathlib import Path

import ray
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env

from envs import make_env


def _resolve_checkpoint_path(checkpoint_path: str) -> str:
    # Accept either a directory or a path to rllib_checkpoint.json
    path = Path(checkpoint_path).expanduser().resolve()

    if path.is_dir():
        if (path / "rllib_checkpoint.json").exists():
            return str(path)

    if path.is_file() and path.name == "rllib_checkpoint.json":
        return str(path.parent)

    return str(path)


def main(checkpoint_path: str, num_agents: int, explore: bool) -> None:
    # Register environment so RLlib can create it by name
    register_env("metadrive_roundabout", make_env)

    # Start Ray runtime (required by RLlib)
    ray.init(ignore_reinit_error=True)

    # Load trained PPO policy from checkpoint
    algo = PPO.from_checkpoint(_resolve_checkpoint_path(checkpoint_path))

    # Build the environment for rendering and evaluation
    env = make_env({
        "use_render": True,
        "manual_control": False,
        "log_level": 50,
        "num_agents": num_agents,
    })

    # Initialize the first episode
    obs, info = env.reset()

    # Run forever (Ctrl+C to stop). Each loop is one environment step.
    while True:
        actions = {}
        for agent_id, agent_obs in obs.items():
            # Compute a single action per agent using the shared policy
            actions[agent_id] = algo.compute_single_action(
                agent_obs,
                policy_id="shared_policy",
                explore=explore,
            )

        # Step the environment, then render the scene
        obs, rew, term, trunc, info = env.step(actions)
        env.render()

        # Reset once all agents are done
        if term.get("__all__", False) or trunc.get("__all__", False):
            obs, info = env.reset()


if __name__ == "__main__":
    # CLI for checkpoint selection and evaluation parameters
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
