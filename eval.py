import argparse
import os
import ray
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env

from envs.env import make_env


def main(checkpoint_path, num_agents):
    # Register the custom env for RLlib
    register_env("metadrive_roundabout", make_env)

    # Initialize Ray runtime
    ray.init(ignore_reinit_error=True)

    # Convert to absolute path for pyarrow compatibility
    checkpoint_path = os.path.abspath(checkpoint_path)
    # Load trained PPO policy from checkpoint
    algo = PPO.from_checkpoint(checkpoint_path)

    # Create environment with rendering enabled
    env = make_env({
        "use_render": True,
        "manual_control": False,
        "log_level": 50,
        "num_agents": num_agents,
    })

    # Start a new episode
    obs, info = env.reset()

    while True:
        actions = {}
        for agent_id, agent_obs in obs.items():
            # Compute each agent's action using the shared policy
            actions[agent_id] = algo.compute_single_action(
                agent_obs,
                policy_id="shared_policy",
                explore=False,
            )

        # Step the environment with the action dict
        obs, rew, terminated, truncated, info = env.step(actions)
        env.render()

        # Reset when all agents are done or truncated
        if terminated.get("__all__", False) or truncated.get("__all__", False):
            obs, info = env.reset()


if __name__ == "__main__":
    # CLI: python eval.py --checkpoint /path/to/checkpoint
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-agents", type=int, default=1)
    args = parser.parse_args()
    main(args.checkpoint, args.num_agents)