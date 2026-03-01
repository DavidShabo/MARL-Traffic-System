import argparse
import os
import ray
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env

from envs import build_env_config, make_env  # ✅ use same config builder as training


def main(args):
    register_env("metadrive_roundabout", make_env)
    ray.init(ignore_reinit_error=True)

    checkpoint_path = os.path.abspath(args.checkpoint)
    algo = PPO.from_checkpoint(checkpoint_path)

    env_config = build_env_config(num_agents=args.num_agents, render=True, stage=args.stage)
    env_config["env_name"] = args.env         
    env_config["allow_respawn"] = False       

    env = make_env(env_config)

    obs, info = env.reset()

    while True:
        actions = {
            agent_id: algo.compute_single_action(
                agent_obs,
                policy_id="shared_policy",
                explore=False,
            )
            for agent_id, agent_obs in obs.items()
        }

        obs, rew, terminated, truncated, info = env.step(actions)
        env.render()

        if terminated.get("__all__", False) or truncated.get("__all__", False):
            obs, info = env.reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-agents", type=int, default=1)
    parser.add_argument("--env", type=str, default="roundabout",
                        choices=["roundabout", "intersection", "tollgate"])
    parser.add_argument("--stage", type=int, default=3, choices=[1, 2, 3])  
    args = parser.parse_args()

    main(args)