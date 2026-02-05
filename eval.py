import argparse
import ray
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env

from train import make_env  
from envs.env import make_env


def main(checkpoint_path):
    register_env("metadrive_roundabout", make_env)

    ray.init(ignore_reinit_error=True)

    algo = PPO.from_checkpoint(checkpoint_path)

    env = make_env({
        "use_render": True,
        "manual_control": False,
        "log_level": 50,
        "num_agents": 8,
    })

    obs, info = env.reset()

    while True:
        actions = {}
        for agent_id, agent_obs in obs.items():
            actions[agent_id] = algo.compute_single_action(
                agent_obs,
                policy_id="shared_policy",
                explore=False,
            )

        obs, rew, term, trunc, info = env.step(actions)
        env.render()

        if term.get("__all__", False) or trunc.get("__all__", False):
            obs, info = env.reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    main(args.checkpoint)