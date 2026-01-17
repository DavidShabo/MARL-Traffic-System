"""RLlib training entry point for MetaDrive MultiAgentRoundaboutEnv."""

from __future__ import annotations
from ray.rllib.env.multi_agent_env import MultiAgentEnv

import argparse

import ray
from metadrive import MultiAgentRoundaboutEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env




class RLLibMetaDriveRoundabout(MultiAgentEnv):
    def __init__(self, config):
        self.env = MultiAgentRoundaboutEnv(config)
        # shared spaces
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed)
        return obs, info  # old RLlib stack expects obs only

    def step(self, action_dict):
        obs, rew, term, trunc, info = self.env.step(action_dict)

    # Ensure required "__all__" keys exist
        if "__all__" not in term:
            term["__all__"] = all(term.get(a, False) for a in obs.keys())
        if "__all__" not in trunc:
            trunc["__all__"] = all(trunc.get(a, False) for a in obs.keys())

        return obs, rew, term, trunc, info

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()

def build_env_config(num_agents: int, use_render: bool) -> dict:
    return {
        "use_render": use_render,
        "manual_control": False,
        "log_level": 50,
        "num_agents": num_agents,
    }


def make_env(config: dict):
    return RLLibMetaDriveRoundabout(config)

def build_algo_config(args: argparse.Namespace) -> PPOConfig:
    env_config = build_env_config(args.num_agents, args.render)

    dummy = MultiAgentRoundaboutEnv(env_config)

    obs_space = dummy.observation_space
    act_space = dummy.action_space

    # If MetaDrive exposes Dict spaces keyed by agent ids, grab one agent's space
    try:
    # Gymnasium Dict space supports .spaces
     if hasattr(obs_space, "spaces"):
        obs_space = list(obs_space.spaces.values())[0]
        if hasattr(act_space, "spaces"):
            act_space = list(act_space.spaces.values())[0]
    except Exception:
       pass

    dummy.close()


    policies = {
        "shared_policy": (
            None,      
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
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",

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
    return p.parse_args()


def main() -> None:
    print("TRAIN.PY STARTED")

    args = parse_args()
    register_env("metadrive_roundabout", make_env)

    ray.init(ignore_reinit_error=True)

    algo = build_algo_config(args).build()

    for it in range(1, args.stop_iters + 1):
        results = algo.train()
        reward = results.get("episode_reward_mean")
        length = results.get("episode_len_mean")
        print(f"iter={it} reward_mean={reward} len_mean={length}")

    checkpoint = algo.save()
    print(f"Checkpoint saved to: {checkpoint}")

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
