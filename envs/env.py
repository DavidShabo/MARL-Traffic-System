from metadrive import MultiAgentRoundaboutEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv

class RLLibMetaDriveRoundabout(MultiAgentEnv):
    def __init__(self, config: dict):
        self.env = MultiAgentRoundaboutEnv(config)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed)
        return obs, info

    def step(self, action_dict):
        obs, rew, term, trunc, info = self.env.step(action_dict)

        # Ensure __all__ exists (RLlib compat)
        if "__all__" not in term:
            term["__all__"] = all(term.get(a, False) for a in obs.keys())
        if "__all__" not in trunc:
            trunc["__all__"] = all(trunc.get(a, False) for a in obs.keys())

        return obs, rew, term, trunc, info

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()


def build_env_config(num_agents: int, render: bool) -> dict:
    return {
        "use_render": render,
        "manual_control": False,
        "log_level": 50,
        "num_agents": num_agents,
    }


def make_env(config: dict):
    return RLLibMetaDriveRoundabout(config)
