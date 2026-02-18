import numpy as np
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

    def _custom_reward(self, obs, info):
        rdict = {}
        for agent_id in obs.keys():
            ainfo = info.get(agent_id, {})
            r = 0.0
            v = ainfo.get("velocity", 0.0)

            if isinstance(v, (list, tuple)):
                    v = v[0]
            v = float(np.clip(np.nan_to_num(v, nan=0.0, posinf=20.0, neginf=0.0), 0.0, 20.0))

            if ainfo.get("crash", False):
                r -= 10.0
            if ainfo.get("out_of_road", False):
                r -= 5.0
            #dont let them just stand still for points
            if v > 0.5:
                r += 0.01
            else:
                r -= 0.02

            target = 12.0
            r += max(0.0, 1.0 - abs(v - target) / target) * 0.02

            if not ainfo.get("out_of_road", False) and v > 0.1:
                r += 0.005

            rdict[agent_id] = float(np.clip(r, -10.0, 10.0))

        return rdict

    def step(self, action_dict):
        obs, rew, term, trunc, info = self.env.step(action_dict)

        if "__all__" not in term:
            term["__all__"] = all(term.get(a, False) for a in obs.keys())
        if "__all__" not in trunc:
            trunc["__all__"] = all(trunc.get(a, False) for a in obs.keys())

        custom_rew = self._custom_reward(obs, info)
        return obs, custom_rew, term, trunc, info

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
        "horizon": 1000,
    }

def make_env(config: dict):
    return RLLibMetaDriveRoundabout(config)
