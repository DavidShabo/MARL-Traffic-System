import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from metadrive import MultiAgentBottleneckEnv, MultiAgentRoundaboutEnv
from metadrive import MultiAgentIntersectionEnv
from metadrive import MultiAgentTollgateEnv

ENV_REGISTRY = {
    "roundabout": MultiAgentRoundaboutEnv,
    "intersection": MultiAgentIntersectionEnv,
    "tollgate": MultiAgentTollgateEnv,
    "bottleneck": MultiAgentBottleneckEnv,  
}


class RLLibMetaDriveEnv(MultiAgentEnv):

    def __init__(self, config: dict):
        cfg = dict(config)

        env_name = cfg.pop("env_name", "roundabout")
        env_cls = ENV_REGISTRY[env_name]

        self.stage = int(cfg.pop("stage", 1))
        self._base_seed = int(cfg.pop("start_seed", 0))

        self.env = env_cls(cfg)

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self._episode_idx = 0
        self._prev_progress = {}
        self._prev_steer = {}
        self._stall_steps = {}

        self._agent_ids = set(self.env.agents.keys()) if hasattr(self.env, "agents") else set()

        self.metadata = {}

    def reset(self, *, seed=None, options=None):
        base = int(getattr(self, "_base_seed", 0))
        idx = int(getattr(self, "_episode_idx", 0))
        n = int(getattr(self.env, "num_scenarios", 1000))

        if seed is None:
            seed = (base + idx) % n

        self._episode_idx = idx + 1

        obs, info = self.env.reset(seed=seed)

        self._prev_progress = {}
        self._prev_steer = {}
        self._stall_steps = {}
        self._agent_ids = set(obs.keys())

        return obs, info

    def _custom_reward(self, obs, info):
        rewards_dict = {}

        for agent_id in obs.keys():
            agent_info = info.get(agent_id, {})
            reward = 0.0

            # 1. PROGRESS TO DESTINATION
            current_progress = float(agent_info.get("route_completion", 0.0))
            prev_progress = self._prev_progress.get(agent_id, 0.0)
            delta_progress = current_progress - prev_progress
            self._prev_progress[agent_id] = current_progress

            reward += 8.0 * delta_progress

            # 2. DESTINATION ARRIVAL
            if agent_info.get("arrive_dest", False):
                reward += 20.0

            # 3. SPEED & STEERING CONTROL
            velocity = float(agent_info.get("velocity", 0.0))
            steering = float(agent_info.get("steering", 0.0))
            abs_steering = abs(steering)

            # Smaller raw steering penalty
            reward -= 0.4 * abs_steering

            # Penalize twitchy left-right steering changes
            prev_steer = self._prev_steer.get(agent_id, steering)
            reward -= 0.02 * abs(steering - prev_steer)
            self._prev_steer[agent_id] = steering

            # Softer speed shaping in turns
            if abs_steering > 0.3:
                target_speed = 4.0
                if velocity > target_speed:
                    reward -= 0.25 * (velocity - target_speed)
                elif velocity > 4.0:
                    reward += 0.25
                elif velocity > 3.0:
                    reward += 0.15
            else:
                target_speed = 10.0
                if velocity > target_speed:
                    reward -= 0.2 * (velocity - target_speed)
                elif velocity >= 5.0:
                    reward += 0.25

            # 4. STALL PENALTY

            moving_threshold = 0.5

            if velocity < moving_threshold and not agent_info.get("arrive_dest", False):
                self._stall_steps[agent_id] = self._stall_steps.get(agent_id, 0) + 1
            else:
                self._stall_steps[agent_id] = 0

            if self._stall_steps[agent_id] > 12:
                reward -= 0.08 * min(self._stall_steps[agent_id] - 6, 20)

            # 5. SAFETY PENALTIES
            if agent_info.get("crash", False):
                reward -= 30.0
            if agent_info.get("crash_sidewalk", False):
                reward -= 3.0
            if agent_info.get("road_line_solid_single_yellow", False):
                reward -= 5.0

            if agent_info.get("out_of_road", False):
                reward -= 40.0
            speed = agent_info.get("velocity", 0.0)
            steer = abs(agent_info.get("steering", 0.0))
            reward -= 0.08 * speed * steer
            # 6. COOPERATIVE BEHAVIOR
            if hasattr(self.env, "agents") and len(self.env.agents) > 1:
                try:
                    current_vehicle = self.env.agents[agent_id]
                    current_pos = current_vehicle.position

                    min_distance = float("inf")

                    for other_id, other_vehicle in self.env.agents.items():
                        if other_id != agent_id:
                            other_pos = other_vehicle.position
                            distance = np.linalg.norm(current_pos - other_pos)
                            min_distance = min(min_distance, distance)

                    if min_distance < 5.0:
                        reward -= 0.25 * (5.0 - min_distance)
                    if min_distance < 3.0 and 0.5 < velocity < 3.0:
                        reward -= 2.0 * (3.0 - min_distance)
                    if min_distance < 3.0 and velocity < 0.5:
                        reward += 0.3
                except Exception:
                    pass

            rewards_dict[agent_id] = float(np.clip(reward, -50.0, 50.0))

        return rewards_dict

    def step(self, action_dict):
        obs, rew, terminated, truncated, info = self.env.step(action_dict)

        if "__all__" not in terminated:
            active_keys = [k for k in terminated if k != "__all__"]
            terminated["__all__"] = all(terminated[k] for k in active_keys) if active_keys else True

        if "__all__" not in truncated:
            active_keys = [k for k in truncated if k != "__all__"]
            truncated["__all__"] = all(truncated[k] for k in active_keys) if active_keys else True

        self._agent_ids = set(obs.keys())

        custom_rew = self._custom_reward(obs, info)

        final_rew = {}
        for agent_id in custom_rew:
            base = rew.get(agent_id, 0.0)  # what this basically does is combine our custom reward with meta drives pre built lane centering reward
            final_rew[agent_id] = base + custom_rew[agent_id]

        return obs, final_rew, terminated, truncated, info


    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()


def build_env_config(num_agents: int, render: bool, stage: int = 1) -> dict:
    base = {
        "use_render": render,
        "stage": int(stage),
        "manual_control": False,
        "log_level": 50,
        "num_agents": num_agents,
        "horizon": 500,
        "use_lateral_reward": True,
        "allow_respawn": True,
        "traffic_density": 0.0,
        "vehicle_config": {
            "enable_reverse": False,
            "show_navi_mark": True,
        },
        "start_seed": 1000,
        "num_scenarios": 1000,
    }

    if stage == 1:
        base.update({
            "random_lane_width": False,
            "random_lane_num": False,
            "random_agent_model": False,
            "traffic_density": 0.0,
            "traffic_mode": "trigger",
            "allow_respawn": False,
            "horizon": 250,
        })

    elif stage == 2:
        base.update({
            "random_lane_width": True,
            "random_lane_num": True,
            "random_agent_model": False,
            "traffic_density": 0.2,
            "traffic_mode": "trigger",
            "allow_respawn": False,
            "horizon": 300,
        })

    elif stage == 3:
        base.update({
            "random_lane_width": True,
            "random_lane_num": True,
            "random_agent_model": False,
            "traffic_density": 0.3,
            "traffic_mode": "trigger",
            "allow_respawn": True,
            "horizon": 500,
        })

    return base


def make_env(config: dict):
    return RLLibMetaDriveEnv(config)