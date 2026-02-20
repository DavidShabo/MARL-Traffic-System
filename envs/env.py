import numpy as np
from metadrive import MultiAgentRoundaboutEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv


class RLLibMetaDriveRoundabout(MultiAgentEnv):
    def __init__(self, config: dict):
        # Wrap MetaDrive's multi-agent roundabout environment for RLlib
        self.env = MultiAgentRoundaboutEnv(config)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        # Track each agent's progress to compute delta rewards
        self._prev_progress = {}
        self._agent_ids = set(self.env.agents.keys()) if hasattr(self.env, "agents") else set()
        self.metadata = {}

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed)
        # Clear progress history on reset
        self._prev_progress = {}
        self._agent_ids = set(obs.keys())
        return obs, info

    def _custom_reward(self, obs, info):
        """
        Custom reward implementing three key behaviors:
        1. Reward getting closer to destination (route_completion progress)
        2. Penalize crashes (objects or agents)
        3. Encourage slowing down when turning, speeding up on straights
        """
        rewards_dict = {}
        
        for agent_id in obs.keys():
            agent_info = info.get(agent_id, {})
            reward = 0.0

            ### 1. PROGRESS TO DESTINATION - Reward getting closer
            current_progress = agent_info.get("route_completion", 0.0)
            prev_progress = self._prev_progress.get(agent_id, 0.0)
            delta_progress = current_progress - prev_progress
            self._prev_progress[agent_id] = current_progress
            
            reward += 10.0 * delta_progress  # Primary reward: moving toward destination

            ### 2. DESTINATION ARRIVAL - Bonus for completing the route
            if agent_info.get("arrive_dest", False):
                reward += 20.0

            ### 3. SPEED & STEERING CONTROL - Slow in real turns, penalize zigzag
            velocity = float(agent_info.get("velocity", 0.0))
            steering = float(agent_info.get("steering", 0.0))
            abs_steering = abs(steering)

            # Penalize unnecessary steering (zigzag) always
            reward -= 0.2 * abs_steering  # Small penalty for any steering

            # Only reward turn speed if actually turning significantly
            if abs_steering > 0.3:  # True turning
                target_speed = 4.0
                if velocity > target_speed:
                    reward -= 0.5 * (velocity - target_speed)
                elif velocity > 2.0:
                    reward += 0.2
            else:  # Straight driving
                target_speed = 10.0
                if velocity > target_speed:
                    reward -= 0.3 * (velocity - target_speed)
                elif velocity >= 6.0:
                    reward += 0.3

            # Always penalize complete stops (blocking traffic)
            if velocity < 0.5:
                reward -= 0.15

            ### 4. SAFETY PENALTIES - Don't crash into objects or agents
            if agent_info.get("crash", False):
                reward -= 10.0
            
            if agent_info.get("out_of_road", False):
                reward -= 8.0

            ### 5. COOPERATIVE BEHAVIOR - Penalty for being too close to other agents
            # Get positions of all agents to compute inter-agent distances
            if hasattr(self.env, "agents") and len(self.env.agents) > 1:
                try:
                    current_vehicle = self.env.agents[agent_id]
                    current_pos = current_vehicle.position
                    
                    # Check distance to all other agents
                    min_distance = float('inf')
                    for other_id, other_vehicle in self.env.agents.items():
                        if other_id != agent_id:
                            other_pos = other_vehicle.position
                            distance = np.linalg.norm(current_pos - other_pos)
                            min_distance = min(min_distance, distance)
                    
                    # Penalize if too close to another agent (< 5 meters)
                    if min_distance < 5.0:
                        reward -= 0.5 * (5.0 - min_distance)  # Increases as agents get closer
                except:
                    # If position access fails, skip cooperative penalty
                    pass

            # Clip final reward to reasonable range
            rewards_dict[agent_id] = float(np.clip(reward, -15.0, 25.0))

        return rewards_dict

    def step(self, action_dict):
        obs, rew, terminated, truncated, info = self.env.step(action_dict)

        # Use terminated/truncated keys (not obs.keys()) to detect episode end.
        # MetaDrive removes crashed agents from obs, so obs can be empty mid-episode,
        # causing all([]) = True (vacuous truth) and ending the episode prematurely.
        if "__all__" not in terminated:
            active_keys = [k for k in terminated if k != "__all__"]
            terminated["__all__"] = all(terminated[k] for k in active_keys) if active_keys else True

        if "__all__" not in truncated:
            active_keys = [k for k in truncated if k != "__all__"]
            truncated["__all__"] = all(truncated[k] for k in active_keys) if active_keys else True

        # Update known agent IDs (respawned agents may have new IDs)
        self._agent_ids = set(obs.keys())

        custom_rew = self._custom_reward(obs, info)
        return obs, custom_rew, terminated, truncated, info

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
        "horizon": 500,              # Long enough to complete full roundabout route
        "use_lateral_reward": True,  # Enables lateral_factor in step_reward → lane centering + turn awareness
        "allow_respawn": True,       # Crashed agents respawn → more training signal per episode
        "traffic_density": 0.0,      # No background traffic during early training (reduce noise)
        "vehicle_config": {
            "enable_reverse": False, # Prevent reversing
            "show_navi_mark": True,  # Visual nav arrow (helpful for debugging in render mode)
        },
    }


def make_env(config: dict):
    return RLLibMetaDriveRoundabout(config)
