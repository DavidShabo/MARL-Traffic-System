import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from metadrive import MultiAgentRoundaboutEnv
from metadrive import MultiAgentIntersectionEnv
from metadrive import MultiAgentTollgateEnv
# from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive import MultiAgentMetaDrive   

CUSTOM_MAP = "SSC"

ENV_REGISTRY = {
    "roundabout": MultiAgentRoundaboutEnv,
    "intersection": MultiAgentIntersectionEnv,
    "tollgate": MultiAgentTollgateEnv,
    "custom": MultiAgentMetaDrive,
}

_VEH_HALF         = 2.3
_BUMPER_GAP       = 0.2
_C2C_STOP         = 2 * _VEH_HALF + _BUMPER_GAP    # 4.8 m — enter front/rear recovery
_MANEUVER_CLEAR   = 8.0                            # m — exit front/rear recovery
_SIDE_WARN        = 0.8                            # 0.2 m — enter lateral recovery
_SIDE_CLEAR       = 1.5                            # 1.0 m — exit lateral recovery (safe gap)
_TARGET_REV_SPEED = 0.8                            # m/s — reverse recovery speed
_TARGET_FWD_SPEED = 0.8                            # m/s — forward recovery speed
_MAX_STEER_DEG    = 40.0                           # MetaDrive raw steering degrees


class RLLibMetaDriveEnv(MultiAgentEnv):

    def __init__(self, config: dict):
        cfg      = dict(config)
        env_name = cfg.pop("env_name", "roundabout")
        env_cls  = ENV_REGISTRY[env_name]

        self.stage      = int(cfg.pop("stage", 1))
        self._base_seed = int(cfg.pop("start_seed", 0))

        self.env = env_cls(cfg)

        self.observation_space = self.env.observation_space
        self.action_space      = self.env.action_space

        self._episode_idx    = 0
        self._prev_progress  = {}
        self._prev_steer     = {}
        self._stall_steps    = {}
        self._boundary_steps = {}
        # Recovery modes:
        #   "reverse"      — front blocked, back up straight
        #   "forward"      — rear blocked, move forward straight
        #   "lateral_left" — right solid line close, steer left
        #   "lateral_right"— left solid line close, steer right
        #   "boxed"        — front AND rear blocked
        #   None           — normal driving
        self._recovery_mode = {}

        self._agent_ids = (
            set(self.env.agents.keys()) if hasattr(self.env, "agents") else set()
        )
        self.metadata = {}

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        base = int(getattr(self, "_base_seed", 0))
        idx  = int(getattr(self, "_episode_idx", 0))
        n    = int(getattr(self.env, "num_scenarios", 1000))

        if seed is None:
            seed = (base + idx) % n

        self._episode_idx = idx + 1
        obs, info = self.env.reset(seed=seed)

        self._prev_progress  = {}
        self._prev_steer     = {}
        self._stall_steps    = {}
        self._boundary_steps = {}
        self._recovery_mode  = {}
        self._agent_ids      = set(obs.keys())

        return obs, info

    # ------------------------------------------------------------------
    def _get_directional_distances(self, agent_id):
        """
        Returns (front_dist, rear_dist) — nearest agent in each ±45° cone.
        """
        if not (hasattr(self.env, "agents") and len(self.env.agents) > 1):
            return None, None
        try:
            vehicle = self.env.agents.get(agent_id)
            if vehicle is None:
                return None, None

            pos     = np.array(vehicle.position, dtype=float)
            heading = float(vehicle.heading_theta)
            fwd     = np.array([np.cos(heading), np.sin(heading)])

            front_dist = float("inf")
            rear_dist  = float("inf")

            for other_id, other_veh in self.env.agents.items():
                if other_id == agent_id:
                    continue
                delta = np.array(other_veh.position, dtype=float) - pos
                d     = float(np.linalg.norm(delta))
                if d < 0.01:
                    continue
                fwd_proj = float(np.dot(delta, fwd))
                lateral  = float(np.sqrt(max(0.0, d * d - fwd_proj * fwd_proj)))

                if abs(fwd_proj) >= lateral:
                    if fwd_proj > 0:
                        front_dist = min(front_dist, d)
                    else:
                        rear_dist  = min(rear_dist, d)

            return (
                front_dist if front_dist != float("inf") else None,
                rear_dist  if rear_dist  != float("inf") else None,
            )
        except Exception:
            return None, None

    # ------------------------------------------------------------------
    def _get_side_distances(self, agent_id):
        """
        Returns (left_dist, right_dist) in meters from the vehicle's
        edge to the nearest road boundary on each side.
        Uses MetaDrive's built-in dist_to_left_side / dist_to_right_side.
        Returns (None, None) if unavailable.
        """
        try:
            vehicle = self.env.agents.get(agent_id)
            if vehicle is None:
                return None, None
            left  = getattr(vehicle, "dist_to_left_side",  None)
            right = getattr(vehicle, "dist_to_right_side", None)
            left  = float(left)  if left  is not None else None
            right = float(right) if right is not None else None
            return left, right
        except Exception:
            return None, None

    # ------------------------------------------------------------------
    def _get_axial_speed(self, agent_id):
        """
        Signed speed along heading axis.
        Positive = forward, negative = reversing.
        """
        try:
            vehicle = self.env.agents.get(agent_id)
            if vehicle is None:
                return 0.0
            heading = float(vehicle.heading_theta)
            fwd     = np.array([np.cos(heading), np.sin(heading)])
            vel_vec = np.array(vehicle.velocity, dtype=float)
            return float(np.dot(vel_vec, fwd))
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    def _straight_recovery_reward(self, abs_steering, axial_speed,
                                   target_speed, direction):
        """
        Shared reward for straight-line front/rear recovery.
        direction: +1 = forward, -1 = reverse.
        Heavily penalises steering during recovery.
        """
        reward = 0.0

        # Straight-only enforcement
        reward -= 4.0 * abs_steering
        if abs_steering > 0.075:
            reward -= 3.0

        # Speed targeting
        actual_speed = axial_speed * direction
        speed_error  = abs(actual_speed - target_speed)
        reward += max(0.0, 1.5 - speed_error)

        if actual_speed > target_speed + 0.3:
            reward -= 2.5 * (actual_speed - (target_speed + 0.3))
        if actual_speed < 0.1:
            reward -= 1.0
        if actual_speed < -0.1:
            reward -= 3.0 * abs(actual_speed)

        return reward

    # ------------------------------------------------------------------
    def _lateral_recovery_reward(self, steering_norm, velocity,
                                  steer_direction, left_dist, right_dist):
        """
        Reward for steering away from a solid line while continuing forward.

        steer_direction: +1 = steer left (away from right wall)
                         -1 = steer right (away from left wall)

        The agent should:
          1. Steer toward steer_direction (positive reward for correct sign)
          2. Keep moving forward at moderate speed (not stop)
          3. Straighten out once clear
        """
        reward = 0.0

        # 1. Correct steering direction
        signed_steer = steering_norm * steer_direction  # positive = correct
        if signed_steer > 0.05:
            reward += 1.5 * signed_steer  # reward steering away
        elif signed_steer < -0.05:
            reward -= 3.0 * abs(signed_steer)  # penalize steering into wall

        # 2. Must keep moving forward (lateral recovery ≠ stop)
        if velocity >= 1.0:
            reward += 0.3
        elif velocity < 0.3:
            reward -= 1.0  # penalize stopping unnecessarily

        # 3. Penalize excessive speed (stay controlled during correction)
        if velocity > 4.0:
            reward -= 0.5 * (velocity - 4.0)

        return reward

    # ------------------------------------------------------------------
    def _custom_reward(self, obs, info):
        rewards_dict = {}

        for agent_id in obs.keys():
            agent_info = info.get(agent_id, {})
            reward     = 0.0

            velocity      = float(agent_info.get("velocity", 0.0))
            steering_raw  = float(agent_info.get("steering", 0.0))
            steering_norm = float(np.clip(steering_raw / _MAX_STEER_DEG, -1.0, 1.0))
            abs_steering  = abs(steering_norm)

            front_dist, rear_dist = self._get_directional_distances(agent_id)
            left_dist,  right_dist = self._get_side_distances(agent_id)
            axial_speed = self._get_axial_speed(agent_id)

            front_blocked = front_dist is not None and front_dist <= _C2C_STOP
            rear_blocked  = rear_dist  is not None and rear_dist  <= _C2C_STOP

            # Side line proximity flags
            left_too_close  = left_dist  is not None and left_dist  <= _SIDE_WARN
            right_too_close = right_dist is not None and right_dist <= _SIDE_WARN

            # ── RECOVERY STATE MACHINE ────────────────────────────────────
            current_mode = self._recovery_mode.get(agent_id, None)

            # Front/rear recovery takes priority over lateral
            if front_blocked and rear_blocked:
                current_mode = "boxed"

            elif front_blocked:
                current_mode = "reverse"

            elif rear_blocked:
                current_mode = "forward"

            # Exit front/rear recovery when enough room
            elif current_mode == "reverse":
                if front_dist is None or front_dist >= _MANEUVER_CLEAR:
                    current_mode = None

            elif current_mode == "forward":
                if rear_dist is None or rear_dist >= _MANEUVER_CLEAR:
                    current_mode = None

            # Lateral recovery (only when not already in front/rear recovery)
            if current_mode is None or current_mode in ("lateral_left", "lateral_right"):
                if right_too_close and not left_too_close:
                    # Right wall close → steer left
                    current_mode = "lateral_left"
                elif left_too_close and not right_too_close:
                    # Left wall close → steer right
                    current_mode = "lateral_right"
                elif left_too_close and right_too_close:
                    # Both sides close (narrow passage) → go straight forward
                    current_mode = "lateral_left"  # neutral: slight left bias
                elif current_mode in ("lateral_left", "lateral_right"):
                    # Check exit condition: both sides clear
                    l_clear = left_dist  is None or left_dist  >= _SIDE_CLEAR
                    r_clear = right_dist is None or right_dist >= _SIDE_CLEAR
                    if l_clear and r_clear:
                        current_mode = None

            self._recovery_mode[agent_id] = current_mode
            in_recovery = current_mode is not None

            # ── 1. PROGRESS ───────────────────────────────────────────────
            cur_prog   = float(agent_info.get("route_completion", 0.0))
            prev_prog  = self._prev_progress.get(agent_id, 0.0)
            delta_prog = cur_prog - prev_prog
            self._prev_progress[agent_id] = cur_prog

            # Suppress progress during front/rear recovery (moving backwards)
            # Allow progress during lateral recovery (still moving forward)
            if current_mode not in ("reverse", "boxed"):
                reward += 5.0 * delta_prog

            # ── 2. ARRIVAL ────────────────────────────────────────────────
            if agent_info.get("arrive_dest", False):
                reward += 25.0

            # ── 3. RECOVERY BEHAVIORS ─────────────────────────────────────
            if current_mode == "reverse":
                reward += self._straight_recovery_reward(
                    abs_steering, axial_speed, _TARGET_REV_SPEED, direction=-1
                )

            elif current_mode == "forward":
                reward += self._straight_recovery_reward(
                    abs_steering, axial_speed, _TARGET_FWD_SPEED, direction=+1
                )

            elif current_mode == "boxed":
                if velocity <= 0.3:
                    reward += 1.0
                else:
                    reward -= 2.0 * velocity

            elif current_mode == "lateral_left":
                # Right wall close → steer left (+1 direction)
                reward += self._lateral_recovery_reward(
                    steering_norm, velocity, +1, left_dist, right_dist
                )

            elif current_mode == "lateral_right":
                # Left wall close → steer right (-1 direction)
                reward += self._lateral_recovery_reward(
                    steering_norm, velocity, -1, left_dist, right_dist
                )

            # ── 4. NORMAL SPEED LIMITS ────────────────────────────────────
            else:
                if abs_steering < 0.2:
                    base_target = 7.0
                elif abs_steering < 0.5:
                    base_target = 3.0
                else:
                    base_target = 1.5

                if front_dist is not None:
                    if front_dist <= 3.0:
                        prox = 0.0
                    elif front_dist < 10.0:
                        prox = (front_dist - 3.0) / 7.0
                    else:
                        prox = 1.0
                else:
                    prox = 1.0

                target_speed = base_target * float(np.clip(prox, 0.0, 1.0))

                if velocity > target_speed:
                    reward -= 0.8 * (velocity - target_speed)
                elif target_speed > 0.5 and 0.5 * target_speed <= velocity <= 1.1 * target_speed:
                    reward += 0.05
                    
                if abs_steering < 0.05:
                    reward += 0.1

            # ── 5. STEERING SMOOTHNESS (normal driving only) ──────────────
            if current_mode is None:
                reward -= 0.20 * abs_steering                          # was 0.08 — punish any steering harder
                prev_steer = self._prev_steer.get(agent_id, steering_norm)
                steer_change = abs(steering_norm - prev_steer)
                reward -= 0.15 * steer_change                          # was 0.03 — punish direction changes much harder
                self._prev_steer[agent_id] = steering_norm
                reward -= 0.25 * velocity * abs_steering               # was 0.15 — punish steering at speed harder
            else:
                self._prev_steer[agent_id] = steering_norm

            # ── 6. SOLID-LINE / OUT-OF-ROAD PENALTIES ─────────────────────
            # These fire when agent has ALREADY crossed — backup catch-all.
            # Lateral recovery above handles the pre-emptive 0.2m case.
            out_of_road   = agent_info.get("out_of_road", False)
            on_solid_line = (
                agent_info.get("on_yellow_continuous_line", False) or
                agent_info.get("on_white_continuous_line",  False)
                # "on_lane_line" excluded — fires on dotted lines too
            )
            boundary_hit = out_of_road or on_solid_line
            prev_bsteps  = self._boundary_steps.get(agent_id, 0)

            if boundary_hit:
                self._boundary_steps[agent_id] = prev_bsteps + 1
            else:
                self._boundary_steps[agent_id] = max(0, prev_bsteps - 1)

            bsteps = self._boundary_steps[agent_id]

            if bsteps > 0:
                reward -= 0.4 * min(bsteps, 10)
                if prev_bsteps == 0:
                    reward -= 35.0 if out_of_road else 10.0
                if velocity <= 0.5:
                    reward += 0.8
                else:
                    reward -= 0.5 * velocity

            # ── 7. STALL PENALTY ──────────────────────────────────────────
            if not in_recovery and bsteps == 0:
                if velocity < 0.5 and not agent_info.get("arrive_dest", False):
                    self._stall_steps[agent_id] = self._stall_steps.get(agent_id, 0) + 1
                else:
                    self._stall_steps[agent_id] = 0
                if self._stall_steps[agent_id] > 6:
                    reward -= 0.08 * min(self._stall_steps[agent_id] - 6, 20)
            else:
                self._stall_steps[agent_id] = 0

            # ── 8. CRASH PENALTY ──────────────────────────────────────────
            if agent_info.get("crash", False):
                reward -= 30.0
            if agent_info.get("crash_sidewalk", False):
                reward -= 10.0

            # ── 9. COOPERATIVE SPACING ────────────────────────────────────
            if current_mode is None:
                overall_min = None
                if front_dist is not None and rear_dist is not None:
                    overall_min = min(front_dist, rear_dist)
                elif front_dist is not None:
                    overall_min = front_dist
                elif rear_dist is not None:
                    overall_min = rear_dist

                if overall_min is not None:
                    if overall_min < 3.0:
                        reward -= 1.0 * (3.0 - overall_min)
                    elif 8.0 <= overall_min <= 12.0:
                        reward += 0.05

            rewards_dict[agent_id] = float(np.clip(reward, -30.0, 30.0))

        return rewards_dict

    # ------------------------------------------------------------------
    def step(self, action_dict):
        obs, rew, terminated, truncated, info = self.env.step(action_dict)

        if "__all__" not in terminated:
            active_keys = [k for k in terminated if k != "__all__"]
            terminated["__all__"] = (
                all(terminated[k] for k in active_keys) if active_keys else True
            )

        if "__all__" not in truncated:
            active_keys = [k for k in truncated if k != "__all__"]
            truncated["__all__"] = (
                all(truncated[k] for k in active_keys) if active_keys else True
            )

        self._agent_ids = set(obs.keys())
        custom_rew = self._custom_reward(obs, info)

        final_rew = {}
        for agent_id in custom_rew:
            base = rew.get(agent_id, 0.0)
            final_rew[agent_id] = base + custom_rew[agent_id]

        return obs, final_rew, terminated, truncated, info

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()


# ──────────────────────────────────────────────────────────────────────
def build_env_config(num_agents: int, render: bool, stage: int = 1, env_name: str = "roundabout") -> dict:
    base = {
        "use_render":         render,
        "stage":              int(stage),
        "manual_control":     False,
        "log_level":          50,
        "num_agents":         num_agents,
        "horizon":            500,
        "use_lateral_reward": True,
        "allow_respawn":      True,
        "traffic_density":    0.0,
        "vehicle_config": {
            "enable_reverse": True,
            "show_navi_mark": True,
            # Enable side detector so MetaDrive computes dist_to_left_side
            # and dist_to_right_side every step
            # "side_detector": {
            #     "num_lasers": 20,
            #     "distance":   20,
            # },
        },
        "start_seed":    1000,
        "num_scenarios": 1000,
    }
    
    # NEW: if env_name is set to "custom", use a simple fixed map
    # env_name = base.get("env_name", None)
    
    if env_name == "custom":
        # Simple: straight → straight → curve → straight
        # base.update({
        #     "map": CUSTOM_MAP,
        #     "traffic_density": 0.0,
        #     "allow_respawn": False,
        #     "random_lane_width": False,
        #     "random_lane_num": False,
        #     "random_agent_model": False,
        #     "traffic_mode": "trigger",
        #     "horizon": 300,
        # })
        
        base.update = ({
            "map": CUSTOM_MAP,
            "traffic_density": 0.0,
            "allow_respawn": False,
            "random_lane_width": False,
            "random_lane_num": False,
            "random_agent_model": False,
            "traffic_mode": "trigger",
            "horizon": 300,
        })
        return base

    if stage == 1:
        base.update({
            "random_lane_width":  False,
            "random_lane_num":    False,
            "random_agent_model": False,
            "traffic_density":    0.0,
            "traffic_mode":       "trigger",
            "allow_respawn":      False,
            "horizon":            250,
        })
    elif stage == 2:
        base.update({
            "random_lane_width":  True,
            "random_lane_num":    True,
            "random_agent_model": False,
            "traffic_density":    0.1,
            "traffic_mode":       "trigger",
            "allow_respawn":      False,
            "horizon":            300,
        })
    elif stage == 3:
        base.update({
            "random_lane_width":  True,
            "random_lane_num":    True,
            "random_agent_model": True,
            "traffic_density":    0.2,
            "traffic_mode":       "trigger",
            "allow_respawn":      True,
            "horizon":            500,
        })

    return base


def make_env(config: dict):
    return RLLibMetaDriveEnv(config)