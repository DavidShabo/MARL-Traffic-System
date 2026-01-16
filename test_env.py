from metadrive import MultiAgentRoundaboutEnv

env = MultiAgentRoundaboutEnv({
    "use_render": True,
    "manual_control": False,
    "log_level": 50,
    "num_agents": 8,
})

obs, info = env.reset(seed=0)

try:
    while True:
        env.render(mode="human")            
        actions = env.action_space.sample()  
        obs, rew, term, trunc, info = env.step(actions)

        if term.get("__all__", False) or trunc.get("__all__", False):
            obs, info = env.reset()
finally:
    env.close()
