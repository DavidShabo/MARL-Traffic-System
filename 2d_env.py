from metadrive.envs import BaseEnv
from metadrive.obs.observation_base import DummyObservation
import logging

class MyEnv(BaseEnv):

    def reward_function(self, agent):
        return 0, {}

    def cost_function(self, agent):
        return 0, {}

    def done_function(self, agent):
        return False, {}
    
    def get_single_observation(self):
        return DummyObservation()
        

if __name__=="__main__":
    # create env
    env=MyEnv(dict(use_render=False, # if you have a screen and OpenGL suppor, you can set use_render=True to use 3D rendering  
                   manual_control=True, # we usually manually control the car to test environment
                   log_level=logging.CRITICAL)) # suppress logging message
    env.reset()
    for i in range(20):
        
        # step
        obs, reward, termination, truncate, info = env.step(env.action_space.sample())
        
        # you can set window=True and remove generate_gif() if you have a screen. 
        # Or just use 3D rendering and remove all stuff related to env.render()  
        frame=env.render(mode="topdown", 
                         window=False, # turn me on, if you have screen
                         screen_record=True, # turn me off, if a window can be poped up
                         screen_size=(200, 200))
    env.top_down_renderer.generate_gif()
    env.close()

from IPython.display import Image
Image(open("demo.gif", 'rb').read())
