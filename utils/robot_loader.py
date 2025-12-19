import gymnasium as gym
import numpy as np

class RobotLoader:
    """
    Class to load a robot environment from Gym with specified parameters.
    """
    def __init__(self, env_name="Ant-v5", xml_file=None, render_mode=None, drop_height=0.5, joint_noise_scale=0.3, frame_skip=5):
        self.env_name = env_name
        self.xml_file = xml_file
        self.render_mode = render_mode
        self.drop_height = drop_height
        self.joint_noise_scale = joint_noise_scale
        self.env = None
        self.frame_skip = frame_skip

    def reset_with_random_pose(self, drop_height=0.5, joint_noise_scale=0.3):
        """
        Reset environment using a random pose for the robot.
        """
        data = self.env.unwrapped.data
        qpos = np.copy(data.qpos)
        qpos[0:3] = np.array([0.0, 0.0, drop_height])
        qpos[7:] += np.random.uniform(-joint_noise_scale, joint_noise_scale, 
                                      size=qpos[7:].shape)
        qvel = np.zeros_like(data.qvel)
        self.env.unwrapped.set_state(qpos, qvel)
        obs, info = self.env.reset()
        return obs, info

    def load_robot(self, **kwargs):
        """
        Load the robot environment from Gym with specified parameters.
        """
        if self.xml_file:
            self.env = gym.make(self.env_name, xml_file=self.xml_file, render_mode=self.render_mode, frame_skip=self.frame_skip, **kwargs)
        else :
            print( "No XML file provided, loading default environment." )
            self.env = gym.make(self.env_name, render_mode=self.render_mode, **kwargs)
        return self.env
