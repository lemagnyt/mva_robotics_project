from envs.self_righting_env import SelfRightingEnv
from policies.create_policies import create_trpo_policy
from utils.robot_loader import RobotLoader
import os
import torch

def make_env():
    gym_env = RobotLoader(
        env_name="Ant-v5",
        xml_file="./mujoco_menagerie/anybotics_anymal_b/scene.xml",
        render_mode=None,
        frame_skip=25
    ).load_robot()

    return SelfRightingEnv(
        gym_env,
        obs_dim=162
    )

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    gym_env = RobotLoader(
        env_name="Ant-v5",
        xml_file="./mujoco_menagerie/anybotics_anymal_b/scene.xml",
        render_mode = None, # put 'human' to visualize
        frame_skip=25 # 50 hz
    ).load_robot()
    
    policy_kwargs = dict(
        net_arch=[128, 128],   # two layers of 128 units
        activation_fn=torch.nn.Tanh  # activation function for the hidden layers
    )

    policy_kwargs = None
    
    # Create the TRPO policy on CPU
    sr_model, sr_env = create_trpo_policy(make_env, device="cpu", policy_kwargs=policy_kwargs)

    total_timesteps = 1000000
    sr_model.learn(total_timesteps=total_timesteps, progress_bar=True)

    sr_model.save("models/self_righting_policy")
    print("Model saved successfully!")
    sr_env.close()