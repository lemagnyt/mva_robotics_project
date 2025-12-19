from envs.standing_env import StandingEnv
from policies.create_policies import create_trpo_policy
from utils.robot_loader import RobotLoader
import os
import torch

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    gym_env = RobotLoader(
        env_name="Ant-v5",
        xml_file="./mujoco_menagerie/anybotics_anymal_b/scene.xml",
        render_mode=None, # put 'human' to visualize
        frame_skip=5 # 100 Hz
    ).load_robot()
    
    policy_kwargs = dict(
        net_arch=[128, 128],   # two layers of 128 units
        activation_fn=torch.nn.Tanh  # activation function for the hidden layers
    )

    # Create the TRPO policy on CPU
    sr_model, sr_env = create_trpo_policy(lambda: StandingEnv(gym_env, obs_dim=165), device="cpu", policy_kwargs=policy_kwargs)
    total_timesteps = 500000
    sr_model.learn(total_timesteps=total_timesteps, progress_bar=True)

    sr_model.save("models/standing_policy")
    print("Model saved successfully!")
    sr_env.close()