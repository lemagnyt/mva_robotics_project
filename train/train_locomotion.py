from envs.locomotion_env import LocomotionEnv
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
        frame_skip=3 # 200 Hz, to be exact we should use 2.5 but frame_skip can only be int
    ).load_robot()
    
    policy_kwargs = dict(
        net_arch=[128, 256],   # one layer of 128 units followed by one layer of 256 units
        activation_fn=torch.nn.Tanh  # activation function for the hidden layers
    )

    # Create the TRPO policy on CPU
    sr_model, sr_env = create_trpo_policy(lambda: LocomotionEnv(gym_env, obs_dim=169), device="cpu", policy_kwargs=policy_kwargs)

    total_timesteps = 100000 # too big value may lead to overfitting with robot that tries to terminate episode quickly by falling
    sr_model.learn(total_timesteps=total_timesteps, progress_bar=True)

    sr_model.save("models/locomotion_policy")
    print("Model saved successfully!")
    sr_env.close()