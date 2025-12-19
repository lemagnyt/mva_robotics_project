from envs.self_righting_env import SelfRightingEnv
from envs.standing_env import StandingEnv
from envs.locomotion_env import LocomotionEnv
from envs.behavior_selector_env import BehaviorSelectorEnv
from policies.create_policies import load_policy
from utils.robot_loader import RobotLoader
import torch
from train.train_behavior_selector import HeightEstimator
from time import time
import os
import imageio



if __name__ == "__main__":

    # create directory to save images if it doesn't exist
    os.makedirs("images", exist_ok=True)
    
    # Load robot
    gym_env = RobotLoader(env_name="Ant-v5", xml_file="./mujoco_menagerie/anybotics_anymal_b/scene.xml", render_mode="human", frame_skip=10).load_robot()

    # Create specialized envs with this robot
    sr_env = SelfRightingEnv(gym_env, obs_dim=162)
    st_env = StandingEnv(gym_env, obs_dim=165)
    lo_env = LocomotionEnv(gym_env, obs_dim=169)


    use_stif = False
    use_estimator = False
    estimator_model = None

    if use_estimator and os.path.exists("models/h_psi.pth"):
        # Initialize height estimator
        estimator_model = HeightEstimator(BehaviorSelectorEnv(sr_env, st_env, lo_env, obs_dim=172).get_estimator_observation()[1].shape[0])

        # Load weights of height estimator
        state_dict = torch.load("models/h_psi.pth", map_location="cpu")
        estimator_model.load_state_dict(state_dict)

        # Evaluate mode
        estimator_model.eval()


    # Load pre-trained specialized policies
    sr_model, sr_env = load_policy("models/self_righting_policy", lambda: SelfRightingEnv(gym_env, obs_dim=162, use_stif=use_stif, height_estimator=estimator_model, max_steps=100))
    st_model, st_env = load_policy("models/standing_policy", lambda: StandingEnv(gym_env, obs_dim=165, use_stif=use_stif, height_estimator=estimator_model))
    lo_model, lo_env = load_policy("models/locomotion_policy", lambda: LocomotionEnv(gym_env, obs_dim=169, use_stif=use_stif, height_estimator=estimator_model))

    envs = [sr_env, st_env, lo_env]
    models = [sr_model, st_model, lo_model]

    selector_action = 2
    current_env = envs[selector_action]
    current_model = models[selector_action]
    current_env.reset()
    time_sr_elapsed = 2
    for t in range(1000):

        X_angle = current_env.get_base_z_angle_deg_quat()
        base_height = current_env.env.unwrapped.data.qpos[2]
        
        # Save images every 5 steps, put render_mode to "rgb_array" in RobotLoader
        '''if t%5 == 0 :
            # Retrieve the RGB image
            frame = current_env.env.render()

            # Save the image
            imageio.imwrite(f"images/frame_{int(t/5):04d}.png", frame)'''

        if selector_action == 0 :
            if not time_sr_elapsed :
                time_sr_elapsed = 0
            if abs(X_angle) < 20 and time_sr_elapsed > 0.5 :
                selector_action = 1
                time_sr_elapsed = None
            else : time_sr_elapsed += current_env.dt
        elif selector_action == 1 :
            if base_height > 0.35 :
                selector_action = 2
            elif X_angle > 35 or base_height<0.2 :
                selector_action = 1
        else :
            if X_angle > 35 or base_height<0.2 :
                selector_action = 0

        current_env = envs[selector_action]
        current_model = models[selector_action]
        
        obs = current_env.get_observation(None)

        # Predict action with the current specialized policy
        action, _ = current_model.predict(obs)

        obs, reward, done, truncated, _ = current_env.step(action)
        print(f"Step {t}, Behavior {selector_action}, Reward {reward:.2f}, Done {done}")

        if truncated:
            current_env.reset()
