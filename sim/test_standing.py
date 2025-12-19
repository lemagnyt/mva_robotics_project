from envs.standing_env import StandingEnv
from policies.create_policies import load_policy
from utils.robot_loader import RobotLoader
import imageio
import os



if __name__ == "__main__":

    # create directory to save images if it doesn't exist
    os.makedirs("images", exist_ok=True)
    
    # Load robot
    gym_env = RobotLoader(env_name="Ant-v5", xml_file="./mujoco_menagerie/anybotics_anymal_b/scene.xml", render_mode="human", frame_skip=5).load_robot()


    # Load pre-trained specialized policies 
    model, env = load_policy("models/standing_policy", lambda: StandingEnv(gym_env, obs_dim=165, use_stif=False, height_estimator=None, max_steps=35))

    env.reset()
    time_sr_elapsed = 2
    obs = env.get_observation(None)
    for t in range(1000):
        
        
        # Save images every 5 steps, put render_mode to "rgb_array" in RobotLoader
        '''if t%5 == 0 :
            # Retrieve the RGB image
            frame = current_env.env.render()

            # Save the image
            imageio.imwrite(f"images/frame_{int(t/5):04d}.png", frame)'''

        # Predict action
        action, _ = model.predict(obs)

        obs, reward, done, truncated, _ = env.step(action)
        print(f"Step {t},  Reward {reward:.2f}, Done {done}")

        if env.get_terminated(obs, action, done) or truncated:
            env.reset()