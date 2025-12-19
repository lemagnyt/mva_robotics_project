from sb3_contrib import TRPO
from envs.self_righting_env import SelfRightingEnv
from envs.standing_env import StandingEnv
from envs.locomotion_env import LocomotionEnv
from envs.behavior_selector_env import BehaviorSelectorEnv
from time import sleep

def create_trpo_policy(env_class, save_path=None, device="cpu", policy_kwargs=None, tensorboard_log="./trpo_logs/"):
    """
    Creates and returns a TRPO model on the provided environment
    """
    env = env_class()
    env.reset()
    sleep(2)  # wait for the environment to be properly initialized

    # If policy_layers is defined, create a kwargs dictionary for the policy MLP
    policy_kwargs = {} if policy_kwargs is None else policy_kwargs

    model = TRPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=2048,
        device=device,
        policy_kwargs=policy_kwargs, # architecture of the policy network
        gae_lambda=0.97,
        gamma=0.995,
        tensorboard_log=tensorboard_log # path for Tensorboard logs
    )

    if save_path:
        model.save(save_path)

    return model, env


def load_policy(path, env_class, device="cpu"):
    """
    Loads a TRPO model from the given path on the provided environment.
    """
    env = env_class()
    model = TRPO.load(path, env=env, device=device)
    return model, env
