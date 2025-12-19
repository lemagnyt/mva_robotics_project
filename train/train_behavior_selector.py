import os
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

from envs.behavior_selector_env import BehaviorSelectorEnv
from envs.self_righting_env import SelfRightingEnv
from envs.standing_env import StandingEnv
from envs.locomotion_env import LocomotionEnv
from utils.robot_loader import RobotLoader
from policies.create_policies import create_trpo_policy

from stable_baselines3.common.callbacks import BaseCallback


# Replay Memory for Height Estimator

class ReplayMemory:
    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)

    def push(self, state, height):
        self.memory.append((state, height))

    def sample(self, batch_size):
        idx = np.random.choice(len(self.memory), batch_size, replace=False)
        states, heights = zip(*[self.memory[i] for i in idx])
        return np.array(states), np.array(heights)

    def __len__(self):
        return len(self.memory)



# Height Estimator h_psi

class HeightEstimator(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Softsign(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softsign(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)


# Callback to train Height Estimator during TRPO rollouts

class HeightEstimatorCallback(BaseCallback):
    def __init__(
        self,
        h_psi,
        optimizer,
        criterion,
        replay_memory,
        batch_size=64,
        warmup_steps=5000,
        verbose=0
    ):
        super().__init__(verbose)
        self.h_psi = h_psi
        self.optimizer = optimizer
        self.criterion = criterion
        self.memory = replay_memory
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps

    def _on_step(self) -> bool:
        # Accès à l'environnement réel
        env = self.training_env.envs[0].unwrapped

        # Récupération de l'observation et de la vraie hauteur
        ht, estimator_obs = env.get_estimator_observation()

        # Stockage dans le replay buffer
        self.memory.push(estimator_obs, ht)

        # Warm-up: pas d'apprentissage
        if self.num_timesteps < self.warmup_steps:
            return True

        # Entraînement off-policy
        if len(self.memory) >= self.batch_size:
            states, heights = self.memory.sample(self.batch_size)

            states = torch.FloatTensor(states)
            heights = torch.FloatTensor(heights).unsqueeze(1)

            self.optimizer.zero_grad()
            loss = self.criterion(self.h_psi(states), heights)
            loss.backward()
            self.optimizer.step()

        return True



# Main Training Script
if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    # Load robot
    gym_env = RobotLoader(
        env_name="Ant-v5",
        xml_file="./mujoco_menagerie/anybotics_anymal_b/scene.xml",
        render_mode=None,
        frame_skip=10 # 50 hz
    ).load_robot()

    # Low-level environments
    sr_env = SelfRightingEnv(gym_env, obs_dim=162)
    st_env = StandingEnv(gym_env, obs_dim=165)
    lo_env = LocomotionEnv(gym_env, obs_dim=169)

    # Create Behavior Selector (TRPO)
    policy_kwargs = dict(
        net_arch=[128, 128],
        activation_fn=torch.nn.Tanh
    )

    selector_model, selector_env = create_trpo_policy(
        lambda: BehaviorSelectorEnv(sr_env, st_env, lo_env, obs_dim=172),
        device="cpu",
        policy_kwargs=policy_kwargs,
        tensorboard_log=None
    )

    # Initiate Height Estimator
    _, estimator_obs = selector_env.get_estimator_observation()
    input_dim = estimator_obs.shape[0]

    h_psi = HeightEstimator(input_dim)
    optimizer_h = optim.Adam(h_psi.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    memory = ReplayMemory(capacity=10000)

    total_timesteps = 100000

    # Initiate Callback
    height_callback = HeightEstimatorCallback(
        h_psi=h_psi,
        optimizer=optimizer_h,
        criterion=criterion,
        replay_memory=memory,
        batch_size=64,
        warmup_steps=5000
    )

    # Train Behavior Selector with Height Estimator Callback
    selector_model.learn(
        total_timesteps=total_timesteps,
        callback=height_callback,
        progress_bar=True
    )

    # Save models
    selector_model.save("models/behavior_selector_trpo")
    torch.save(h_psi.state_dict(), "models/height_estimator_h_psi.pth")

    selector_env.close()
    print("Training completed and models saved.")
