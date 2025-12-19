import gymnasium as gym
import numpy as np
from collections import deque
from sb3_contrib import TRPO
from utils.utils_functions import rotate_vector_by_quat, sample_velocity_command, logistic_kernel
import torch

class BehaviorSelectorEnv(gym.Env):
    """
    # Hierarchical environment that selects among SelfRighting, Standing, and Locomotion.
    """
    def __init__(self, sr_env, st_env, lo_env, max_steps=1000, history_len=5, action_dim=12, obs_dim=113, init_height=0.5, use_tsif=False, height_estimator=None):
        super().__init__()
        self.env = sr_env.env
        self.sr_env = sr_env
        self.st_env = st_env
        self.lo_env = lo_env
        self.sr_model = TRPO.load("models/self_righting_policy", env=sr_env, device="cpu")
        self.st_model = TRPO.load("models/standing_policy", env=st_env, device="cpu")
        self.lo_model = TRPO.load("models/locomotion_policy", env=lo_env, device="cpu")

        self.policies = [self.sr_model, self.st_model, self.lo_model]
        self.envs = [self.sr_env, self.st_env, self.lo_env]
        self.current_policy = None
        self.current_sub_env_obs = None

        # Global observation: simplification
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3)
        self.step_count = 0
        self.max_steps = max_steps

        self.history_len = history_len
        
        self.action_dim = action_dim
        self.velocity_command = sample_velocity_command()  # Initial velocity command

        ## ---- Buffers with default values ----
        self.joint_positions_history = deque([np.zeros(self.env.unwrapped.data.qpos[7:].shape)] * history_len, maxlen=history_len)
        self.joint_velocities_history = deque([np.zeros(self.env.unwrapped.data.qvel[6:].shape)] * history_len, maxlen=history_len)
        self.actions_history = deque([np.zeros(action_dim)] * history_len, maxlen=history_len)
        self.geom_positions_history = deque([np.copy(self.env.unwrapped.data.geom_xpos) for _ in range(history_len)], maxlen=history_len)
        
        self.foot_geom_ids = [45, 74, 103, 132]  # Foot IDs
        self.jslim = 40.0  # Joint speed limit
        self.dt = self.env.unwrapped.model.opt.timestep
        self.init_height = init_height

        self.prev_action = np.zeros(3) # One-hot vector

        # Reward coefficients
        self.K_orientation        = 1.7     # Maintain correct orientation (very important)
        self.K_action             = 0.001   # Low cost on action effort
        self.K_joint_limit        = 0.0003    # Penalty if joints reach their limits
        self.K_torque             = 0.0001  # Low penalty on high torques
        self.K_angular_velocity   = 1     # Follow angular velocity targets
        self.K_linear_velocity    = 2     # Follow linear velocity targets (essential for locomotion)
        self.alpha_linear  = 1   # Weight to calculate linear velocity cost
        self.alpha_angular = 0.5   # Weight to calculate angular velocity cost
        self.K_height             = 2.5      # Stable height = essential
        self.K_power              = 0.0001   # Power cost

        self.need_reset = True

        self.use_tsif = use_tsif
        self.angular_velocity_std = 0.2      # rad/s
        self.linear_velocity_std = 0.25  # m/s
        self.joint_position_std = 0.05       # rad
        self.joint_velocity_std = 0.5    # rad/s
        self.height_estimator = height_estimator
    

    def step(self, action):
        self.step_count += 1

        # One-hot previous action (for next obs)
        self.prev_action = np.zeros(3)
        self.prev_action[action] = 1.0

        # Select sub-policy
        policy = self.policies[action]
        sub_obs = self.envs[action].get_observation(None)
        policy_action, _ = policy.predict(sub_obs)

        self.last_policy_action = policy_action

        # Step the shared env
        obs_raw, _, terminated_env, truncated_env, info = self.env.step(policy_action)

        # Update buffers
        self._update_histories(policy_action)

        obs = self.get_observation(obs_raw)
        reward = self.get_reward(obs, action)

        terminated = terminated_env
        truncated = truncated_env or (self.step_count >= self.max_steps)

        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        self.step_count = 0
        self.need_reset = False
        self.env.reset(seed=seed)
        obs = self.get_observation(None)
        self.current_obs = obs
        self.prev_action = np.zeros(3)
        return obs, {}

    # -- Build the observation vector --
    def get_observation(self, obs_raw):
        data = self.env.unwrapped.data

        # --- Base state ---
        base_quaternion = data.qpos[3:7]
        base_angular_velocity = data.qvel[3:6]
        base_linear_velocity = data.qvel[0:3]
        base_height = data.qpos[2] # To change with base height estimation !!! (Done)
        angular_velocity = rotate_vector_by_quat(base_angular_velocity, base_quaternion)

        # --- Joint state ---
        joint_positions = data.qpos[7:]
        joint_velocities = data.qvel[6:]

        # --- Add noise if TSIF is used ---
        if self.use_tsif:
            angular_velocity += np.random.normal(0, self.angular_velocity_std * self.dt, len(angular_velocity))
            base_linear_velocity += np.random.normal(0, self.linear_velocity_std * self.dt, len(base_linear_velocity))
            joint_positions += np.random.normal(0, self.joint_position_std, len(joint_positions))
            joint_velocities += np.random.normal(0, self.joint_velocity_std * self.dt, len(joint_velocities))

        if self.height_estimator is not None:
            # Estimate height with the height estimator
            estimator_obs = self.get_estimator_observation()[1]
            estimator_obs_tensor = torch.FloatTensor(estimator_obs).unsqueeze(0)
            with torch.no_grad():
                estimated_height = self.height_estimator(estimator_obs_tensor).item()
            base_height = estimated_height

        # --- Gravity vector ---
        g_world = np.array([0, 0, -1])
        gravity_body = rotate_vector_by_quat(g_world, base_quaternion)

        # --- Build observation vector ---
        obs = np.concatenate([
            gravity_body,       
            angular_velocity,     
            joint_positions,       
            joint_velocities,
            np.array(self.joint_positions_history).flatten(),    
            np.array(self.joint_velocities_history).flatten(),        
            self.actions_history[-2],
            base_linear_velocity,   
            self.velocity_command, 
            np.array([base_height]), 
            self.prev_action 
        ])

        return obs.astype(np.float32)

    def get_estimator_observation(self):
        data = self.env.unwrapped.data

        # --- Base state ---
        base_quaternion = data.qpos[3:7]
        base_height = data.qpos[2]

        # --- Joint state ---
        joint_positions = data.qpos[7:]
        joint_velocities = data.qvel[6:]

        # --- Gravity vector ---
        g_world = np.array([0, 0, -1])
        gravity_body = rotate_vector_by_quat(g_world, base_quaternion)

        joint_positions_error_history = np.array(self.actions_history).flatten() - np.array(self.joint_positions_history).flatten()

        # --- Build observation vector ---
        obs = np.concatenate([
            gravity_body,    
            joint_positions,
            joint_velocities,   
            joint_positions_error_history,
            np.array(self.joint_velocities_history).flatten()
        ])

        return base_height, obs.astype(np.float32)
    
    # -- Compute the reward --
    def get_reward(self, obs, action):
        data = self.env.unwrapped.data
        g_world = np.array([0, 0, -1])
        base_linear_velocity = data.qvel[0:3]
        base_angular_velocity = data.qvel[3:6]
        base_height = data.qpos[2]
        base_quaternion = data.qpos[3:7]
        linear_velocity = rotate_vector_by_quat(base_linear_velocity, base_quaternion)
        angular_velocity = rotate_vector_by_quat(base_angular_velocity, base_quaternion)
        gravity_body = rotate_vector_by_quat(g_world, base_quaternion)
        joint_velocities = data.qvel[6:]
        joint_torques = data.qfrc_actuator[6:]
        angular_velocity_target = self.velocity_command[2] * np.array([0, 0, 1]) 
        linear_velocity_target = np.array([
            self.velocity_command[0],
            self.velocity_command[1],
            0.0
        ])

        contact_pairs = self.get_contact_pairs()
        If_sup = [c for c in contact_pairs if ((c[0][0] in self.foot_geom_ids or c[0][1] in self.foot_geom_ids) and c[1] > 0)]
        foot_contact_indices = []
        for c in If_sup:
            if c[0][0] in self.foot_geom_ids:
                foot_contact_indices.append(c[0][0])
            elif c[0][1] in self.foot_geom_ids:
                foot_contact_indices.append(c[0][1])  
                
        angular_velocity_cost = logistic_kernel(np.linalg.norm(angular_velocity - angular_velocity_target, ord=1), self.alpha_angular)
        linear_velocity_cost = logistic_kernel(np.linalg.norm(linear_velocity - linear_velocity_target, ord=1), self.alpha_linear)
        orientation_cost = np.sum(np.abs(gravity_body - g_world))
        action_cost = np.linalg.norm(self.actions_history[-2] - action)**2
        joint_limit_cost = np.sum(np.maximum(self.jslim -np.abs(joint_velocities), 0.0)**2)
        torque_cost = np.linalg.norm(joint_torques)**2
        height_cost = 1.0 if base_height < 0.35 else 0.0
        power_cost = np.sum(np.maximum(joint_torques * joint_velocities, 0))

        return -(self.K_orientation * orientation_cost +
                 self.K_action * action_cost +
                 self.K_joint_limit * joint_limit_cost +
                 self.K_torque * torque_cost +
                 self.K_height * height_cost +
                 self.K_power * power_cost +
                 self.K_angular_velocity * angular_velocity_cost +
                 self.K_linear_velocity * linear_velocity_cost)
   
    # -- Update histories after each step --
    def _update_histories(self, action):
        # Joint positions & velocities
        qpos = self.env.unwrapped.data.qpos[7:].copy()
        qvel = self.env.unwrapped.data.qvel[6:].copy()
        
        self.joint_positions_history.append(qpos)
        self.joint_velocities_history.append(qvel)
        self.actions_history.append(action)
        
        # Geom positions
        self._extract_geom_positions()

    # -- Extract geom positions for velocity calculation --
    def _extract_geom_positions(self):
        self.geom_positions = {i:self.env.unwrapped.data.geom_xpos[i].copy() for i in range(self.env.unwrapped.model.ngeom)}
        self.geom_positions_history.append(self.geom_positions.copy())

    # -- Get geom velocities based on position history --
    def get_geom_velocities(self):
        geom_velocities = {i: np.zeros(3) for i in range(self.env.unwrapped.model.ngeom)}
        if len(self.geom_positions_history) >= 2:
            geom_velocities = {i:(self.geom_positions[i] - self.geom_positions_history[-2][i]) / self.dt
                               for i in range(self.env.unwrapped.model.ngeom)}
        return geom_velocities
    
    # -- Get contact pairs from the environment --
    def get_contact_pairs(self):
        data = self.env.unwrapped.data
        contact_pairs = []
        for c in data.contact:
            contact_pairs.append(((c.geom1, c.geom2), c.dist, c.solimp))
        return contact_pairs
    
    # -- Determine termination condition --
    def get_terminated(self, obs, action, terminated_env):
        return terminated_env

