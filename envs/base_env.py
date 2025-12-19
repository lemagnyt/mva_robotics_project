import gymnasium as gym
import numpy as np
from collections import deque

class BaseEnv(gym.Env):
    """
    Base environment that uses an existing Gym environment.
    Each specialized environment can override get_observation, get_reward, and get_terminated.
    """
    def __init__(self, env, obs_dim=113, action_dim=12, max_steps=100, history_len=5, use_tsif=False, height_estimator=None):
        super().__init__()
        self.env = env
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)
        
        self.action_dim = action_dim
        self.max_steps = max_steps
        self.step_count = 0
        self.current_obs = None
        self.history_len = history_len
        
        ## ---- Buffers with default values ----
        self.joint_positions_history = deque([np.zeros(self.env.unwrapped.data.qpos[7:].shape)] * history_len, maxlen=history_len)
        self.joint_velocities_history = deque([np.zeros(self.env.unwrapped.data.qvel[6:].shape)] * history_len, maxlen=history_len)
        self.actions_history = deque([np.zeros(action_dim)] * history_len, maxlen=history_len)
        self.geom_positions_history = deque([np.copy(self.env.unwrapped.data.geom_xpos) for _ in range(history_len)], maxlen=history_len)
        
        self.foot_geom_ids = [45, 74, 103, 132]  # Foot IDs
        self.jslim = 40.0  # Joint velocity limit
        
        self.current_obs = None

        # ------------- TSIF parameters -------------
        self.use_tsif = use_tsif
        self.angular_velocity_std = 0.2      # rad/s
        self.linear_velocity_std = 0.25      # m/s
        self.joint_position_std = 0.05       # rad
        self.joint_velocity_std = 0.5        # rad/s

        self.height_estimator = height_estimator

        self.geom_positions = np.zeros((self.env.unwrapped.model.ngeom, 3))
        # Calculate dt
        self.dt = self.env.unwrapped.model.opt.timestep * self.env.unwrapped.frame_skip

    def reset(self, *, seed=None, options=None):
        return self.current_obs, {}

    def step(self, action):
        self.step_count += 1
        self.dt = self.env.unwrapped.model.opt.timestep * self.env.unwrapped.frame_skip
        
        # Update buffers
        self._update_histories(action)

        # Reduce the action (Test)
        current_joint_pos = self.env.unwrapped.data.qpos[7:].copy()
        action = current_joint_pos + (action-current_joint_pos)*0.8

        obs_raw, _, terminated, truncated, info = self.env.step(action)
        obs = self.get_observation(obs_raw)
        reward = self.get_reward(obs, action, debug=False)
        terminated = self.get_terminated(obs, action, terminated)
        
        truncated = truncated or (self.step_count >= self.max_steps)
        self.current_obs = obs
        self.last_action = action

        #if terminated or truncated:
        #    self.reset()
        return obs, reward, terminated, truncated, info

    # Override these methods in specialized environments
    def get_observation(self, obs_raw):
        return np.array(obs_raw, dtype=np.float32)

    def get_reward(self, obs, action, debug=False):
        return 0.0

    def get_terminated(self, obs, action, terminated_env):
        return terminated_env

    def get_contact_pairs(self):
        data = self.env.unwrapped.data
        contact_pairs = []
        for c in data.contact:
            contact_pairs.append(((c.geom1, c.geom2), c.dist, c.solimp))
        return contact_pairs

    def _extract_geom_positions(self):
        self.geom_positions = {i:self.env.unwrapped.data.geom_xpos[i].copy() for i in range(self.env.unwrapped.model.ngeom)}
        self.geom_positions_history.append(self.geom_positions.copy())

    def get_geom_velocities(self):
        geom_velocities = {i: np.zeros(3) for i in range(self.env.unwrapped.model.ngeom)}
        if len(self.geom_positions_history) >= 2:
            geom_velocities = {i:(self.geom_positions[i] - self.geom_positions_history[-2][i]) / self.dt
                               for i in range(self.env.unwrapped.model.ngeom)}
        return geom_velocities

    # Update histories after each step
    def _update_histories(self, action):
        qpos = self.env.unwrapped.data.qpos[7:].copy()
        qvel = self.env.unwrapped.data.qvel[6:].copy()
        
        self.joint_positions_history.append(qpos)
        self.joint_velocities_history.append(qvel)
        self.actions_history.append(action)
        
        self._extract_geom_positions()

    # Used for naive selector to get robot tilt angle
    def get_base_z_angle_deg_quat(self):
        data = self.env.unwrapped.data
        
        # Body quaternion (w, x, y, z)
        q = data.qpos[3:7]
        w, x, y, z = q

        # Quaternion to 3x3 rotation matrix conversion
        R = np.array([
            [1 - 2*(y**2 + z**2),     2*(x*y - z*w),       2*(x*z + y*w)],
            [2*(x*y + z*w),           1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
            [2*(x*z - y*w),           2*(y*z + x*w),       1 - 2*(x**2 + y**2)]
        ])

        base_z = R[:, 2]  # Robot's z axis expressed in world frame
        inertial_z = np.array([0.0, 0.0, 1.0])

        base_z = base_z / np.linalg.norm(base_z)
        X_rad = np.arccos(np.clip(np.dot(base_z, inertial_z), -1.0, 1.0))
        X_deg = np.degrees(X_rad)

        return X_deg
