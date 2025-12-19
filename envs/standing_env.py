from envs.base_env import BaseEnv
import numpy as np
from utils.utils_functions import rotate_vector_by_quat, dphi
from collections import deque

class StandingEnv(BaseEnv):
    """
    Specialized environment for standing position.
    """

    def __init__(self, env, obs_dim=90, init_height=0.3, use_stif=False, height_estimator=None, max_steps=100):
        super().__init__(env, obs_dim, use_tsif=use_stif, height_estimator=height_estimator, max_steps=max_steps)
        # Target joint positions for standing
        self.target_position = np.zeros(12)
        # Initial sitting joint positions
        self.init_joint_position = np.array([0,  0.5, -1.5,  0.0,  0.5, -1.5,
                                             0, -0.5,  1.5,  0.0, -0.5,  1.5])

        self.init_height = init_height

        self.K_orientation        = 1.7      # The robot must remain upright
        self.K_height             =  8       # Stable height = essential
        self.K_action             = 0.05     # Encourage immobility
        self.K_joint_acceleration = 0.0004   # Prevents shaking
        self.K_joint_limit        = 0.0002   # Prevents extreme postures
        self.K_torque             = 0.0002   # Limits unnecessary effort
        self.K_joint_positions    = 0.6      # Keeps close to target (straight) positions

        self.K_desired = 1.0 # For the actuator

    def reset(self, *, seed=None, options=None):
        # print('Reward : ',self.get_reward(self.current_obs, self.actions_history[-1], debug=True),' - nsteps : '+str(self.step_count)) # Debug info
        self.step_count = 0
        obs_raw, _ = self.env.reset(seed=seed)
        
        qpos = self.env.unwrapped.data.qpos.copy()
        qvel = self.env.unwrapped.data.qvel.copy()

        # Initial height
        qpos[2] = self.init_height

        # Initial joint positions: upright sitting
        qpos[7:] = self.init_joint_position

        # Initial velocities set to zero
        qvel[:] = 0.0
                
        self.env.unwrapped.set_state(qpos, qvel)
        
        obs_raw = self.env.unwrapped._get_obs()
        self.current_obs = self.get_observation(obs_raw)

        # Reset buffers
        self.joint_positions_history = deque([np.zeros(self.env.unwrapped.data.qpos[7:].shape)] * self.history_len, maxlen=self.history_len)
        self.joint_velocities_history = deque([np.zeros(self.env.unwrapped.data.qvel[6:].shape)] * self.history_len, maxlen=self.history_len)
        self.actions_history = deque([np.zeros(self.action_dim)] * self.history_len, maxlen=self.history_len)
        self.geom_positions_history = deque([np.zeros(self.env.unwrapped.data.geom_xpos.shape)] * self.history_len, maxlen=self.history_len)
        
        return self.current_obs, {}
    
    # Build observation vector
    def get_observation(self, obs_raw):
        data = self.env.unwrapped.data

        # --- Base state ---
        base_quaternion = data.qpos[3:7]
        base_angular_velocity = data.qvel[3:6]
        base_linear_velocity = data.qvel[0:3]

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

        # --- Gravity vector ---
        g_world = np.array([0, 0, -1])
        gravity_body = rotate_vector_by_quat(g_world, base_quaternion)

        # --- Build observation vector ---
        obs = np.concatenate([
            gravity_body,
            angular_velocity,
            base_linear_velocity,
            joint_positions,
            joint_velocities,
            np.array(self.joint_positions_history).flatten(),
            np.array(self.joint_velocities_history).flatten(),
            self.actions_history[-1]
        ])
        return obs.astype(np.float32)

    # Compute reward
    def get_reward(self, obs, action, debug = False):
        data = self.env.unwrapped.data
        g_world = np.array([0, 0, -1])
        base_quaternion = data.qpos[3:7]
        base_height = data.qpos[2]
        gravity_body = rotate_vector_by_quat(g_world, base_quaternion)
        joint_velocities = data.qvel[6:]
        joint_torques = data.qfrc_actuator[6:]
        prev_joint_velocities = self.joint_velocities_history[-2]  # velocity at t-1
        joint_positions = data.qpos[7:]

        joint_acceleration_cost = np.sum(((joint_velocities - prev_joint_velocities))**2)
        orientation_cost = np.sum(np.abs(gravity_body - g_world))
        action_cost = np.linalg.norm(self.actions_history[-2] - action)**2
        joint_limit_cost = np.sum(np.maximum(self.jslim - np.abs(joint_velocities), 0.0)**2)
        torque_cost = np.linalg.norm(joint_torques)**2
        height_cost = 1.0 if base_height < 0.35 else 0.0
        joint_position_cost = np.sum(np.abs(dphi(joint_positions, self.target_position)))

        contact_pairs = self.get_contact_pairs()
        ic = [c[2] for c in contact_pairs]  # all impulse
        ic_f = [c[0] for c in contact_pairs if c[0][0] in self.foot_geom_ids or c[0][1] in self.foot_geom_ids] # foot impulses
        if debug:
            print('Joint accel cost:', joint_acceleration_cost*self.K_joint_acceleration,
                'Orientation cost:', orientation_cost*self.K_orientation,
                'Action cost:', action_cost*self.K_action,
                'Joint limit cost:', joint_limit_cost*self.K_joint_limit,
                'Torque cost:', torque_cost*self.K_torque,
                'Height cost:', height_cost*self.K_height,
                'Joint pos cost:', joint_position_cost*self.K_joint_positions)
            
        return -(self.K_joint_acceleration * joint_acceleration_cost +
                 min(self.K_orientation * orientation_cost,5.0) +
                 self.K_action * action_cost +
                 self.K_joint_limit * joint_limit_cost +
                 self.K_torque * torque_cost +
                 self.K_height * height_cost+
                 self.K_joint_positions * joint_position_cost-
                 2*(max(len(ic_f)-1,0))**2)  # Added by us, to penalize if at least one foot does not touch the ground

    # Check termination condition
    def get_terminated(self, obs, action, terminated_env):
        return self.step_count>=self.max_steps

    # Desired joint positions for the actuator
    def get_desired_position(self,action):
        return self.K_desired * action + self.env.unwrapped.data.qpos[7:]
