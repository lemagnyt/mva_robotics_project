from envs.base_env import BaseEnv
import numpy as np
from utils.utils_functions import rotate_vector_by_quat
from utils.utils_functions import logistic_kernel, sample_velocity_command
from collections import deque
import torch

class LocomotionEnv(BaseEnv):
    """
    Specialized environment for locomotion.
    """

    def __init__(self, env, obs_dim=29, action_dim=12, max_steps=300, init_height=0.55, use_stif=False, height_estimator=None):
        super().__init__(env, obs_dim, action_dim, max_steps, use_tsif=use_stif, height_estimator=height_estimator)
        self.velocity_command = np.array([1,0,0]) # Initial velocity command
        self.K_orientation        = 1.7     # Maintain correct orientation (very important)
        self.K_action             = 0.05   # Low cost on action effort
        self.K_joint_limit        = 0.0002   # Penalty if joints reach their limits
        self.K_torque             = 0.0002  # Low penalty on high torques
        self.K_foot_slippage      = 0.5    # Avoid foot slippage (important for locomotion)
        self.K_foot_clearance     = 0.3     # Maintain proper foot clearance
        self.K_angular_velocity   = 10   # Track angular velocity targets
        self.K_linear_velocity    =  5   # Track linear velocity targets (essential for locomotion)

        # Weights for kernel functions
        self.alpha_linear  = 1   # Weight to calculate linear velocity cost
        self.alpha_angular = 0.5 # Weight to calculate angular velocity cost

        self.init_height = init_height
        self.K_desired = 1.0

    def reset(self, *, seed=None, options=None):
        #print('Reward : ',self.get_reward(self.current_obs, self.actions_history[-1], debug=True),' - nsteps : '+str(self.step_count)) # Debug info
        self.step_count = 0
        obs_raw, _ = self.env.reset(seed=seed)
        
        qpos = self.env.unwrapped.data.qpos.copy()
        qvel = self.env.unwrapped.data.qvel.copy()

        qpos[2] = self.init_height  # z height set to init_height

        self.velocity_command = sample_velocity_command()  # New velocity command at each reset

        # Add small noise to joints for diversity
        joint_noise = 0.01 * np.random.randn(*qpos[7:].shape)
        qpos[7:] += joint_noise
        
        qvel[:] = 0.0   # zero velocity at start
        
        # Apply state
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
        base_height = data.qpos[2] # To change with base height estimation !!!
        angular_velocity = rotate_vector_by_quat(base_angular_velocity, base_quaternion)

        # --- Joint state ---
        joint_positions = data.qpos[7:]
        joint_velocities = data.qvel[6:]

        if self.use_tsif:
            angular_velocity += np.random.normal(0, self.angular_velocity_std * self.dt, len(angular_velocity))
            base_linear_velocity += np.random.normal(0, self.linear_velocity_std * self.dt, len(base_linear_velocity))
            joint_positions += np.random.normal(0, self.joint_position_std, len(joint_positions))
            joint_velocities += np.random.normal(0, self.joint_velocity_std * self.dt, len(joint_velocities))

        if self.height_estimator is not None:
            # Estimate height with the model
            obs_tensor = torch.FloatTensor(self.current_obs).unsqueeze(0)
            with torch.no_grad():
                height_est = self.height_estimator(obs_tensor).item()
            base_height = height_est

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
            np.array([base_height])
        ])
        
        return obs.astype(np.float32)

    # Compute reward
    def get_reward(self, obs, action, debug=False):
        data = self.env.unwrapped.data
        g_world = np.array([0, 0, -1])
        base_linear_velocity = data.qvel[0:3]
        base_angular_velocity = data.qvel[3:6]
        base_quaternion = data.qpos[3:7]
        base_height = data.qpos[2]
        linear_velocity = rotate_vector_by_quat(base_linear_velocity, base_quaternion)
        angular_velocity = rotate_vector_by_quat(base_angular_velocity, base_quaternion)
        gravity_body = rotate_vector_by_quat(g_world, base_quaternion)
        joint_velocities = data.qvel[6:]
        joint_torques = data.qfrc_actuator[6:]
        angular_velocity_target = self.velocity_command[2] * np.array([0, 0, 1])  # yaw rate around z-axis
        linear_velocity_target = np.array([
            self.velocity_command[0],  # forward velocity
            self.velocity_command[1],  # lateral velocity
            0.0
        ])
        geom_velocities = self.get_geom_velocities()

        contact_pairs = self.get_contact_pairs()
        If_inf = [c for c in contact_pairs if ((c[0][0] in self.foot_geom_ids or c[0][1] in self.foot_geom_ids) and c[1] <= 0)]
        If_sup = [c for c in contact_pairs if ((c[0][0] in self.foot_geom_ids or c[0][1] in self.foot_geom_ids) and c[1] > 0)]
        ic_f = [c[0] for c in contact_pairs if ((0 in c[0]) and any(f in c[0] for f in self.foot_geom_ids))]
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
        joint_limit_cost = np.sum(np.maximum(np.abs(joint_velocities)-self.jslim, 0.0)**2)
        torque_cost = np.linalg.norm(joint_torques)**2
        foot_slippage_cost = np.sum([abs(geom_velocities[c[0][1]]-geom_velocities[c[0][0]]) for c in If_inf])
        foot_clearance_cost = np.sum([(self.geom_positions[i]-0.07)**2 * np.linalg.norm(geom_velocities[i]) for i in foot_contact_indices])

        if debug:
            print('Angular vel cost:', angular_velocity_cost*self.K_angular_velocity,
                'Linear vel cost:', linear_velocity_cost*self.K_linear_velocity,
                'Orientation cost:', orientation_cost*self.K_orientation,
                'Foot slippage cost:', foot_slippage_cost*self.K_foot_slippage,
                'Foot clearance cost:', foot_clearance_cost*self.K_foot_clearance,
                'Action cost:', action_cost*self.K_action,
                'Joint limit cost:', joint_limit_cost*self.K_joint_limit,
                'Torque cost:', torque_cost*self.K_torque,
                "Height cost:", (base_height<0.3)*4.0,
                "Contacts cost:", 2*(max(3-len(ic_f),0))**2)
        return  -(
                 self.K_orientation * orientation_cost +
                 self.K_action * action_cost +
                 self.K_joint_limit * joint_limit_cost +
                 self.K_torque * torque_cost +
                 self.K_foot_slippage * foot_slippage_cost +
                 self.K_foot_clearance * foot_clearance_cost-
                 self.K_angular_velocity * angular_velocity_cost -
                 self.K_linear_velocity * linear_velocity_cost -
                 3*(max(len(ic_f)-2,0)**2) + # Added by us, to penalize if at least two feet do not touch the ground
                 (base_height<0.35)*4.0) # Added by us to avoid falling

    # Determine if episode is terminated
    def get_terminated(self, obs, action, terminated_env):
        contact_pairs = self.get_contact_pairs()
        ic_f = [c[0] for c in contact_pairs if ((0 in c[0]) and any(f in c[0] for f in self.foot_geom_ids))]
        fall = (self.env.unwrapped.data.qpos[2] < 0.15 and len(ic_f)<=0) # fall if base height < 0.25m and less than 2 feet in contact
        joint_velocities = self.env.unwrapped.data.qvel[6:]
        excessive_joint_speed = np.any(np.abs(joint_velocities) > (self.jslim+10)) # excessive joint speed
        return fall or excessive_joint_speed or terminated_env

    # Update velocity command
    def set_velocity_command(self, command):
        self.velocity_command = np.array(command, dtype=np.float32)

    # Get desired joint position for actuator input
    def get_desired_position(self,action):
        return self.K_desired * action + np.zeros(len(self.env.unwrapped.data.qpos[7:]))

    
    
