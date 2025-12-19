from envs.base_env import BaseEnv
import numpy as np
from utils.utils_functions import rotate_vector_by_quat
from utils.utils_functions import dphi
from collections import deque

class SelfRightingEnv(BaseEnv):
    """
    Specialized environment for self-righting.
    """
    def __init__(self, env, obs_dim=90, init_height=1, use_stif=False, height_estimator=None, max_steps=300):
        super().__init__(env, obs_dim, use_tsif=use_stif, height_estimator=height_estimator, max_steps=max_steps)
        self.target_position = np.array([0.0, 1.4, -2.5] * 4)
        self.K_body_impulse       = 1.0       # penalize less for the robot touching the ground
        self.K_body_slippage      = 0.05      # tolerate some slippage
        self.K_joint_acceleration = 0.0004    # slightly penalize fast movements
        self.K_joint_position     = 0.2       # encourage correct leg positions
        self.K_orientation        = 6         # strong penalty for being on the back
        self.K_self_collision     = 2.0       # avoid folding onto itself
        self.K_action             = 0.05      # keep actions low-cost
        self.K_joint_limit        = 0.005     # allow use of limits to get up
        self.K_torque             = 0.0001    # small penalty for joint forces

        self.init_height          = init_height
        self.K_desired            = 1.0       # general objective
        self.dt = self.env.unwrapped.model.opt.timestep * self.env.unwrapped.frame_skip

    def reset(self, *, seed=None, options=None):
        #print('Reward : ',self.get_reward(self.current_obs, self.actions_history[-1], debug=True),' - nsteps : '+str(self.step_count)) # Debug info
        self.step_count = 0

        obs_raw, _ = self.env.reset(seed=seed)
        
        qpos = self.env.unwrapped.data.qpos.copy()
        qvel = self.env.unwrapped.data.qvel.copy()

        qpos[2] = self.init_height  # set z height to init_height
        qvel[:] = 0.0   # zero velocity at start
        
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
        angular_velocity = rotate_vector_by_quat(base_angular_velocity, base_quaternion)

        # --- Joint state ---
        joint_positions = data.qpos[7:]
        joint_velocities = data.qvel[6:]

        if self.use_tsif:
            angular_velocity += np.random.normal(0, self.angular_velocity_std * self.dt, len(angular_velocity))
            joint_positions += np.random.normal(0, self.joint_position_std, len(joint_positions))
            joint_velocities += np.random.normal(0, self.joint_velocity_std * self.dt, len(joint_velocities))

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
            self.actions_history[-2]
        ])

        return obs.astype(np.float32)

    # Compute reward
    def get_reward(self, obs, action, debug = False):
        data = self.env.unwrapped.data
        g_world = np.array([0, 0, -1])
        base_quaternion = data.qpos[3:7]
        gravity_body = rotate_vector_by_quat(g_world, base_quaternion)
        joint_positions = data.qpos[7:]
        joint_velocities = data.qvel[6:]
        joint_torques = data.qfrc_actuator[6:]
        prev_joint_velocities = self.joint_velocities_history[-2]  # velocity at t-1

        contact_pairs = self.get_contact_pairs()
        ic = [c[2] for c in contact_pairs]  # all impulse
        ic_f = [c[2] for c in contact_pairs if c[0][0] in self.foot_geom_ids or c[0][1] in self.foot_geom_ids] # foot impulses
        Ic_in = [c for c in contact_pairs if (c[0][0] == c[0][1])]
        geom_velocities = self.get_geom_velocities()

        body_impulse_cost = np.sum([
            np.linalg.norm(imp) 
            for imp in ic 
            if not any(np.array_equal(imp, f) for f in ic_f)
        ]) / (len(ic) - len(ic_f)) if (len(ic) - len(ic_f)) > 0 else 0.0
        body_slippage_cost = np.sum([abs(geom_velocities[c[0][1]]-geom_velocities[c[0][0]]) for c in contact_pairs]) / max(len(contact_pairs), 1)
        joint_acceleration_cost = np.sum(((joint_velocities - prev_joint_velocities))**2)
        joint_position_cost = np.sum(np.abs(dphi(joint_positions, self.target_position)))
        orientation_cost = np.sum(np.abs(gravity_body - g_world))
        self_colision_cost = len(Ic_in)
        action_cost = np.linalg.norm(self.actions_history[-2] - action)**2
        joint_limit_cost = np.sum(np.maximum(np.abs(joint_velocities)-self.jslim, 0.0)**2)
        torque_cost = np.linalg.norm(joint_torques)**2

        if debug:
            print("Costs: Body Impulse:", body_impulse_cost*self.K_body_impulse,
                "Body Slippage:", body_slippage_cost*self.K_body_slippage,
                "Joint Acceleration:", joint_acceleration_cost*self.K_joint_acceleration,
                "Joint Position:", joint_position_cost*self.K_joint_position,
                "Orientation:", orientation_cost*self.K_orientation,
                "Self-Collision:", self_colision_cost*self.K_self_collision,
                "Action:", action_cost*self.K_action,
                "Joint Limit:", joint_limit_cost*self.K_joint_limit,
                "Torque:", torque_cost*self.K_torque)  # Debug info
            
        total_cost = (self.K_body_impulse * body_impulse_cost +
                 self.K_body_slippage * body_slippage_cost +
                 self.K_joint_acceleration * joint_acceleration_cost +
                 self.K_joint_position * joint_position_cost +
                 self.K_orientation * orientation_cost +
                 self.K_self_collision * self_colision_cost +
                 self.K_action * action_cost +
                 self.K_joint_limit * joint_limit_cost +
                 self.K_torque * torque_cost)
        return -total_cost  

    def get_terminated(self, obs, action, terminated_env):
        return self.step_count>=self.max_steps
    
    # Desired joint positions for the actuator
    def get_desired_position(self,action):
        return self.K_desired * action + self.env.unwrapped.data.qpos[7:]
    
