import numpy as np

def rotate_vector_by_quat(v, q):
    # Rotate vector v by quaternion q
    w, x, y, z = q
    R = np.array([
        [1 - 2*(y**2 + z**2),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x**2 + z**2),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
    ])
    return R @ v

def logistic_kernel(e, alpha):
    # Logistic kernel function
    return -1.0 / (np.exp(alpha * e) + 2 + np.exp(-alpha * e))

def dphi(phi, phi_target):
    # Difference between two angles, wrapped to [-pi, pi]
    diff = phi_target - phi
    return np.arctan2(np.sin(diff), np.cos(diff))

def sample_velocity_command():
    # Sample random velocity command within specified ranges
    v_forward = np.random.uniform(-1.0, 1.0)
    v_lateral = np.random.uniform(-0.4, 0.4)
    yaw_rate  = np.random.uniform(-1.2, 1.2)

    return np.array([v_forward, v_lateral, yaw_rate], dtype=np.float32)