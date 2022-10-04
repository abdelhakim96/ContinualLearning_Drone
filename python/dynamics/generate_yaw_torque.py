
import numpy as np

from dynamics.inverse_dynamics import Inverse
from dynamics.unicycle_dynamics import Unicycle
from utils.save_data import save_data_dynamics

k_end = 1000 * 1000
dt = 0.001

pose = np.zeros((k_end + 1, 3))
reference = np.zeros((k_end + 1, 3))
command = np.zeros((k_end + 1, 2))
command_random = np.zeros((k_end + 1, 2))
command_pid = np.zeros((k_end + 1, 2))
command_dnn = np.zeros((k_end + 1, 2))
command_inverse = np.zeros((k_end + 1, 2))
t = np.linspace(0, dt * k_end, num=(k_end + 1))

inverse = Inverse(Unicycle(dt, [0, 0, 0]))

command_inverse[k, :] = inverse.control(pose[k, :], reference[k + 1, :])

save_data_dynamics(t, reference, pose, command, 'unicycle_dynamics_fake')