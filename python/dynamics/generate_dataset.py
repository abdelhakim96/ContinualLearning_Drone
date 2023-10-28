import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

# Constants
r = 0.3  # wheel radius
m = 20  # mass
i_y = m * r ** 2  # wheel rotational inertia around y-axis
i_z = m * r ** 2 / 2  # wheel rotational inertia around z-axis
dt = 0.001
k = 100000

# Arrays
t = np.linspace(0, (k - 1)*dt, num=k)
e_x = 1*np.random.randn(k, 1)
e_y = 1*np.random.randn(k, 1)
yaw = np.clip(2*np.random.randn(k, 1), -math.pi, math.pi)
v = np.clip(10*np.random.randn(k, 1), -20, 20)
w = np.clip(10*np.pi*np.random.randn(k, 1), -20*np.pi, 20*np.pi)

yaw_ref = np.arctan2(e_y, e_x)
e_yaw = yaw_ref - yaw
e_yaw -= (abs(e_yaw) > math.pi) * 2 * math.pi * np.sign(e_yaw)  # denormalize heading

# Inverse law
tau_y = i_y*(e_x*np.cos(yaw + w*dt) + e_y*np.sin(yaw + w*dt) - v*dt*(1 + np.cos(w*dt)))/(r*dt**2)
tau_z = i_z*(e_yaw - 2*w*dt)/(dt**2)

# Damping to compensate for saturated control inputs
# tau_y -= 100000*v/r
# tau_z -= 100000*w
# tau_y = np.clip(tau_y, -100, 100)
# tau_z = np.clip(tau_z, -100, 100)

# Save dataset
data = np.column_stack([t, np.zeros((k, 3)), yaw, e_x, e_y, e_yaw, v, w, tau_y, tau_z])
header = ['time', 'x_d', 'y_d', 'yaw_d', 'yaw', 'e_x', 'e_y', 'e_yaw', 'v', 'w', 'tau_y', 'tau_z']
dataset = pd.DataFrame(data, columns=header)
dataset.to_csv('../data/log_unicycle_dynamics_artificial.csv', index=False)

# Plot dataset
plt.figure(10)
plt.title('yaw')
plt.hist(dataset[['yaw']].values)
plt.figure(3)
plt.title('e_x')
plt.hist(dataset[['e_x']].values)
plt.figure(4)
plt.title('e_y')
plt.hist(dataset[['e_y']].values)
plt.figure(5)
plt.title('e_yaw')
plt.hist(dataset[['e_yaw']].values)
plt.figure(6)
plt.title('v')
plt.hist(dataset[['v']].values)
plt.figure(7)
plt.title('w')
plt.hist(dataset[['w']].values)
plt.figure(8)
plt.title('tau_y')
plt.hist(dataset[['tau_y']].values)
plt.figure(9)
plt.title('tau_z')
plt.hist(dataset[['tau_z']].values)
plt.grid()
plt.show()
