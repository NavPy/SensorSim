from matplotlib import pyplot as plt
import numpy as np
from imu_sensor import Imu_sensor
from PIL import Image # for flipping picture

# Load Simulation Data
dt = .02 # [sec]
hm = np.loadtxt('car_mag.txt')
delta_s = np.loadtxt('car_odometer.txt')
car_pos = np.loadtxt('car_pos.txt')
t, x, y, psi = car_pos[:,0], car_pos[:,1], car_pos[:,2], car_pos[:,4]

imu = Imu_sensor('car_imu.txt', _store=True, sensor_qual='con', biasdrift=True, nullshift=False)

# Update the IMU error specifications
imu._sqd['sigma_w_g'] = np.deg2rad(1.5)
imu._sqd['sigma_n_g'] = np.deg2rad(2) # note, nullshift may be off
imu._sqd['sigma_c_g'] = np.deg2rad(1.5)

# Declare placeholders
x_hat, y_hat = np.zeros_like(x), np.zeros_like(y)
psi_hat = np.zeros_like(psi)

# Initialize - start simulation at 0 + dt to avoid NaN as t=0
x_hat[0:2], y_hat[0:2] = x[0:2], y[0:2]
psi_hat[0:2] = psi[0:2]

for k in range(2, len(t)):
    # At time t[k], measurement delta_s[k-1] is available and our goal is
    # to bring our estimates from the 'past' (k-1) to the 'present' (k). 
    # So this is NOT predictive.
    
    # Get IMU data, but r is the only meaningful one for this simulation
    p, q, r, ax, ay, az = imu.get_imu(t[k-1])
    
    x_hat[k] = x_hat[k-1] + delta_s[k-1] * np.sin(psi_hat[k-1])
    y_hat[k] = y_hat[k-1] + delta_s[k-1] * np.cos(psi_hat[k-1])
    psi_hat[k] = psi_hat[k-1] + dt * r

# Plot  Results
pathImg = Image.open('car_path.png')

# Flip image over vertical axis.
# This is so that that plotting with 'lower' origin restores original shape
pathImg = pathImg.transpose(Image.FLIP_TOP_BOTTOM)

fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(1,1,1)

ax.imshow(pathImg, origin='lower')
ax.set_autoscale_on(False) # turn off autoscalling
ax.plot(    x,     y, alpha=.3, color='green', lw=3, label='True')
ax.plot(x_hat, y_hat,  alpha=1, color='red'  , lw=2, label='Estimate')
ax.legend()
plt.show()