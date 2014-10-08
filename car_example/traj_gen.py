from PIL import Image # for flipping picture
from matplotlib import pyplot as plt
from scipy import interpolate
from scipy.misc import derivative
import numpy as np
from pos2imu import linvel
from sim_mag import simulate_magnetometer

pathImg = Image.open('car_path.png')

# Flip image over vertical axis.
# This is so that that plotting with 'lower' origin restores original shape
pathImg = pathImg.transpose(Image.FLIP_TOP_BOTTOM)

fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(1,1,1)

ax.imshow(pathImg, origin='lower')
ax.set_autoscale_on(False) # turn off autoscalling

# Get Locations
print('Select critical points defining path.  End selection by \'middle click\'')
critical_pts = plt.ginput(-1, timeout=0)

# Loop through points and assign times
# Plot a marker placeholder
marker, = ax.plot([], [], '*r', markersize=10)

critical_t = np.linspace(0, 60, len(critical_pts))

#critical_t = []
#
#for k, (x_k,y_k) in enumerate(critical_pts):
#    marker.set_xdata(x_k)
#    marker.set_ydata(y_k)
#    
#    plt.draw()
#    
#    critical_t.append(float(raw_input('t%i: ' % k)))

    
## Define a fine time vector
dt = .02 # [sec]
tlim = [critical_t[0], critical_t[-1]]
t = np.arange(tlim[0], tlim[1], dt)

# Interpolate x,y positions using cubic-spline
critical_pts = np.array(critical_pts)
xfun = interpolate.interp1d(critical_t, critical_pts[:,0], kind='cubic')
yfun = interpolate.interp1d(critical_t, critical_pts[:,1], kind='cubic')

x, y = xfun(t), yfun(t)
vx, vy = np.nan * np.zeros_like(x), np.nan * np.zeros_like(y)
for k in range(1, len(t)-1):
    
    vx[k] = derivative(xfun, t[k], dx=dt/2)
    vy[k] = derivative(yfun, t[k], dx=dt/2)

# Get rid of 'nan by setting first an last points equal to nearest neighbor
vx[0], vx[-1] = vx[1], vx[-2]
vy[0], vy[-1] = vy[1], vy[-2]

# Derive heading and rotation-rate
psi = np.arctan2(vx, vy)
psidot = linvel(np.unwrap(psi), dt)


# Write output to file
ZR = np.zeros_like(t)
# [t, p, q, r, ax, ay, az]
fmt_ = ['%5.4f'] + 2*['%i'] + ['%4.9f'] + 3*['%i']
np.savetxt('car_imu.txt', np.transpose([t, ZR, ZR, psidot, ZR, ZR, ZR]), fmt=fmt_)

# Store true position
# [t, x, y, z, yaw, pitch, roll]
fmt_ = ['%5.4f'] + 2*['%4.9f'] + ['%i'] + ['%4.9f'] + 2*['%i']
np.savetxt('car_pos.txt', np.transpose([t, x, y, ZR, psi, ZR, ZR]), fmt=fmt_)

# Odometer (delta positions)
# [delta_s]
delta_s = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
np.savetxt('car_odometer.txt', delta_s)

# Magnetometer
# [hm_x, hm_y, hm_z]
hb, hm = simulate_magnetometer(psi, ZR, ZR)
# Get rid of bias, so that only noise is simulated
b = np.array([0.2, 0.2, -0.1]) # TODO: These are the biases that were hardcoded 
                               # in simulate_magntometer() on 10/23/2013.  These
                               # could have changed!
hm -= b
np.savetxt('car_mag.txt', hm)

    

# Plot true path on map
fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(1,1,1)

ax.imshow(pathImg, origin='lower')
ax.set_autoscale_on(False) # turn off autoscalling
ax.plot(x,y, alpha=.3, lw=3)
plt.show()
