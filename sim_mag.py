import numpy as np # tested to work with v1.7.1 
from navpy import angle2dcm


##########
# STEP 1 # 
##########
# Define magnetic fieled in navigation frame --> (h)

# Reference: http://www.ngdc.noaa.gov/geomag/magfield.shtml
nT2G = 1e-5# conversion from nanoTesla to Gauss
h = np.array([17491.9, 129.3, 52886.7])*nT2G # [Guass] for Zip Code: 55455 (U of MN)

def simulate_magnetometer(yaw, pitch, roll, input_units='rad'):
    """
    Simulate magnetometer measurements, as measured by a 3-axis magnetomter.  
    This is useful to demonstrate the ability of our calibration algorithm to 
    recover a calibrated measurement in proximity to the truth.  

    Parameters
    ----------
    yaw, pitch, roll : Length N numpy arrays of truth/assumed Euler angles,
                       assuming 3-2-1 rotation sequence, where N is the number
                       of data points.
    input_units: units for input angles {'rad', 'deg'}, optional.

    Returns
    -------
    hb: (N x 3) numpy array of truth magnetic field in the body-frame
    hm: (N x 3) numpy array of corrupted magnetometer measurements in body-frame,
        where N is the numberof data points.

    Notes
    -----
    The following describes the steps for simulating the magentometer measurements:

    1. Define magnetic fieled in navigation frame --> (h)
    2. Load simulated true attitude (Euler Angles) profile
    3. Map true magnetic field measurements to the body-frame using DCM --> (hb)
    4. Load simulated sensor errors for magnetometer & form corrupting transformation (C, b).
    5. Corrupt true body-frame measurements accoding to:  hm = C * hb + b + n
    6. Return true and corrupted body-axis magnetometer measurements.

    Todo
    ----
    If position information is availble, we can use a magnetic field model to vary
    the true observed magnetic field.

    Date: September 25, 2013
    Updated: Octtober 6, 2013
    Author: Hamid Mokhtarzadeh
    """
    print('Truth magnetic field assumed for Minneapolis, MN.')
    # TODO: Find way to define desired error terms - currently they are hard-coded!
    print('Warning: Assumed error terms are currently hard-coded in function!')
    
    # Apply necessary unit transformations.
    if input_units == 'rad':
        yaw_rad, pitch_rad, roll_rad = yaw, pitch, roll
    elif input_units == 'deg':
        yaw_rad, pitch_rad, roll_rad = np.radians([yaw, pitch, roll])

    drl = len(yaw_rad)

    ##########
    # STEP 3 #
    ##########
    hb = np.nan * np.zeros((drl,3))
    # Map true magnetic field measurements to the body-frame using DCM --> (hb)
    for k, [y, p, r] in enumerate(zip(yaw_rad, pitch_rad, roll_rad)):
        Rbody2nav = angle2dcm(y, p, r)
        hb[k,:] = np.dot(Rbody2nav, h)
        

    ##########
    # STEP 4 #
    ##########
    # Load simulated sensor errors for magnetometer & form
    # sigma_n - standard deviation of additive Guassian white-noise
    # b - 3x1 vector, null-shift or hard iron errors
    # Cs - 3x3 matrix, scale factor
    # Ce - 3x3 matrix, misalignment affects
    # Ca - 3x3 matrix, soft-iron and axes nonorthogonality
    #
    # C = Cs * Ce * Ca
    sigma_n = 0.0007  # Guass # TODO: decide whether to add noise term or not

    bx, by, bz = 0.2, 0.2, -0.1 # bias
    sx, sy, sz = 0.0, 0.0, 0.0 # scale factor (0: no scale factor error)
    ex, ey, ez = 0., 0., 0. # misalignment errors (0: no misalignment error)

    # Combined soft-iron and axes nonorthogonality effects (0: no error)
    axx, ayy, azz = 0., 0., 0.

    # We'll assume symmetric effects for now.
    axy = ayx = 0.
    axz = azx = 0.
    ayz = azy = 0.

    # Form error terms
    Cs = np.eye(3) + np.diag([sx, sy, sz])
    Ce = np.array([[ 1.,  ez, -ey],
                   [-ez,  1.,  ex],
                   [ ey, -ex,  1.]])
    Ca = np.array([[1.+axx,   axy,   axz],
                   [  ayx, 1.+ayy,   ayz],
                   [  azx,   azy, 1.+azz]])
    b = np.array([bx, by, bz])
    n = sigma_n * np.random.randn(drl, 3) 

    C = Cs.dot(Ce).dot(Ca)

    ##########
    # STEP 5 #
    ##########
    # Corrupt true body-frame measurements according to:
    # hm = C * hb + b + n
    hm = C.dot(hb.T).T + b + n
    
    return hb, hm
    
    
if __name__ == '__main__':
    # Simulate magnetometer measurements for hard-coded attitude.
    
    ##########
    # STEP 2 #
    ##########
    # Load simulated true attitude profile
    # Stored attitude will be in units of Radians
    N = 100
    swing_y = np.deg2rad(np.linspace(0., 360., N))
    swing_p = np.deg2rad(np.linspace(-120, 120., N))
    swing_r = np.deg2rad(np.linspace(-90., 90., N))
    zr = np.zeros(len(swing_y))

    yaw = np.hstack((swing_y, zr, zr, swing_y))
    pitch = np.hstack((zr, swing_p, zr, swing_p))
    roll  = np.hstack((zr, zr, swing_r, swing_r))
    
    hb, hm = simulate_magnetometer(yaw, pitch, roll, input_units='rad')
