from numpy import zeros, arctan2, pi, matrix, array
from math import sin, cos
# Pos2imu.py


def linvel(X, dt):
    """
    Linear approximation of velocity using n-dimensional position
    Compute velocity: V(k) = (P(k+1) - P(k))/dt;
    
    Inputs:
       P  - position history vector(nxN)
       dt - time step
    
    Outputs:
       V - velocity vector (nxN)
    
    Example: 
       dt = .5;
       P  = 0:dt:10
       [V] = linvel(P,dt)

    Author: Hamid Mokhtarzadeh
    email: mokh0006@umn.edu
    June 2011; Last revision: 21-June-2011
    June 21 2011: updated to match time update equations written as:
    P(k) = P(k-1) + dt*V(k-1)
    
    """
    # TODO: WARNING: This only works with 1-D vectors, as it is currently
    # coded.  It will given questionable results if higher dimensions
    # are attempted! (Oct 30 2012, Hamid)
    
    # assumes time is spanned by columns
    #transpose_flag = False
    #r,c = X.shape
    #if c < r:
    #    X = X.T
    #    transpose_flag = True
        
    
    V = zeros(X.shape)
    V[:-1] = (X[1::] - X[:-1])/dt
    V[-1] = V[-2]

    #if transpose_flag
    #    V = V.T
        
    return V
    
def linaccel(V, dt):
    """
    Linear approximation of acceleration using n-dimensional velocity
    Compute acceleration: A(k) = (V(k+1) - V(k))/dt;
    
    Inputs:
       V  - velocity history vector(nxN)
       dt - time step
    
    Outputs:
       A - acceleration vector (nxN)
    
    Example: 
       dt = .5;
       P  = [0:dt:10;0:dt:10]; % 2D position
       [V] = linvel(P,dt)
       [A] = linaccel(V,dt)

    Author: Hamid Mokhtarzadeh
    email: mokh0006@umn.edu
    Oct 2012; Last revision: 10-Oct-2012
    Based on linaccel.m function
    """

    A = zeros(V.shape)
    A[:-1] = (V[1::] - V[:-1])/dt
    A[-1] = A[-2]
    
    return A
    
def headangle(Vn, Ve):
    """
    Approximation of 2D heading using velocity where
    heading angle is with respect to north, positive 
    is clockwise rotation
    Compute heading: psi(k) = atan(Ve/Vn);
     
    Inputs:
       Vn - north velocity vector(1xN)
       Ve - east  velocity vector(1xN)
    
    Outputs:
       Psi - heading vector (1xN) [degrees]
    
    Example: 
       Vn = 1/sqrt(2);
       Ve = 1/sqrt(2);
       psi= headangle(Vn,Ve)

    Author: Hamid Mokhtarzadeh
    email: mokh0006@umn.edu
    Oct 2012; Last revision: 10-Oct-2012
    Based on headangle.m function
    """

    r2d = 180/pi # rad to degree factor
    psi_deg = 90 - r2d*arctan2(Vn,Ve);

    return psi_deg
    
def nav2body(v, psi_deg):
    """
    Rotate 2D vector from navigation to body frame (e.g. NE to XY)
    
    Inputs:
       v   - vector expressed in navigation frame(2xN array/matrix)
       psi - rotation angle history defining orientation of "X" 
             axis with respect to "North"  (1xN list) [degrees]
             (0 deg means "X" aligned with "North", clockwise positive)
    
    Outputs:
       vprime - vector v expressed in body frame (2xN)
    
    Example: 
       P  = [1;0]; % position [north,east] 
       psi_deg = 15; degrees from north
       Pprime = nav2body(P,psi_deg);

    Author: Hamid Mokhtarzadeh
    email: mokh0006@umn.edu
    June 2011; Last revision: 15-June-2011
    """

    vprime = matrix(zeros(v.shape)) # initialize placeholder

    # conver angle from degrees to radians
    d2r = pi/180.0
    psi_rad = array(psi_deg) * d2r

    for ind, psi in enumerate(psi_rad):
        # build rotation matrix
        R = matrix([[cos(psi), -sin(psi)],
                    [sin(psi),  cos(psi)]], dtype=float)
        R = R.T
        
        vprime[:,ind] = R*v[:,ind];

    return vprime
