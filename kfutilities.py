
from numpy import shape, zeros, hstack, vstack, matrix
from scipy.linalg import expm
import numpy as np

def discrete_process_noise(F,G,dt,Q, order=2):
    """   
    This functions determines the equivalent discrete   
    process noise matrix Q_k for the continuous system     

                     x_dot = Fx + Lu + Gw
         *deterministic input Lu doesn't contribute                        

    driven by white noise, w, of power spectral density Q.  
    The sampling time is dt.  The order of the approximation
    can be specified.  If the order desired is > 4, the most
    exact computation is called (_disrw())
    
    Note: The approximation implementation chooses readibility over speed,
          and therefore the full implementation (order  > 4) is about twice
          as fast as order = 4.
  
    Programmer:     Hamid Mokhtarzadeh
    Created:        April 23, 2012
    Last Modified:  August 18, 2012
    """
    
    if order > 4:
        # Use full computation without approximation.
        return  _disrw(F, G, dt, Q)
    
    if  order >= 1:
        Q_k = G*Q*G.T*dt    
    
    if order >= 2:
        Q_k = Q_k + 0.5*(G*Q*G.T*F.T + F*G*Q*G.T)*dt**2
    
    if order >= 3:
        c = (1.0 / 3.0)
        Q_k = Q_k + c * ( 0.5 * (G*Q*G.T*F.T**2 + F**2*G*Q*G.T) \
                         + F*G*Q*G.T*F.T )*dt**3
    
    if order >= 4:
        c1 = 0.25
        c2 = 0.5
        c3 = 1.0/6.0
        Q_k = Q_k + c1 * ( c3*(  G*Q*G.T*F.T**3 + F**2*G*Q*G.T    ) \
                         + c2*(F*G*Q*G.T*F.T**2 + F**2*G*Q*G.T*F.T) ) *dt**4
    
    return Q_k

def _disrw(F, G, dt, Q):    
    """
    This computes the discrete process noise without the approximations.
    This function was written as a conversion of disrw.m written by 
    Franklin, Powell and Workman described in [1].              

       [1]   G. F. Franklin, J. D. Powell and Workman,      
             Digital Control of Dynamic Systems, 2nd Edition.
    
    Call this function via discrete_process_noise() by specifying an order> 4.
    
    Programmer:     Hamid Mokhtarzadeh
    Last Modified:  August 18, 2012
    """    
    r, c = shape(F)  # Should be square.
    ZF = zeros((r,r))
    
    n, m = shape(G)
    
    row0 = hstack((-F, G*Q*G.T))
    row1 = hstack((ZF, F.T))
    ME = vstack((row0, row1))
    
    phi = matrix(expm(ME * dt), dtype=float)
    
    phi12 = phi[0:n, n:2*n]
    phi22 = phi[n:2*n, n:2*n]
    
    return phi22.T * phi12

def discreteInput(A,B,dt):
    """ zero-order hold 
            Continuous-time: x_dot = Ax + Lu
        Discrete -time: x(k) = Phi*x(k-1) + L(k)*u(k)
    """
    Phi = F2Phi(A,dt)
    
    return Phi*B*dt

def F2Phi(A,dt):
    """ Return approximate discrete-time equivalent state-transition matrix
        for 
        Continuous-time: x_dot = Ax + Lu
        Discrete -time: x(k) = Phi*x(k-1) + L(k)*u(k)
        
        Uses Scipy's matrix exponential based onPade approximation.
        A first order approximation would be: eye(len(A)) + A*dt
    """
    return expm(A*dt)
   
def cov2corr(P):
    """ 
    Return correlation matrix associated with input covariance matrix.
    
    Parameters
    ----------
    P : symmetric positive definite covariance array or matrix, shape (M, M)
    
    Returns
    -------
    corr: symmetric correlation matrix (type array), shape (M, M)
    
    Notes
    -----
    Correlation is defined as corr(i,j) = P(i,j) / sqrt(P(i,i) * P(j,j))
    The returned corr will ALWAYS be of type `array`.
    
    Reference
    ---------
    "Converting Between Correlation and Covariance Matrices"
    Post by Rick Wicklin, December 10 2010
    http://proc-x.com/2010/12/converting-between-correlation-and-covariance-matrices/
    """
    
    # Standard deviation of each state
    D = np.diag(np.sqrt(np.diag(P)))
    Dinv = np.linalg.inv(D)
    
    # Compute correlation
    corr = np.dot(Dinv, np.dot(P, Dinv))
    
    return corr
